# -*- coding: utf-8 -*-
"""
LLM Agent for House HVAC Energy Management (0429/house_energy)
==============================================================
실제 LLM (Ollama / OpenAI) tool-calling으로 건물 에너지 최적화.

전체 흐름:
  1. inspect_scenario  → 시나리오 설명 + FMU 파라미터 구조 이해
  2. query_rag         → 도메인 용어 + 과거 trial prior 검색 (G4)
  3. run_simulation    → house_sim.py 직접 호출 → risk 지표
  4. add_to_rag        → trial 결과 RAG DB 저장
  5. save_finding      → LLM 분석 인사이트 기록
  6. generate_report   → LLM이 직접 Markdown 보고서 작성
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = Path(__file__).resolve().parent
ROOT_ENV = ROOT.parents[1]
AGENT_ROOT = ROOT.parent / "energy_llm_agent"  # RAG 인프라 공유

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_ENV / ".env")
except ImportError:
    pass

from openai import OpenAI

_BASE_URL  = os.getenv("OPENAI_BASE_URL", "").strip()
_API_KEY   = os.getenv("OPENAI_API_KEY", "ollama").strip()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"

client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY) if _BASE_URL else OpenAI()

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# MCP Tool 스키마
# ══════════════════════════════════════════════════════════════════════════════

TOOL_INSPECT_SCENARIO = {
    "type": "function",
    "function": {
        "name": "inspect_scenario",
        "description": (
            "Get the full description of the House HVAC 7-FMU co-simulation scenario: "
            "FMU structure, all tunable parameters with ranges and units, "
            "and the risk metric definition. Call this first."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

TOOL_QUERY_RAG = {
    "type": "function",
    "function": {
        "name": "query_rag",
        "description": (
            "Search RAG knowledge base for domain vocabulary and past trial results. "
            "mode='vocab' → thermal engineering terms mapped to FMU variables. "
            "mode='trials' → past parameter configs with their risk scores. "
            "mode='both' → all knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "mode":  {"type": "string", "enum": ["vocab", "trials", "both"]},
            },
            "required": ["query"],
        },
    },
}

TOOL_RUN_SIMULATION = {
    "type": "function",
    "function": {
        "name": "run_simulation",
        "description": (
            "Run the 7-FMU HVAC co-simulation with the given parameters and return metrics. "
            "Returns: risk_score (lower=better), comfort_rate (higher=better), "
            "mean_power_w, mean_temp_room1_c, mean_temp_room2_c. "
            "Parameters must be physically justified — do not random guess.\n"
            "Example: {'controller': {'T_heatStart': 19.0, 'OvenHeatTransfer': 6.0}, "
            "'outer_wall1': {'k_outsidewall': 0.002}}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Dict of {fmu_name: {param: value}}. FMU names: controller, room1, room2, inner_wall, outer_wall1, outer_wall2",
                },
                "n_trials": {"type": "integer", "description": "Repeated trials for statistics (default 2)"},
            },
            "required": ["params"],
        },
    },
}

TOOL_ADD_TO_RAG = {
    "type": "function",
    "function": {
        "name": "add_to_rag",
        "description": "Store trial result in RAG for future sessions.",
        "parameters": {
            "type": "object",
            "properties": {
                "trial_id": {"type": "string"},
                "params":   {"type": "object"},
                "result":   {"type": "object"},
            },
            "required": ["trial_id", "params", "result"],
        },
    },
}

TOOL_SAVE_FINDING = {
    "type": "function",
    "function": {
        "name": "save_finding",
        "description": "Record an analysis insight for the final report.",
        "parameters": {
            "type": "object",
            "properties": {"finding": {"type": "string"}},
            "required": ["finding"],
        },
    },
}

TOOL_GENERATE_REPORT = {
    "type": "function",
    "function": {
        "name": "generate_report",
        "description": (
            "Write the final optimization report. Call after finding the optimal config. "
            "Include analysis of why these parameters achieve energy efficiency + comfort."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title":          {"type": "string"},
                "optimal_params": {"type": "object"},
                "optimal_result": {"type": "object"},
                "analysis":       {"type": "string", "description": "1-3 paragraph analysis"},
            },
            "required": ["title", "optimal_params", "optimal_result", "analysis"],
        },
    },
}

ALL_TOOLS = [
    TOOL_INSPECT_SCENARIO, TOOL_QUERY_RAG, TOOL_RUN_SIMULATION,
    TOOL_ADD_TO_RAG, TOOL_SAVE_FINDING, TOOL_GENERATE_REPORT,
]
G2_TOOLS = [TOOL_INSPECT_SCENARIO, TOOL_RUN_SIMULATION, TOOL_SAVE_FINDING, TOOL_GENERATE_REPORT]
G3_TOOLS = G2_TOOLS


# ══════════════════════════════════════════════════════════════════════════════
# Tool 실행기
# ══════════════════════════════════════════════════════════════════════════════

_findings: list[str] = []
_trial_history: list[dict] = []

SCENARIO_DESC = {
    "name": "House HVAC Energy Management (7-FMU OSP)",
    "fmus": {
        "Clock":         "Provides simulation time signal → T_clock to TempController",
        "TempController": "Bang-bang thermostat. Inputs: T_room1, T_room2, T_clock. Outputs: h_room1, h_room2 (heater power [W])",
        "Room1":         "Thermal model of room 1. Input: h_powerHeater [W]. Output: T_room [°C]",
        "Room2":         "Thermal model of room 2. Same as Room1.",
        "InnerWall":     "Thermal coupling between rooms. k [W/K] controls heat flow between them.",
        "OuterWall1":    "Outer wall of room 1. k_outsidewall [W/K] = insulation. Lower = better insulated.",
        "OuterWall2":    "Outer wall of room 2. Same as OuterWall1.",
    },
    "tunable_params": {
        "controller": {
            "OvenHeatTransfer": {"min": 2.0,  "max": 18.0, "unit": "W",    "desc": "Heater on-power. Lower = more energy efficient but slower warmup."},
            "T_heatStart":      {"min": 15.0, "max": 20.0, "unit": "°C",   "desc": "Heater turns ON below this threshold."},
            "T_heatStop":       {"min": 20.0, "max": 24.0, "unit": "°C",   "desc": "Heater turns OFF above this. Deadband = T_heatStop - T_heatStart."},
            "transferTime":     {"min": 5.0,  "max": 60.0, "unit": "s",    "desc": "Thermal response time constant."},
        },
        "room1":  {"Tinit_Room1": {"min": 10.0, "max": 18.0, "unit": "°C", "desc": "Initial room 1 temperature."}},
        "room2":  {"Tinit_Room2": {"min": 10.0, "max": 18.0, "unit": "°C", "desc": "Initial room 2 temperature."}},
        "inner_wall":  {"k":              {"min": 0.001, "max": 0.02,  "unit": "W/K", "desc": "Inner wall conductance. Lower = rooms thermally decoupled."}},
        "outer_wall1": {"k_outsidewall":  {"min": 0.001, "max": 0.015, "unit": "W/K", "desc": "Outer wall 1 insulation. Lower = less heat loss to outside (5°C)."}},
        "outer_wall2": {"k_outsidewall":  {"min": 0.001, "max": 0.015, "unit": "W/K", "desc": "Outer wall 2 insulation."}},
    },
    "objective": "Minimize risk = energy_ratio + comfort_violation + (1−success)×0.3",
    "metrics": {
        "risk_score":    "energy_ratio + comfort_violation + (1-success)*0.3  (lower=better)",
        "energy_ratio":  "mean_heater_power / (2×20W)  (lower=more efficient)",
        "comfort_rate":  "fraction of 30-min simulation where BOTH rooms in 18-24°C  (higher=better)",
        "success":       "1 if comfort_rate >= 0.5 else 0",
    },
    "hint": (
        "The thermostat heats whichever room is colder. With C=1 J/K thermal capacity, "
        "heater cycles rapidly. Key trade-off: wider T_heatStop-T_heatStart deadband "
        "→ less cycling → less energy, but risk of short comfort windows. "
        "Lower outer wall k → better insulation → rooms stay warm longer without heating. "
        "Starting rooms near T_heatStart (e.g. 17-18°C) minimises initial heat-up energy."
    ),
}


def _exec_inspect_scenario(_args: dict) -> str:
    return json.dumps(SCENARIO_DESC, ensure_ascii=False, indent=2)


def _exec_query_rag(args: dict) -> str:
    try:
        from rag_store import RagStore
        store = RagStore(db_path=AGENT_ROOT / "rag_db")
        result: dict = {"query": args["query"]}
        mode = args.get("mode", "both")
        top_k = args.get("top_k", 5)
        if mode in ("vocab", "both"):
            result["vocab_hits"] = store.query_vocab(args["query"], n=top_k)
        if mode in ("trials", "both"):
            result["trial_hits"] = store.query_trials(args["query"], n=top_k, max_risk=1.5)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _exec_run_simulation(args: dict) -> str:
    from house_sim import HouseCoSimRunner
    runner = HouseCoSimRunner()
    params = args.get("params", {})
    n = args.get("n_trials", 2)
    result = runner.run_n_trials(override_params=params, n=n)
    _trial_history.append({"params": params, "result": result})
    return json.dumps({k: v for k, v in result.items() if k != "trials"}, ensure_ascii=False, indent=2)


def _exec_add_to_rag(args: dict) -> str:
    try:
        from rag_store import RagStore
        store = RagStore(db_path=AGENT_ROOT / "rag_db")
        store.add_trial(args["trial_id"], args["params"], args["result"], group="llm_house")
        return json.dumps({"status": "ok", "trial_id": args["trial_id"]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _exec_save_finding(args: dict) -> str:
    _findings.append(args.get("finding", ""))
    return json.dumps({"saved": True, "n_findings": len(_findings)})


def _exec_generate_report(args: dict) -> str:
    title         = args.get("title", "HVAC LLM Agent Report")
    optimal_params = args.get("optimal_params", {})
    optimal_result = args.get("optimal_result", {})
    analysis       = args.get("analysis", "")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "reports" / f"llm_agent_report_{ts}.md"
    out.parent.mkdir(exist_ok=True)

    lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        "",
        analysis,
        "",
        "## Optimal Parameters",
        "",
        "```json",
        json.dumps(optimal_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Optimal Result",
        f"- **Risk Score:** {optimal_result.get('risk_score', 'N/A')}",
        f"- **Comfort Rate:** {optimal_result.get('mean_comfort_rate', optimal_result.get('comfort_rate', 'N/A'))}",
        f"- **Mean Power:** {optimal_result.get('mean_power_kw', 'N/A')} kW",
        "",
        "## LLM Findings",
        "",
    ]
    for i, f in enumerate(_findings, 1):
        lines.append(f"**{i}.** {f}")
        lines.append("")

    lines += ["## Experiment Trajectory", "", "| # | risk | comfort | key params |", "|---|---|---|---|"]
    for i, t in enumerate(_trial_history, 1):
        r  = t["result"].get("risk_score", "?")
        cr = t["result"].get("mean_comfort_rate", t["result"].get("comfort_rate", "?"))
        flat = [(f"{k}.{p}", v) for k, ps in t["params"].items() for p, v in ps.items()]
        kp = ", ".join(f"{n}={v}" for n, v in flat[:3]) if flat else "default"
        lines.append(f"| {i} | {r} | {cr} | {kp} |")

    lines += [
        "",
        "## RAG Contribution",
        "- Domain vocabulary mapped 'thermal comfort', 'heating deadband', 'insulation' → FMU variables",
        "- Prior trial results provided warm-start parameter recommendations",
        "- This session's results stored for future agent runs",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return json.dumps({"status": "report_generated", "path": str(out)}, ensure_ascii=False)


_DISPATCH = {
    "inspect_scenario": _exec_inspect_scenario,
    "query_rag":        _exec_query_rag,
    "run_simulation":   _exec_run_simulation,
    "add_to_rag":       _exec_add_to_rag,
    "save_finding":     _exec_save_finding,
    "generate_report":  _exec_generate_report,
}


def execute_tool(name: str, args: dict) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(args)
    except Exception as e:
        return json.dumps({"error": f"{name} failed: {e}"})


# ══════════════════════════════════════════════════════════════════════════════
# Reflection 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

REFLECTION_PROMPT = """\
[REFLECTION CHECKPOINT — {n} simulations completed]

Pause and reflect:
1. **Energy vs Comfort Trade-off**: Which parameters most affected risk_score?
2. **Pattern**: Is there a clear direction (wider deadband, lower k, etc.)?
3. **RAG Alignment**: Did prior trial suggestions help or mislead?
4. **Strategy**: Exploit (refine current best) or explore (new region)?
5. **Next hypothesis**: Specific next parameter set with physical justification.

Continue after reflection.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 에이전트 루프
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(
    mode: str = "G4",
    max_iterations: int = 35,
    reflection_interval: int = 3,
    verbose: bool = True,
    llm_seed: int | None = None,
) -> dict:
    global _findings, _trial_history
    _findings.clear()
    _trial_history.clear()

    tools = ALL_TOOLS if mode == "G4" else G2_TOOLS if mode == "G2" else G3_TOOLS
    rag_active = (mode == "G4")

    system_prompt = f"""\
You are an expert LLM agent for building HVAC energy management optimization.

SCENARIO: 7-FMU OSP House Co-simulation (30-minute, 1-second step)
  - TempController: bang-bang thermostat controlling 2 room heaters
  - Room1, Room2: thermal models (C=1 J/K, fast cycling)
  - InnerWall, OuterWall1/2: thermal coupling and insulation
  - Outside temperature: 5°C (cold winter day)

GOAL: Minimize risk_score = energy_ratio + comfort_violation + (1-success)×0.3
  - energy_ratio  = mean_heater_power / 40W  (keep small → efficient)
  - comfort_rate  = fraction of time BOTH rooms in 18-24°C  (maximize)
  - success       = 1 if comfort_rate ≥ 0.5
  Target: risk < 0.15 (achievable — best known is ~0.008)

MODE: {mode}
STEP 1: Call inspect_scenario() to understand parameters and physics
{"STEP 2: Call query_rag(mode='both') to get domain knowledge and prior best params" if rag_active else "STEP 2: [RAG disabled — reason from domain knowledge only]"}
STEP 3: Propose physically-justified parameters and call run_simulation()
STEP 4: Analyze results — adjust based on comfort_rate and energy_ratio
{"STEP 4b: Call add_to_rag() to store each trial result" if rag_active else ""}
STEP 5: Use save_finding() to record key insights
STEP 6: After finding optimal config (risk < 0.15), call generate_report()

PHYSICS:
  - Lower OvenHeatTransfer → less peak power, slower warmup
  - Narrower deadband (T_heatStop - T_heatStart ≈ 4°C) → more cycling, same comfort
  - Wider deadband (6-7°C) → less cycling, but comfort zone harder to maintain
  - Lower k_outsidewall → better insulation → rooms retain heat longer (less heater use)
  - Start Tinit close to T_heatStart → minimal warm-up energy

Start by calling inspect_scenario(), then optimize systematically.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Begin HVAC optimization in {mode} mode. "
            "First inspect the scenario, "
            f"{'then query RAG for prior knowledge, ' if rag_active else ''}"
            "then run simulations to minimize risk_score. Target risk < 0.15."
        )},
    ]

    n_simulations = 0
    n_reflections = 0
    n_llm_calls   = 0
    done_reason   = "max_iterations"
    report_path   = ""
    min_sims_before_report = 12 if mode == "G4" else 5

    if verbose:
        print(f"\n{'='*60}")
        print(f"  HVAC LLM Agent [{mode}]  model={MODEL_NAME}")
        print(f"{'='*60}")

    for _iter in range(max_iterations):
        try:
            req = dict(model=MODEL_NAME, messages=messages, tools=tools,
                       tool_choice="auto", temperature=0.3,
                       max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")))
            if llm_seed is not None:
                req["seed"] = llm_seed
            resp = client.chat.completions.create(**req)
        except Exception as e:
            print(f"[AGENT] LLM error: {e}", file=sys.stderr)
            break

        n_llm_calls += 1
        msg = resp.choices[0].message

        if not msg.tool_calls:
            final = msg.content or ""
            if verbose:
                print(f"\n[FINAL]\n{final[:400]}")
            messages.append({"role": "assistant", "content": final})
            done_reason = "natural_completion"
            break

        messages.append(msg.model_dump())
        needs_reflection = False

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"\n[TOOL] {name}({json.dumps(args, ensure_ascii=False)[:120]})")

            result_str = execute_tool(name, args)

            if verbose:
                print(f"  → {result_str[:250]}")

            if name == "run_simulation":
                try:
                    r = json.loads(result_str)
                    if not r.get("error"):
                        n_simulations += 1
                        if n_simulations % reflection_interval == 0 and mode in ("G3", "G4"):
                            needs_reflection = True
                except Exception:
                    pass

            if name == "generate_report":
                if n_simulations < min_sims_before_report:
                    result_str = json.dumps({
                        "error": (
                            f"Too early to report: {n_simulations} simulations completed, "
                            f"need at least {min_sims_before_report}."
                        )
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str[:10000],
                    })
                    continue
                try:
                    r = json.loads(result_str)
                    report_path = r.get("path", "")
                except Exception:
                    pass
                done_reason = "report_generated"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str[:10000],
            })

        if needs_reflection:
            n_reflections += 1
            messages.append({"role": "user", "content": REFLECTION_PROMPT.format(n=n_simulations)})
            if verbose:
                print(f"\n[REFLECTION #{n_reflections}]")

        if done_reason == "report_generated":
            break

    if verbose:
        print(f"\n[DONE] mode={mode} sims={n_simulations} reflections={n_reflections} "
              f"llm_calls={n_llm_calls} reason={done_reason}")

    return {
        "n_simulations": n_simulations,
        "n_reflections": n_reflections,
        "n_llm_calls":   n_llm_calls,
        "done_reason":   done_reason,
        "report_path":   report_path,
        "n_findings":    len(_findings),
        "trial_history": _trial_history,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="G4", choices=["G2", "G3", "G4"])
    ap.add_argument("--max-iter", type=int, default=35)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    result = run_agent(mode=args.mode, max_iterations=args.max_iter, llm_seed=args.seed)
    best_risks = [t["result"].get("risk_score", 9999) for t in result["trial_history"]]
    if best_risks:
        print(f"\n[BEST RISK] {min(best_risks):.4f} (over {len(best_risks)} trials)")
