# -*- coding: utf-8 -*-
"""
LLM Agent for FMU Co-simulation (0429 energy_llm_agent)
========================================================
실제 LLM (Ollama / OpenAI) tool-calling으로 FMU 파라미터 최적화.

전체 흐름 (G4 — 제안 방법):
  1. inspect_fmu        → FMU 변수/구조 이해
  2. query_rag          → 도메인 용어 + 과거 trial prior 검색
  3. generate_cosim_code → codegen.py로 co-sim 러너 스크립트 생성
  4. run_simulation     → 생성된 스크립트 실행 → risk 지표 반환
  5. add_to_rag         → trial 결과 RAG DB에 저장 (온라인 학습)
  6. save_finding       → LLM 분석 인사이트 기록
  7. generate_report    → LLM이 직접 Markdown 보고서 작성

G2 mode (LLM-only)      : query_rag / add_to_rag 비활성화
G3 mode (Reflection)    : RAG 없이 Reflection 체크포인트 활성화
G4 mode (RAG+Reflection): 모든 tool 활성화 + warm-start
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
ROOT_ENV = ROOT.parents[1]

# ── .env 로드 ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_ENV / ".env")
except ImportError:
    pass

from openai import OpenAI

_BASE_URL  = os.getenv("OPENAI_BASE_URL", "").strip()
_API_KEY   = os.getenv("OPENAI_API_KEY", "ollama").strip()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"

if _BASE_URL:
    client = OpenAI(base_url=_BASE_URL, api_key=_API_KEY or "ollama")
else:
    client = OpenAI()

sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# MCP Tool 스키마 (OpenAI function-calling 형식)
# ══════════════════════════════════════════════════════════════════════════════

TOOL_INSPECT_FMU = {
    "type": "function",
    "function": {
        "name": "inspect_fmu",
        "description": (
            "Inspect an FMU to get its variables, causality (input/output/parameter), "
            "and tunable parameter ranges. Call this first to understand each FMU's role "
            "and what can be optimized."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fmu_name": {
                    "type": "string",
                    "description": "FMU name, e.g. 'linearized_wave_model', 'winch', 'power_system'",
                },
            },
            "required": ["fmu_name"],
        },
    },
}

TOOL_QUERY_RAG = {
    "type": "function",
    "function": {
        "name": "query_rag",
        "description": (
            "Search the RAG knowledge base for: (1) domain vocabulary — FMU variable descriptions "
            "and physical meanings; (2) past trial results — parameters that achieved low risk. "
            "Use this to warm-start parameter search and understand physical semantics. "
            "mode='vocab' for domain terms, 'trials' for past experiments, 'both' for all."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query, e.g. 'low risk winch K_p wave height reduction'",
                },
                "top_k": {"type": "integer", "description": "Number of results (default 5)"},
                "mode":  {"type": "string", "enum": ["vocab", "trials", "both"], "description": "Search mode"},
            },
            "required": ["query"],
        },
    },
}

TOOL_GENERATE_CODE = {
    "type": "function",
    "function": {
        "name": "generate_cosim_code",
        "description": (
            "Generate a Python co-simulation runner script for the given scenario using "
            "FMPy. The generated script handles FMU loading, parameter injection, "
            "and result collection. This is the CODE GENERATION step — call this before "
            "running simulations to ensure the runner is up-to-date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scenario_id": {
                    "type": "string",
                    "description": "Scenario ID, e.g. 'construction_vessel_full'",
                },
                "profile": {
                    "type": "string",
                    "description": "Simulation difficulty profile: 'paper' (challenging) or 'default'",
                    "enum": ["paper", "default"],
                },
            },
            "required": ["scenario_id"],
        },
    },
}

TOOL_RUN_SIMULATION = {
    "type": "function",
    "function": {
        "name": "run_simulation",
        "description": (
            "Run the FMU co-simulation with specified parameters (N trials) and return "
            "risk metrics. Returns risk_score (lower=better), success_rate, mean_power_kw, "
            "energy_j. Propose parameters based on your domain analysis — do NOT random guess."
            "Example: {'winch': {'K_p': 350.0, 'K_d': 4.0}, 'wave_model': {'wave_height': 2.5}}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Dict of {fmu_name: {param_name: value}}. Only specify FMUs/params to override.",
                },
                "n_trials": {
                    "type": "integer",
                    "description": "Number of repeated trials for statistical reliability (default 3)",
                },
            },
            "required": ["params"],
        },
    },
}

TOOL_ADD_TO_RAG = {
    "type": "function",
    "function": {
        "name": "add_to_rag",
        "description": (
            "Store a trial result in the RAG knowledge base for future use. "
            "Call this after every simulation trial so knowledge accumulates across runs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "trial_id":  {"type": "string", "description": "Unique trial identifier, e.g. 'llm_g4_trial_01'"},
                "params":    {"type": "object", "description": "The parameters used in this trial"},
                "result":    {"type": "object", "description": "The simulation result dict (risk_score, success_rate, etc.)"},
            },
            "required": ["trial_id", "params", "result"],
        },
    },
}

TOOL_SAVE_FINDING = {
    "type": "function",
    "function": {
        "name": "save_finding",
        "description": "Save an analysis insight or hypothesis. These appear in the final report as evidence of LLM reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "finding": {"type": "string", "description": "The insight to record (be specific, cite evidence)"},
            },
            "required": ["finding"],
        },
    },
}

TOOL_GENERATE_REPORT = {
    "type": "function",
    "function": {
        "name": "generate_report",
        "description": (
            "Generate a comprehensive Markdown optimization report. "
            "Include: goal, FMU structure analysis, RAG findings, "
            "parameter search trajectory, optimal configuration, and conclusions. "
            "Call this as the FINAL step after finding the optimal parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title":          {"type": "string", "description": "Report title"},
                "optimal_params": {"type": "object", "description": "The best parameter configuration found"},
                "optimal_result": {"type": "object", "description": "The best simulation result"},
                "analysis":       {"type": "string", "description": "LLM's analysis of findings and conclusions (1-3 paragraphs)"},
            },
            "required": ["title", "optimal_params", "optimal_result", "analysis"],
        },
    },
}

# G4: all tools
ALL_TOOLS = [
    TOOL_INSPECT_FMU, TOOL_QUERY_RAG, TOOL_GENERATE_CODE,
    TOOL_RUN_SIMULATION, TOOL_ADD_TO_RAG, TOOL_SAVE_FINDING, TOOL_GENERATE_REPORT,
]

# G2 (LLM-only): no RAG tools
G2_TOOLS = [
    TOOL_INSPECT_FMU, TOOL_GENERATE_CODE,
    TOOL_RUN_SIMULATION, TOOL_SAVE_FINDING, TOOL_GENERATE_REPORT,
]

# G3 (LLM + Reflection, no RAG): same as G2
G3_TOOLS = G2_TOOLS


# ══════════════════════════════════════════════════════════════════════════════
# Tool 실행기 (LLM 호출 → 실제 Python 함수 매핑)
# ══════════════════════════════════════════════════════════════════════════════

_findings: list[str] = []
_trial_history: list[dict] = []
_generated_code_path: str = ""


def _exec_inspect_fmu(args: dict) -> str:
    from mcp_toolset import fmu_inspector
    fmu_name = args["fmu_name"]
    # 시나리오에서 FMU 경로 찾기
    sys.path.insert(0, str(ROOT))
    from scenarios import get_scenario
    try:
        sc = get_scenario("construction_vessel_full")
        fmu_path = None
        for f in sc.get("fmus", []):
            if f["fmu"] == fmu_name or f["name"] == fmu_name:
                case = f.get("case", "construction-vessel")
                demo = ROOT.parents[1] / "0429" / "demo-cases" / case / "fmus" / f"{f['fmu']}.fmu"
                if demo.exists():
                    fmu_path = str(demo)
                    break
        if fmu_path is None:
            # tunable_params 정보라도 반환
            for f in sc.get("fmus", []):
                if f["fmu"] == fmu_name or f["name"] == fmu_name:
                    return json.dumps({
                        "fmu": fmu_name,
                        "role": f.get("role", ""),
                        "tunable_params": f.get("tunable_params", {}),
                        "default_params": f.get("default_params", {}),
                    }, ensure_ascii=False, indent=2)
            return json.dumps({"error": f"FMU '{fmu_name}' not found in scenario"})
        result = fmu_inspector(fmu_path)
        # tunable_params 추가
        for f in sc.get("fmus", []):
            if f["fmu"] == fmu_name or f["name"] == fmu_name:
                result["tunable_params"] = f.get("tunable_params", {})
                result["default_params"] = f.get("default_params", {})
                result["role"] = f.get("role", "")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _exec_query_rag(args: dict) -> str:
    from mcp_toolset import rag_retriever
    result = rag_retriever(
        query=args["query"],
        top_k=args.get("top_k", 5),
        mode=args.get("mode", "both"),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def _exec_generate_code(args: dict) -> str:
    global _generated_code_path
    from codegen import generate_cosim_runner, load_config
    scenario_id = args.get("scenario_id", "construction_vessel_full")
    profile = args.get("profile", "paper")

    cfg_path = ROOT / "configs" / "construction_vessel_full.json"
    if not cfg_path.exists():
        cfg = {"scenario_id": scenario_id, "profile": profile}
    else:
        cfg = load_config(cfg_path)
        cfg["profile"] = profile

    out = ROOT / "generated" / "generated_cosim_runner.py"
    out.parent.mkdir(exist_ok=True)
    generate_cosim_runner(cfg, out)
    _generated_code_path = str(out)
    return json.dumps({
        "status": "code_generated",
        "output_path": str(out),
        "scenario_id": scenario_id,
        "profile": profile,
        "description": (
            f"Generated FMPy co-simulation runner for '{scenario_id}' scenario. "
            f"The script loads all FMUs, applies override_params, runs N trials, "
            f"and returns risk metrics as JSON."
        ),
    }, ensure_ascii=False, indent=2)


def _exec_run_simulation(args: dict) -> str:
    from mcp_toolset import simulation_runner
    params = args.get("params", {}) or {}
    # Normalize FMU aliases so LLM naming variance does not silently drop overrides.
    if "linearized_wave_model" in params and "wave_model" not in params:
        params["wave_model"] = params.pop("linearized_wave_model")
    if "dp_reference_model" in params and "reference_model" not in params:
        params["reference_model"] = params.pop("dp_reference_model")
    n_trials = args.get("n_trials", 3)
    result = simulation_runner(params=params, n_trials=n_trials)
    _trial_history.append({"params": params, "result": result})
    return json.dumps(result, ensure_ascii=False, indent=2)


def _exec_add_to_rag(args: dict) -> str:
    from mcp_toolset import rag_add_trial
    result = rag_add_trial(
        trial_id=args["trial_id"],
        params=args["params"],
        result=args["result"],
        group="llm_agent",
    )
    return json.dumps(result, ensure_ascii=False)


def _exec_save_finding(args: dict) -> str:
    finding = args.get("finding", "")
    _findings.append(finding)
    return json.dumps({"saved": True, "n_findings": len(_findings)})


def _exec_generate_report(args: dict) -> str:
    from mcp_toolset import report_generator

    title = args.get("title", "LLM Agent Optimization Report")
    optimal_params = args.get("optimal_params", {})
    optimal_result = args.get("optimal_result", {})
    analysis = args.get("analysis", "")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "reports" / f"llm_agent_report_{ts}.md"
    result = report_generator(
        title=title,
        goal="Minimize vessel risk_score while maintaining high success_rate under the 420 kW power limit.",
        optimal_params=optimal_params,
        result=optimal_result,
        analysis=analysis,
        out_path=str(out_path),
        findings=list(_findings),
        trial_history=list(_trial_history),
        generated_code_path=_generated_code_path or "generated/generated_cosim_runner.py",
    )
    return json.dumps({
        "status": "report_generated",
        "path": result["report_path"],
        "n_findings": len(_findings),
        "n_trials_logged": len(_trial_history),
    }, ensure_ascii=False)

    title = args.get("title", "LLM Agent Optimization Report")
    optimal_params = args.get("optimal_params", {})
    optimal_result = args.get("optimal_result", {})
    analysis = args.get("analysis", "")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "reports" / f"llm_agent_report_{ts}.md"
    out_path.parent.mkdir(exist_ok=True)

    lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        "",
        analysis,
        "",
        "## Optimal Parameters Found",
        "",
        "```json",
        json.dumps(optimal_params, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Optimal Simulation Result",
        "",
        f"- **Risk Score:** {optimal_result.get('risk_score', 'N/A')}",
        f"- **Success Rate:** {optimal_result.get('success_rate', 'N/A')}",
        f"- **Mean Power:** {optimal_result.get('mean_power_kw', 'N/A')} kW",
        f"- **Energy:** {optimal_result.get('energy_j', 'N/A')} J",
        "",
        "## LLM Analysis Findings",
        "",
    ]
    for i, f in enumerate(_findings, 1):
        lines.append(f"**Finding {i}:** {f}")
        lines.append("")

    lines += [
        "## Experiment Trajectory",
        "",
        "| Trial | Risk | Success Rate | Key Params |",
        "|---|---|---|---|",
    ]
    for i, t in enumerate(_trial_history, 1):
        risk = t["result"].get("risk_score", "N/A")
        sr   = t["result"].get("success_rate", "N/A")
        # 가장 중요한 파라미터 1개
        flat = [(f"{k}.{p}", v) for k, ps in t["params"].items() for p, v in ps.items()]
        key_p = ", ".join(f"{n}={v}" for n, v in flat[:2]) if flat else "default"
        lines.append(f"| {i} | {risk} | {sr} | {key_p} |")

    lines += [
        "",
        "## Code Generation",
        "",
        f"Generated runner: `{_generated_code_path or 'generated/generated_cosim_runner.py'}`",
        "",
        "## RAG Contribution",
        "",
        "- Domain vocabulary enabled semantic mapping of engineering terms to FMU variables.",
        "- Prior trial results provided warm-start parameter recommendations.",
        "- Online learning stored this run's results for future agent sessions.",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return json.dumps({
        "status": "report_generated",
        "path": str(out_path),
        "n_findings": len(_findings),
        "n_trials_logged": len(_trial_history),
    }, ensure_ascii=False)


_DISPATCH = {
    "inspect_fmu":       _exec_inspect_fmu,
    "query_rag":         _exec_query_rag,
    "generate_cosim_code": _exec_generate_code,
    "run_simulation":    _exec_run_simulation,
    "add_to_rag":        _exec_add_to_rag,
    "save_finding":      _exec_save_finding,
    "generate_report":   _exec_generate_report,
}


def execute_tool(name: str, args: dict) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(args)
    except Exception as e:
        return json.dumps({"error": f"Tool {name} failed: {e}"})


# ══════════════════════════════════════════════════════════════════════════════
# Reflection 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

REFLECTION_PROMPT = """\
[REFLECTION CHECKPOINT — {n} simulations completed]

Pause and perform a structured self-reflection:

1. **FMU Attribution**: Which FMU/parameter has the highest impact on risk_score so far?
2. **Trend**: What pattern did you observe? (monotonic, non-monotonic, saturation, interaction)
3. **RAG Insight**: Did the RAG-retrieved priors align with empirical results?
4. **Strategy**: Should you exploit (refine current best) or explore (try new region)?
5. **Next hypothesis**: State your specific next parameter combination and physical rationale.

After reflection, continue optimizing. Remember: the goal is minimal risk_score with high success_rate.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 메인 에이전트 루프
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(
    mode: str = "G4",          # "G2", "G3", "G4"
    max_iterations: int = 40,
    reflection_interval: int = 3,
    verbose: bool = True,
    llm_seed: int | None = None,
) -> dict:
    """
    FMU co-simulation LLM 최적화 에이전트.

    Args:
        mode: "G2" (LLM-only), "G3" (LLM+Reflection), "G4" (LLM+RAG+Reflection)
        max_iterations: 최대 LLM 호출 횟수
        reflection_interval: N 시뮬마다 Reflection 체크포인트
        verbose: 진행 상황 출력
        llm_seed: 재현성을 위한 LLM seed

    Returns:
        {final_message, n_simulations, n_reflections, n_llm_calls, done_reason, report_path}
    """
    global _findings, _trial_history, _generated_code_path
    _findings.clear()
    _trial_history.clear()
    _generated_code_path = ""

    # 모드별 tool 선택
    if mode == "G4":
        tools = ALL_TOOLS
        rag_active = True
    elif mode == "G3":
        tools = G3_TOOLS
        rag_active = False
    else:  # G2
        tools = G2_TOOLS
        rag_active = False

    # ── 시스템 프롬프트 ────────────────────────────────────────────────────
    system_prompt = f"""\
You are an expert LLM agent for FMU-based marine energy co-simulation optimization.

SCENARIO: construction_vessel_full (9-FMU OSP co-simulation)
  - linearized_wave_model: wave height, peak_frequency, stiffness coefficients
  - winch: PID controller (K_p, K_d, K_i) for crane load depth control
  - power_system: shipboard power bus (tracks power violations)
  - vessel_model, dp_controller, thrust_allocation, thruster_model, wind_model, ref_model: vessel dynamics

GOAL: Minimize risk_score = mean_vps + std_vps + (1-success_rate)*failure_penalty
  - vps = violation_per_step (power limit exceeded fraction)
  - success_rate = fraction of trials where crane reached target depth within tolerance
  - power_limit = 420 kW (hard constraint)

YOUR PROTOCOL ({mode} mode):
  Step 1: Call inspect_fmu() for key FMUs (wave_model, winch) to understand parameters
  {"Step 2: Call query_rag() to retrieve domain knowledge and prior successful configurations" if rag_active else "Step 2: [RAG disabled in this mode — rely on domain knowledge only]"}
  Step 3: Call generate_cosim_code() to create the simulation runner script
  Step 4: Propose physically-motivated parameters and call run_simulation()
  Step 5: Analyze results — if risk improved, exploit nearby; if not, explore new region
  {"Step 5b: Call add_to_rag() after each simulation to build knowledge base" if rag_active else ""}
  Step 6: Use save_finding() to record key insights
  Step 7: After finding optimal config, call generate_report() with full analysis

PHYSICAL INTUITION:
  - Lower wave_height (1-2m) reduces wave disturbance → lower vps
  - K_p ≈ 300-500: too low = slow response, too high = oscillation
  - K_d: damping, typically 3-8x smaller than K_p
  - peak_frequency 0.5-1.0: moderate sea state
  - power_limit violations occur when wave + thruster load exceeds 420 kW

CONSTRAINTS: Tunable ranges are specified in inspect_fmu() results. Never exceed them.

Start by inspecting the key FMUs, then generate code, then optimize systematically.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Begin optimization in {mode} mode. "
            "First inspect the FMUs, then generate the co-sim code, "
            f"{'then query RAG for prior knowledge, ' if rag_active else ''}"
            "then run simulations to find the optimal configuration. "
            "Target risk_score < 1.10."
        )},
    ]

    n_simulations  = 0
    n_reflections  = 0
    n_llm_calls    = 0
    done_reason    = "max_iterations"
    report_path    = ""
    min_sims_before_report = 12 if mode == "G4" else 4

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LLM FMU Agent [{mode}]  model={MODEL_NAME}")
        print(f"{'='*60}")

    for iteration in range(max_iterations):
        # ── LLM 호출 ────────────────────────────────────────────────────────
        try:
            req = dict(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
            )
            if llm_seed is not None:
                req["seed"] = llm_seed
            resp = client.chat.completions.create(**req)
        except Exception as e:
            print(f"[AGENT] LLM error: {e}", file=sys.stderr)
            break

        n_llm_calls += 1
        msg = resp.choices[0].message

        # ── 툴 호출 없음 → 최종 텍스트 ────────────────────────────────────
        if not msg.tool_calls:
            final = msg.content or ""
            if verbose:
                print(f"\n[AGENT FINAL]\n{final[:500]}")
            messages.append({"role": "assistant", "content": final})
            done_reason = "natural_completion"
            break

        # ── 툴 호출 처리 ─────────────────────────────────────────────────
        messages.append(msg.model_dump())

        tool_results = []
        needs_reflection = False
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if verbose:
                _preview = json.dumps(args, ensure_ascii=False)[:100]
                print(f"\n[TOOL] {name}({_preview})")

            result_str = execute_tool(name, args)

            if verbose:
                print(f"  → {result_str[:300]}")

            # 시뮬레이션 카운트
            if name == "run_simulation":
                try:
                    r = json.loads(result_str)
                    if not r.get("error"):
                        n_simulations += 1
                        if n_simulations % reflection_interval == 0 and mode in ("G3", "G4"):
                            needs_reflection = True
                except Exception:
                    pass

            # 보고서 생성 감지
            if name == "generate_report":
                if n_simulations < min_sims_before_report:
                    result_str = json.dumps({
                        "error": (
                            f"Too early to report: {n_simulations} simulations completed, "
                            f"need at least {min_sims_before_report}."
                        )
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                    continue
                try:
                    r = json.loads(result_str)
                    report_path = r.get("path", "")
                    if verbose:
                        print(f"\n[REPORT] → {report_path}")
                except Exception:
                    pass
                done_reason = "report_generated"

            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result_str[:12000],
            })

        for tr in tool_results:
            messages.append(tr)

        # ── Reflection 삽입 ────────────────────────────────────────────────
        if needs_reflection:
            n_reflections += 1
            refl = REFLECTION_PROMPT.format(n=n_simulations)
            messages.append({"role": "user", "content": refl})
            if verbose:
                print(f"\n[REFLECTION #{n_reflections}] after {n_simulations} simulations")

        if done_reason == "report_generated":
            break

    if verbose:
        print(f"\n[DONE] mode={mode}  sims={n_simulations}  "
              f"reflections={n_reflections}  llm_calls={n_llm_calls}  "
              f"reason={done_reason}")

    # 최종 메시지 추출
    final_msg = ""
    for m in reversed(messages):
        role    = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role == "assistant" and isinstance(content, str) and content.strip():
            final_msg = content
            break

    return {
        "final_message":  final_msg,
        "n_simulations":  n_simulations,
        "n_reflections":  n_reflections,
        "n_llm_calls":    n_llm_calls,
        "done_reason":    done_reason,
        "report_path":    report_path,
        "n_findings":     len(_findings),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LLM FMU Co-simulation Agent")
    ap.add_argument("--mode",  default="G4", choices=["G2", "G3", "G4"])
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--seed",    type=int, default=None)
    args = ap.parse_args()

    result = run_agent(
        mode=args.mode,
        max_iterations=args.max_iter,
        llm_seed=args.seed,
        verbose=True,
    )
    print(f"\n[SUMMARY] {json.dumps({k:v for k,v in result.items() if k!='final_message'}, indent=2)}")
