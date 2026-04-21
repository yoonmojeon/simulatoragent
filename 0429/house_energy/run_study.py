"""
run_study.py — House Energy Management G0~G4 비교 실험
=======================================================
Building HVAC 에너지 관리: 7-FMU 코시뮬레이션으로
  TempController + Room1/2 + InnerWall + OuterWall1/2 + Clock 연결

목적: 에너지 소비 최소화 + 쾌적도 유지
Risk = energy_ratio + comfort_violation + (1-sr) * 0.5

G0~G4 ablation:
  G0: Random Search
  G1: Grid Search
  G2: LLM-only (random → local exploit)
  G3: LLM + Reflection (adaptive sigma)
  G4: LLM + RAG + Reflection (proposed)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent
AGENT_ROOT = ROOT.parent / "energy_llm_agent"  # RAG 인프라 재사용

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT_ROOT))

from house_sim import HouseCoSimRunner

CFG = ROOT / "configs" / "house_energy_objective.json"
REP = ROOT / "reports"

GROUPS = ["G0_random", "G1_grid", "G2_llm_only", "G3_reflection", "G4_rag_reflection"]

# ── RAG 싱글턴 ──────────────────────────────────────────────────────────────
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        try:
            from rag_store import RagStore
            _rag = RagStore(db_path=ROOT / "rag_db")
        except Exception as e:
            print(f"[WARN] RAG init failed: {e}")
            _rag = None
    return _rag


# ── 탐색 공간 헬퍼 ──────────────────────────────────────────────────────────
def _load_space(cfg: dict) -> dict:
    return cfg["param_space"]


def _sample_random(space: dict, rng: random.Random) -> dict:
    out: dict = {}
    for fmu, ps in space.items():
        out[fmu] = {}
        for p, spec in ps.items():
            lo, hi = float(spec["min"]), float(spec["max"])
            out[fmu][p] = round(rng.uniform(lo, hi), 4)
    return out


def _flatten_order(space: dict) -> list:
    rows = []
    for fmu, ps in space.items():
        for p, spec in ps.items():
            rows.append((fmu, p, float(spec["min"]), float(spec["max"])))
    return rows


def _sample_grid(space: dict, i: int, budget: int) -> dict:
    frac = (i - 0.5) / max(1, budget)
    dims = _flatten_order(space)
    out: dict = {}
    for d, (fmu, p, lo, hi) in enumerate(dims):
        phase = (frac + d * 0.173) % 1.0
        out.setdefault(fmu, {})[p] = round(lo + phase * (hi - lo), 4)
    return out


def _jitter(base: dict, space: dict, sigma: float, rng: random.Random) -> dict:
    out: dict = {}
    for fmu, ps in space.items():
        out[fmu] = {}
        for p, spec in ps.items():
            lo, hi = float(spec["min"]), float(spec["max"])
            b = float(base.get(fmu, {}).get(p, (lo + hi) / 2.0))
            v = b + rng.gauss(0.0, sigma) * (hi - lo)
            out[fmu][p] = round(max(lo, min(hi, v)), 4)
    return out


def _eval(runner: HouseCoSimRunner, params: dict, n_trials: int) -> dict:
    res = runner.run_n_trials(override_params=params, n=n_trials)
    return {"params": params, "result": res, "risk": float(res.get("risk_score", 9999.0))}


# ── 그룹별 실험 ─────────────────────────────────────────────────────────────
def _run_group(
    group: str,
    runner: HouseCoSimRunner,
    space: dict,
    budget: int,
    n_trials: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    hist: list[dict] = []
    best = None
    sigma = 0.14
    rag = get_rag()

    # G4 RAG warm-start
    warm_params = None
    if group == "G4_rag_reflection" and rag is not None:
        try:
            top = rag.query_trials(
                "low risk energy efficient thermostat temperature comfort building heating",
                n=5, max_risk=1.2,
            )
            if top:
                warm_params = top[0]["params"]
                print(f"  [G4] RAG warm-start: prior risk={top[0]['risk']:.4f}")
            else:
                print("  [G4] no RAG prior (DB empty) — random start")
        except Exception as e:
            print(f"  [G4] RAG warm-start failed: {e}")

    for i in range(1, budget + 1):
        if group == "G0_random":
            cand = _sample_random(space, rng)
        elif group == "G1_grid":
            cand = _sample_grid(space, i, budget)
        elif group == "G2_llm_only":
            if i <= max(2, budget // 3) or best is None:
                cand = _sample_random(space, rng)
            else:
                cand = _jitter(best["params"], space, sigma=0.08, rng=rng)
        elif group == "G3_reflection":
            if best is None:
                cand = _sample_random(space, rng)
            else:
                cand = _jitter(best["params"], space, sigma=sigma, rng=rng)
        else:  # G4
            if i == 1 and warm_params:
                cand = _jitter(warm_params, space, sigma=0.02, rng=rng)
            elif i <= 3 and warm_params and best is None:
                cand = _jitter(warm_params, space, sigma=0.06, rng=rng)
            elif best is None:
                cand = _sample_random(space, rng)
            else:
                cand = _jitter(best["params"], space, sigma=sigma * 0.75, rng=rng)

        rec = _eval(runner, cand, n_trials)
        rec["iter"] = i
        hist.append(rec)

        improved = best is None or rec["risk"] < best["risk"]
        if improved:
            best = rec
            if group in ("G3_reflection", "G4_rag_reflection"):
                sigma = max(0.025, sigma * 0.80)
        else:
            if group in ("G3_reflection", "G4_rag_reflection"):
                sigma = min(0.28, sigma * 1.12)

        # G4 온라인 RAG 업데이트
        if group == "G4_rag_reflection" and rag is not None:
            try:
                rag.add_trial(f"house_g4_s{seed}_i{i:03d}", cand, rec["result"], group=group)
            except Exception:
                pass

        tag = " ✓" if improved else ""
        cr = rec["result"].get("mean_comfort_rate", rec["result"].get("comfort_rate", 0))
        pwr = rec["result"].get("mean_power_kw", 0)
        print(f"  [{group}] i={i:02d}  risk={rec['risk']:.4f}{tag}  comfort={cr:.2f}  pwr={pwr:.2f}kW")

    return hist


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget",   type=int, default=12)
    ap.add_argument("--n-trials", type=int, default=2)
    ap.add_argument("--seeds",    default="11,22,33,44,55")
    ap.add_argument("--groups",   default="all")
    args = ap.parse_args()

    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    space = _load_space(cfg)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    groups = GROUPS if args.groups.lower() == "all" else [
        g for g in GROUPS if g in args.groups.split(",")
    ]

    # RAG vocab 초기화 (house 도메인 추가)
    rag = get_rag()
    if rag is not None:
        try:
            _build_house_vocab(rag)
        except Exception as e:
            print(f"[WARN] house vocab failed: {e}")

    runner = HouseCoSimRunner()
    rows: list[dict] = []
    total = len(seeds) * len(groups)

    print(f"[INFO] House Energy Management — budget={args.budget}  n_trials={args.n_trials}  seeds={seeds}")
    print(f"[INFO] Groups: {groups}")
    print(f"[INFO] Param space: {sum(len(v) for v in space.values())} dimensions")

    for si, sd in enumerate(seeds):
        for gi, g in enumerate(groups):
            run_no = si * len(groups) + gi + 1
            print(f"\n{'='*60}")
            print(f"[RUN {run_no}/{total}] group={g}  seed={sd}")
            print(f"{'='*60}")

            hist = _run_group(g, runner, space,
                              budget=args.budget, n_trials=args.n_trials,
                              seed=sd + gi * 97)
            best = min(hist, key=lambda r: r["risk"])
            row = {
                "seed": sd, "group": g,
                "best_risk": round(best["risk"], 4),
                "best_comfort_rate": best["result"].get("mean_comfort_rate"),
                "best_mean_power_kw": best["result"].get("mean_power_kw"),
                "best_energy_ratio": best["result"].get("mean_energy_ratio"),
                "n_evals": len(hist),
            }
            rows.append(row)
            print(f"  → BEST: risk={row['best_risk']}  comfort={row['best_comfort_rate']}  pwr={row['best_mean_power_kw']}kW")

    # ── 집계 ────────────────────────────────────────────────────────────────
    agg = []
    for g in groups:
        vals = [r["best_risk"] for r in rows if r["group"] == g]
        cr_v = [r["best_comfort_rate"] for r in rows if r["group"] == g and r["best_comfort_rate"] is not None]
        pw_v = [r["best_mean_power_kw"] for r in rows if r["group"] == g and r["best_mean_power_kw"] is not None]
        er_v = [r["best_energy_ratio"] for r in rows if r["group"] == g and r["best_energy_ratio"] is not None]
        agg.append({
            "group": g, "n": len(vals),
            "mean_best_risk": round(mean(vals), 4) if vals else None,
            "std_best_risk":  round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min_best_risk":  round(min(vals), 4) if vals else None,
            "mean_comfort_rate": round(mean(cr_v), 3) if cr_v else None,
            "mean_power_kw":     round(mean(pw_v), 3) if pw_v else None,
            "mean_energy_ratio": round(mean(er_v), 4) if er_v else None,
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    REP.mkdir(exist_ok=True)
    (REP / f"study_rows_{ts}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (REP / f"study_agg_{ts}.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md = [
        "# House Energy Management — G0~G4 Study Report",
        "",
        f"**Scenario:** 7-FMU Building HVAC Co-simulation  ",
        f"**budget={args.budget}**, **n_trials={args.n_trials}**, **seeds={seeds}**  ",
        f"**Run:** {ts}",
        "",
        "## Results",
        "",
        "| Group | mean risk ↓ | std | min risk | comfort rate ↑ | mean power (kW) | energy ratio ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in agg:
        mr = f"{a['mean_best_risk']:.4f}" if a["mean_best_risk"] is not None else "N/A"
        sd = f"{a['std_best_risk']:.4f}"  if a["std_best_risk"]  is not None else "N/A"
        mn = f"{a['min_best_risk']:.4f}"  if a["min_best_risk"]  is not None else "N/A"
        cr = f"{a['mean_comfort_rate']:.3f}" if a["mean_comfort_rate"] is not None else "N/A"
        pw = f"{a['mean_power_kw']:.3f}"  if a["mean_power_kw"]  is not None else "N/A"
        er = f"{a['mean_energy_ratio']:.4f}" if a["mean_energy_ratio"] is not None else "N/A"
        md.append(f"| {a['group']} | {mr} | {sd} | {mn} | {cr} | {pw} | {er} |")

    md += [
        "",
        "## Metric Definitions",
        "",
        "- **energy_ratio** = mean_heater_power / (2 × 4000 W). 낮을수록 에너지 효율 ↑",
        "- **comfort_rate** = 두 방 모두 18-23°C 유지 비율. 높을수록 쾌적 ↑",
        "- **risk** = energy_ratio + comfort_violation + (1−sr)×0.5. 낮을수록 좋음",
        "",
        "## ADVEI Informatics Points",
        "",
        "1. **Semantic Interoperability** — RAG vocab이 'thermal comfort', 'heating setpoint' 등",
        "   자연어 쿼리를 FMU 변수(T_heatStart, OvenHeatTransfer)로 자동 매핑",
        "2. **Explainable Optimization** — G4 warm-start 근거(prior risk) 로그 추적",
        "3. **Safety-critical Automation** — 쾌적 범위 이탈(< 18°C 또는 > 23°C) 자동 페널티",
    ]

    out_md = REP / f"study_report_{ts}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    # 터미널 요약
    print(f"\n{'='*65}")
    print(f"{'Group':<25} {'mean risk':>10} {'std':>8} {'comfort':>9} {'pwr kW':>8}")
    print("-" * 65)
    for a in agg:
        mr = f"{a['mean_best_risk']:.4f}" if a["mean_best_risk"] else "N/A"
        sd = f"{a['std_best_risk']:.4f}"  if a["std_best_risk"] is not None else "N/A"
        cr = f"{a['mean_comfort_rate']:.3f}" if a["mean_comfort_rate"] else "N/A"
        pw = f"{a['mean_power_kw']:.3f}"  if a["mean_power_kw"] else "N/A"
        print(f"{a['group']:<25} {mr:>10} {sd:>8} {cr:>9} {pw:>8}")

    print(f"\n[DONE] {out_md}")


def _build_house_vocab(rag) -> None:
    """House 도메인 용어를 RAG DB에 추가."""
    house_vocab = [
        {"id": "h_OvenHeatTransfer",  "doc": "OvenHeatTransfer: Heater thermal transfer coefficient [W/K]. Higher value means faster heating response but more energy use."},
        {"id": "h_T_heatStart",       "doc": "T_heatStart: Temperature [K] below which heating activates. Lower value delays heating onset, saving energy."},
        {"id": "h_T_heatStop",        "doc": "T_heatStop: Temperature [K] above which heating deactivates. Wider deadband (T_heatStop - T_heatStart) reduces cycling energy."},
        {"id": "h_transferTime",      "doc": "transferTime: Thermal response time constant [s]. Longer time means slower heat transfer, smoother temperature curves."},
        {"id": "h_Tinit_Room1",       "doc": "Tinit_Room1: Initial temperature of room 1 [K]. Starting closer to setpoint reduces warm-up energy."},
        {"id": "h_Tinit_Room2",       "doc": "Tinit_Room2: Initial temperature of room 2 [K]. Starting closer to setpoint reduces warm-up energy."},
        {"id": "h_inner_k",           "doc": "InnerWall.k: Inner wall thermal conductance [W/K]. Higher k means more heat exchange between rooms, balancing temperatures."},
        {"id": "h_outer_k",           "doc": "OuterWall.k_outsidewall: Outer wall thermal conductance [W/K]. Lower k = better insulation, less heat loss to outside."},
        {"id": "h_energy_ratio",      "doc": "energy_ratio: mean_heater_power / (2*4000W). Dimensionless energy efficiency metric. Lower is better."},
        {"id": "h_comfort_rate",      "doc": "comfort_rate: Fraction of simulation steps where BOTH rooms are within 18-23°C comfort zone. Higher is better."},
        {"id": "h_risk",              "doc": "risk = energy_ratio + comfort_violation + (1-success)*0.5. Building energy management risk metric. Lower is better."},
        {"id": "h_T_room",            "doc": "T_room: Current room air temperature [K]. Output of Room1/Room2 FMU. Target: 291.15-296.15 K (18-23°C)."},
        {"id": "h_h_powerHeater",     "doc": "h_powerHeater: Heater power input to room [W]. Controlled by TempController. Minimize for energy efficiency."},
        {"id": "h_comfort_zone",      "doc": "thermal comfort zone: 18-23°C (291-296 K). Both rooms must stay in this range for occupant comfort."},
    ]
    existing = rag._vocab_col.get(ids=[v["id"] for v in house_vocab])["ids"]
    new_docs = [v for v in house_vocab if v["id"] not in existing]
    if new_docs:
        rag._vocab_col.upsert(
            documents=[v["doc"] for v in new_docs],
            ids=[v["id"] for v in new_docs],
            metadatas=[{"fmu": "house", "term": v["id"]} for v in new_docs],
        )
        print(f"[RAG] house vocab added: {len(new_docs)} entries")
    else:
        print(f"[RAG] house vocab already indexed")


if __name__ == "__main__":
    main()
