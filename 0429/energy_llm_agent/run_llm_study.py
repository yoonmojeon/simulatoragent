"""
run_llm_study.py — 실제 LLM 기반 G0~G4 비교 실험
==================================================
G0, G1: Python 휴리스틱 (빠른 베이스라인)
G2: 실제 LLM (tool-calling, RAG 없음)
G3: 실제 LLM + Reflection 체크포인트
G4: 실제 LLM + RAG warm-start + Reflection (제안 방법)

실행:
  python run_llm_study.py
  python run_llm_study.py --groups G4          # G4만
  python run_llm_study.py --groups G0,G1,G4    # 선택 실행
  python run_llm_study.py --skip-heuristics    # G2/G3/G4만
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent
BASE_SIM = ROOT.parents[1] / "0422" / "simulatoragent"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE_SIM))

os.environ.setdefault("SIM_PROTOCOL_PROFILE", "paper")

from fmu_sim import CoSimRunner
from scenarios import get_scenario

REP = ROOT / "reports"
REP.mkdir(exist_ok=True)

SCENARIO_ID = "construction_vessel_full"

# ── 탐색 공간 (G0/G1 휴리스틱 기준) ────────────────────────────────────────
SPACE = {
    "wave_model": {
        "wave_height":    (1.0, 5.0),
        "peak_frequency": (0.3, 2.0),
        "k_down":         (0.5, 4.0),
        "k_north":        (0.5, 4.0),
        "k_east":         (0.5, 4.0),
    },
    "winch": {
        "K_p": (100.0, 800.0),
        "K_d": (5.0,  100.0),
        "K_i": (0.01,  5.0),
    },
}


def _sample_random(rng: random.Random) -> dict:
    return {
        fmu: {p: round(rng.uniform(lo, hi), 4) for p, (lo, hi) in params.items()}
        for fmu, params in SPACE.items()
    }


def _sample_grid(i: int, budget: int) -> dict:
    frac = (i - 0.5) / max(1, budget)
    out: dict = {}
    d = 0
    for fmu, params in SPACE.items():
        out[fmu] = {}
        for p, (lo, hi) in params.items():
            phase = (frac + d * 0.137) % 1.0
            out[fmu][p] = round(lo + phase * (hi - lo), 4)
            d += 1
    return out


def _run_heuristic(group: str, runner: CoSimRunner, budget: int, n_trials: int, seed: int) -> dict:
    """G0/G1 Python 휴리스틱 (LLM 없음)."""
    rng = random.Random(seed)
    best = None
    for i in range(1, budget + 1):
        params = _sample_random(rng) if group == "G0_random" else _sample_grid(i, budget)
        res = runner.run_n_trials(override_params=params, n=n_trials)
        risk = float(res.get("risk_score", 9999.0))
        if best is None or risk < best["risk"]:
            best = {"params": params, "result": res, "risk": risk}
        print(f"  [{group}] i={i:02d}  risk={risk:.4f}  best={best['risk']:.4f}")
    return best


def _run_llm(group: str, seed: int | None = None) -> dict:
    """G2/G3/G4 실제 LLM 에이전트."""
    from llm_agent import run_agent

    # G4: RAG vocab 먼저 빌드
    if group == "G4_rag_reflection":
        try:
            from rag_store import RagStore
            rag = RagStore()
            n = rag.build_fmu_vocab()
            print(f"  [RAG] vocab built/verified: {n} entries")
        except Exception as e:
            print(f"  [RAG] vocab build skipped: {e}")

    mode_map = {
        "G2_llm_only":     "G2",
        "G3_reflection":   "G3",
        "G4_rag_reflection": "G4",
    }
    mode = mode_map[group]
    result = run_agent(
        mode=mode,
        max_iterations=35,
        reflection_interval=3,
        verbose=True,
        llm_seed=seed,
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget",    type=int, default=12, help="G0/G1 평가 횟수")
    ap.add_argument("--n-trials",  type=int, default=3,  help="G0/G1 trial 반복")
    ap.add_argument("--seeds",     default="11,22,33",   help="G0/G1 시드 (쉼표 구분)")
    ap.add_argument("--groups",    default="all",         help="실행할 그룹, 예: G0,G4")
    ap.add_argument("--skip-heuristics", action="store_true", help="G0/G1 건너뜀")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    all_groups = ["G0_random", "G1_grid", "G2_llm_only", "G3_reflection", "G4_rag_reflection"]
    if args.groups.lower() == "all":
        groups = all_groups
    else:
        req = set(args.groups.split(","))
        groups = [g for g in all_groups if any(r in g for r in req)]

    if args.skip_heuristics:
        groups = [g for g in groups if g not in ("G0_random", "G1_grid")]

    runner = CoSimRunner(get_scenario(SCENARIO_ID))

    summary: list[dict] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for group in groups:
        print(f"\n{'='*60}")
        print(f"[GROUP] {group}")
        print(f"{'='*60}")

        if group in ("G0_random", "G1_grid"):
            # ── 휴리스틱: 여러 seed 평균 ────────────────────────────────
            bests = []
            for seed in seeds:
                t0 = time.time()
                best = _run_heuristic(group, runner, args.budget, args.n_trials, seed)
                elapsed = time.time() - t0
                bests.append(best["risk"])
                print(f"  seed={seed}  best_risk={best['risk']:.4f}  {elapsed:.0f}s")
            summary.append({
                "group":          group,
                "type":           "heuristic",
                "mean_best_risk": round(mean(bests), 4),
                "std_best_risk":  round(pstdev(bests), 4) if len(bests) > 1 else 0.0,
                "min_best_risk":  round(min(bests), 4),
                "n_seeds":        len(seeds),
            })

        else:
            # ── 실제 LLM 에이전트 ───────────────────────────────────────
            t0 = time.time()
            result = _run_llm(group, seed=seeds[0] if seeds else None)
            elapsed = time.time() - t0

            # LLM trial 이력에서 best risk 추출
            from llm_agent import _trial_history
            if _trial_history:
                risks = [t["result"].get("risk_score", 9999.0) for t in _trial_history
                         if not t["result"].get("error")]
                best_risk = min(risks) if risks else 9999.0
                mean_risk = mean(risks) if risks else 9999.0
            else:
                best_risk = 9999.0
                mean_risk = 9999.0

            summary.append({
                "group":          group,
                "type":           "llm_agent",
                "model":          os.getenv("OPENAI_MODEL", "unknown"),
                "mode":           group.split("_", 1)[1] if "_" in group else group,
                "n_simulations":  result.get("n_simulations", 0),
                "n_reflections":  result.get("n_reflections", 0),
                "n_llm_calls":    result.get("n_llm_calls", 0),
                "n_findings":     result.get("n_findings", 0),
                "best_risk":      round(best_risk, 4),
                "mean_risk":      round(mean_risk, 4),
                "report_path":    result.get("report_path", ""),
                "done_reason":    result.get("done_reason", ""),
                "elapsed_s":      round(elapsed, 1),
            })
            print(f"  → best_risk={best_risk:.4f}  sims={result.get('n_simulations',0)}"
                  f"  llm_calls={result.get('n_llm_calls',0)}  {elapsed:.0f}s")

    # ── 결과 저장 및 출력 ─────────────────────────────────────────────────
    agg_path = REP / f"llm_study_agg_{ts}.json"
    agg_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown 요약
    md_lines = [
        "# LLM Agent Study — G0~G4 비교",
        "",
        f"**Scenario:** {SCENARIO_ID}  ",
        f"**Run:** {ts}  ",
        f"**Model:** {os.getenv('OPENAI_MODEL', 'N/A')}",
        "",
        "## 결과",
        "",
        "| Group | Type | best risk ↓ | LLM calls | Simulations | Report |",
        "|---|---|---:|---:|---:|---|",
    ]
    for s in summary:
        t = s.get("type", "")
        if t == "heuristic":
            md_lines.append(
                f"| {s['group']} | heuristic | {s['mean_best_risk']:.4f} (±{s['std_best_risk']:.4f}) | — | — | — |"
            )
        else:
            rp = f"`{Path(s['report_path']).name}`" if s.get("report_path") else "—"
            md_lines.append(
                f"| {s['group']} | LLM | {s['best_risk']:.4f} | {s['n_llm_calls']} | {s['n_simulations']} | {rp} |"
            )

    md_lines += [
        "",
        "## G4 방법론 특징 (제안)",
        "",
        "- **코드 생성:** `generate_cosim_code()` 로 FMPy 러너 스크립트 자동 생성",
        "- **Semantic Interoperability:** `query_rag(mode='vocab')` 로 자연어 → FMU 변수 매핑",
        "- **파라미터 제안:** LLM이 물리적 근거로 파라미터 추론 (랜덤 아님)",
        "- **Reflection:** N 시뮬마다 자기 점검 → 전략 재수립",
        "- **RAG 온라인 학습:** `add_to_rag()` 로 성공/실패 결과 축적",
        "- **보고서 생성:** `generate_report()` 로 LLM이 직접 Markdown 분석 보고서 작성",
    ]

    md_path = REP / f"llm_study_report_{ts}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"{'Group':<25} {'Type':12} {'best risk':>12} {'LLM calls':>10} {'sims':>6}")
    print("-" * 60)
    for s in summary:
        t = s.get("type", "")
        if t == "heuristic":
            print(f"{s['group']:<25} {'heuristic':12} {s['mean_best_risk']:>12.4f} {'—':>10} {'—':>6}")
        else:
            print(f"{s['group']:<25} {'llm_agent':12} {s['best_risk']:>12.4f} "
                  f"{s['n_llm_calls']:>10} {s['n_simulations']:>6}")

    print(f"\n[SAVED] {md_path}")
    print(f"[SAVED] {agg_path}")


if __name__ == "__main__":
    main()
