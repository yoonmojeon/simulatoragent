"""Paper-oriented House HVAC LLM repetition and RAG ablation experiments.

This runner keeps experiment artifacts separated from the default RAG DB.
It runs:
  1. Independent G2/G3/G4 repetitions over multiple LLM seeds.
  2. G4 ablations for semantic-only, cold memory, warm memory, and full RAG.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent
AGENT_ROOT = ROOT.parent / "energy_llm_agent"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT_ROOT))

from house_sim import HouseCoSimRunner
from llm_agent_house import run_agent
from rag_store import RagStore


HOUSE_VOCAB = [
    ("thermal comfort", "Fraction of time both rooms remain inside 18-24 degC comfort band.", "metric"),
    ("energy_ratio", "Mean heater power divided by nominal 40 W two-room heating capacity.", "metric"),
    ("comfort_violation", "1 - comfort_rate; main penalty once energy use is already low.", "metric"),
    ("OvenHeatTransfer", "Bang-bang heater power parameter. Higher warms faster but increases energy.", "TempController"),
    ("T_heatStart", "Thermostat lower threshold. Heater turns on below this temperature.", "TempController"),
    ("T_heatStop", "Thermostat upper threshold. Heater turns off above this temperature.", "TempController"),
    ("deadband", "T_heatStop - T_heatStart. Controls cycling and comfort-energy tradeoff.", "TempController"),
    ("k_outsidewall", "Outer wall thermal conductance. Lower means better insulation and lower heat loss.", "OuterWall"),
    ("inner_wall.k", "Thermal coupling between rooms. Higher values equalize room temperatures.", "InnerWall"),
    ("Tinit_Room", "Initial room temperature. Starting near comfort band reduces warm-up energy.", "Room"),
]


PRIOR_PARAMS = [
    {
        "name": "prior_best_02079",
        "params": {
            "controller": {"OvenHeatTransfer": 6, "T_heatStart": 18, "T_heatStop": 22},
            "inner_wall": {"k": 0.01},
            "outer_wall1": {"k_outsidewall": 0.001},
            "outer_wall2": {"k_outsidewall": 0.001},
            "room1": {"Tinit_Room1": 17},
            "room2": {"Tinit_Room2": 17},
        },
    },
    {
        "name": "prior_g4_02508",
        "params": {
            "controller": {"OvenHeatTransfer": 7, "T_heatStart": 19, "T_heatStop": 22.5},
            "inner_wall": {"k": 0.012},
            "outer_wall1": {"k_outsidewall": 0.001},
            "outer_wall2": {"k_outsidewall": 0.001},
            "room1": {"Tinit_Room1": 17.5},
            "room2": {"Tinit_Room2": 17.5},
        },
    },
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_rag_db(path: Path, preload: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    store = RagStore(db_path=path)
    store.build_fmu_vocab()
    docs, ids, metas = [], [], []
    for i, (term, desc, fmu) in enumerate(HOUSE_VOCAB):
        docs.append(f"{term}: {desc}")
        ids.append(f"house_vocab_{i:03d}_{term}".replace(" ", "_"))
        metas.append({"term": term, "fmu": fmu, "scenario": "house_energy"})
    store._vocab_col.upsert(documents=docs, ids=ids, metadatas=metas)

    if preload:
        runner = HouseCoSimRunner()
        for prior in PRIOR_PARAMS:
            result = runner.run_n_trials(override_params=prior["params"], n=10)
            store.add_trial(
                trial_id=prior["name"],
                params=prior["params"],
                result=result,
                group="paper_prior",
            )


def trial_stats(trials: list[dict]) -> tuple[float, float]:
    risks = [
        float(t["result"].get("risk_score", 9999.0))
        for t in trials
        if isinstance(t, dict) and isinstance(t.get("result"), dict) and not t["result"].get("error")
    ]
    if not risks:
        return 9999.0, 9999.0
    return min(risks), mean(risks)


def run_one(label: str, mode: str, seed: int, rag_db: Path | None, add_disabled: bool, max_iter: int) -> dict:
    old_db = os.environ.get("HOUSE_RAG_DB_PATH")
    old_add = os.environ.get("HOUSE_RAG_ADD_DISABLED")
    try:
        if rag_db is None:
            os.environ.pop("HOUSE_RAG_DB_PATH", None)
        else:
            os.environ["HOUSE_RAG_DB_PATH"] = str(rag_db)
        if add_disabled:
            os.environ["HOUSE_RAG_ADD_DISABLED"] = "1"
        else:
            os.environ.pop("HOUSE_RAG_ADD_DISABLED", None)

        t0 = time.time()
        result = run_agent(
            mode=mode,
            max_iterations=max_iter,
            reflection_interval=3,
            verbose=False,
            llm_seed=seed,
        )
        elapsed = time.time() - t0
    finally:
        if old_db is None:
            os.environ.pop("HOUSE_RAG_DB_PATH", None)
        else:
            os.environ["HOUSE_RAG_DB_PATH"] = old_db
        if old_add is None:
            os.environ.pop("HOUSE_RAG_ADD_DISABLED", None)
        else:
            os.environ["HOUSE_RAG_ADD_DISABLED"] = old_add

    best_risk, mean_risk = trial_stats(result.get("trial_history", []))
    return {
        "label": label,
        "mode": mode,
        "seed": seed,
        "best_risk": round(best_risk, 4),
        "mean_risk": round(mean_risk, 4),
        "n_simulations": result.get("n_simulations", 0),
        "n_reflections": result.get("n_reflections", 0),
        "n_llm_calls": result.get("n_llm_calls", 0),
        "n_findings": result.get("n_findings", 0),
        "done_reason": result.get("done_reason", ""),
        "report_path": result.get("report_path", ""),
        "elapsed_s": round(elapsed, 1),
    }


def summarize(rows: list[dict], key: str) -> list[dict]:
    out = []
    for label in sorted({r[key] for r in rows}):
        subset = [r for r in rows if r[key] == label]
        bests = [r["best_risk"] for r in subset]
        means = [r["mean_risk"] for r in subset]
        out.append({
            key: label,
            "n": len(subset),
            "mean_best_risk": round(mean(bests), 4),
            "std_best_risk": round(pstdev(bests), 4) if len(bests) > 1 else 0.0,
            "min_best_risk": round(min(bests), 4),
            "mean_trial_risk": round(mean(means), 4),
            "mean_simulations": round(mean([r["n_simulations"] for r in subset]), 2),
            "mean_llm_calls": round(mean([r["n_llm_calls"] for r in subset]), 2),
        })
    return out


def write_outputs(ts: str, payload: dict) -> None:
    json_path = REPORTS / f"paper_house_experiments_{ts}.json"
    md_path = REPORTS / f"paper_house_experiments_{ts}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Paper House Experiments",
        "",
        f"Run: {ts}",
        "",
        "## Repetition Summary",
        "",
        "| method | n | mean best risk | std | min | mean trial risk | sims | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["repetition_summary"]:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['mean_best_risk']:.4f} | "
            f"{row['std_best_risk']:.4f} | {row['min_best_risk']:.4f} | "
            f"{row['mean_trial_risk']:.4f} | {row['mean_simulations']:.2f} | {row['mean_llm_calls']:.2f} |"
        )

    lines += [
        "",
        "## G4 Ablation Summary",
        "",
        "| variant | n | mean best risk | std | min | mean trial risk | sims | calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["ablation_summary"]:
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['mean_best_risk']:.4f} | "
            f"{row['std_best_risk']:.4f} | {row['min_best_risk']:.4f} | "
            f"{row['mean_trial_risk']:.4f} | {row['mean_simulations']:.2f} | {row['mean_llm_calls']:.2f} |"
        )

    lines += ["", "## Artifact", f"- JSON: `{json_path.name}`"]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["repeat", "ablation", "all"], default="all")
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--max-iter-g2", type=int, default=28)
    parser.add_argument("--max-iter-g3", type=int, default=32)
    parser.add_argument("--max-iter-g4", type=int, default=50)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = ROOT / "rag_db_experiments" / ts
    exp_root.mkdir(parents=True, exist_ok=True)

    repetition_rows: list[dict] = []
    ablation_rows: list[dict] = []

    if args.phase in ("repeat", "all"):
        specs = [
            ("G2_llm_only", "G2", None, False, args.max_iter_g2),
            ("G3_reflection", "G3", None, False, args.max_iter_g3),
            ("G4_full_independent", "G4", "repeat_g4_full", False, args.max_iter_g4),
        ]
        for label, mode, db_name, add_disabled, max_iter in specs:
            for seed in seeds:
                rag_db = None
                if db_name is not None:
                    rag_db = exp_root / "repeat" / label / f"seed_{seed}"
                    reset_dir(rag_db)
                    prepare_rag_db(rag_db, preload=False)
                print(f"[REPEAT] {label} seed={seed}")
                row = run_one(label, mode, seed, rag_db, add_disabled, max_iter)
                repetition_rows.append(row)
                print(f"  best={row['best_risk']:.4f} mean={row['mean_risk']:.4f}")

    if args.phase in ("ablation", "all"):
        variants = [
            ("semantic_only", False, True),
            ("cold_memory", False, False),
            ("warm_memory", True, True),
            ("full", True, False),
        ]
        for variant, preload, add_disabled in variants:
            for seed in seeds:
                rag_db = exp_root / "ablation" / variant / f"seed_{seed}"
                reset_dir(rag_db)
                prepare_rag_db(rag_db, preload=preload)
                print(f"[ABLATION] {variant} seed={seed}")
                row = run_one(variant, "G4", seed, rag_db, add_disabled, args.max_iter_g4)
                row["variant"] = variant
                ablation_rows.append(row)
                print(f"  best={row['best_risk']:.4f} mean={row['mean_risk']:.4f}")

    payload = {
        "timestamp": ts,
        "seeds": seeds,
        "experiment_root": str(exp_root),
        "repetition_rows": repetition_rows,
        "repetition_summary": summarize(repetition_rows, "label") if repetition_rows else [],
        "ablation_rows": ablation_rows,
        "ablation_summary": summarize(ablation_rows, "variant") if ablation_rows else [],
    }
    write_outputs(ts, payload)


if __name__ == "__main__":
    main()
