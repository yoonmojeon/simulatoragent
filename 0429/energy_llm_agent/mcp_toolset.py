"""
MCP Toolset — 실제 ChromaDB 기반 RAG 포함
==========================================
Tools:
  fmu_inspector      FMU 메타데이터/변수 목록 추출
  rag_retriever      ChromaDB에서 시맨틱 검색 (vocab + trial 결과)
  simulation_runner  생성된 co-sim 스크립트 실행
  rag_add_trial      trial 결과를 RAG DB에 추가
  report_generator   Markdown 보고서 작성
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fmpy import read_model_description

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------
# 1) FMU 검사
# ------------------------------------------------------------------
def fmu_inspector(fmu_path: str) -> dict[str, Any]:
    """MCP tool: inspect FMU variables/metadata."""
    fp = Path(fmu_path)
    md = read_model_description(str(fp), validate=False)
    vars_meta = []
    for v in md.modelVariables[:120]:
        vars_meta.append(
            {
                "name": v.name,
                "causality": getattr(v, "causality", None),
                "variability": getattr(v, "variability", None),
                "description": getattr(v, "description", None),
            }
        )
    return {
        "path": str(fp),
        "model_name": md.modelName,
        "guid": md.guid,
        "n_variables": len(md.modelVariables),
        "variables_preview": vars_meta,
    }


# ------------------------------------------------------------------
# 2) RAG retriever — ChromaDB 실제 연결
# ------------------------------------------------------------------
def rag_retriever(query: str, top_k: int = 5, mode: str = "both") -> dict[str, Any]:
    """
    MCP tool: ChromaDB 시맨틱 검색.
    mode: "vocab" | "trials" | "both"
    """
    from rag_store import RagStore

    store = RagStore()
    result: dict[str, Any] = {"query": query}

    if mode in ("vocab", "both"):
        result["vocab_hits"] = store.query_vocab(query, n=top_k)

    if mode in ("trials", "both"):
        result["trial_hits"] = store.query_trials(
            query, n=top_k, max_risk=2.5
        )
        result["best_params"] = store.best_params(n=3)

    return result


# ------------------------------------------------------------------
# 3) trial 결과 → RAG DB 추가
# ------------------------------------------------------------------
def rag_add_trial(trial_id: str, params: dict, result: dict, group: str = "") -> dict[str, Any]:
    """MCP tool: simulation 결과를 RAG DB에 추가."""
    from rag_store import RagStore

    store = RagStore()
    store.add_trial(trial_id, params, result, group=group)
    return {"status": "ok", "trial_id": trial_id, "risk": result.get("risk_score")}


# ------------------------------------------------------------------
# 4) 시뮬레이션 실행
# ------------------------------------------------------------------
def simulation_runner(params: dict, n_trials: int = 3) -> dict[str, Any]:
    """MCP tool: run generated FMU co-simulation code."""
    script = ROOT / "generated" / "generated_cosim_runner.py"
    cmd = [
        sys.executable,
        str(script),
        "--params-json",
        json.dumps(params, ensure_ascii=False),
        "--n-trials",
        str(n_trials),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        return {"risk_score": 9999.0, "error": p.stderr.strip() or "runner_failed"}
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    try:
        return json.loads(lines[-1])
    except Exception:
        return {"risk_score": 9999.0, "error": "invalid_json_output", "raw": p.stdout[-300:]}


# ------------------------------------------------------------------
# 5) 보고서 생성
# ------------------------------------------------------------------
def report_generator(goal: str, result: dict, out_path: str) -> dict[str, Any]:
    """MCP tool: write a compact markdown report."""
    risk = result.get("risk_score", "N/A")
    sr = result.get("success_rate", "N/A")
    pwr = result.get("mean_power_kw", "N/A")
    energy = result.get("energy_j", "N/A")

    lines = [
        "# MCP Auto Report",
        "",
        f"**Goal:** {goal}",
        "",
        "## Key Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Risk Score | {risk} |",
        f"| Success Rate | {sr} |",
        f"| Mean Power kW | {pwr} |",
        f"| Energy J | {energy} |",
        "",
        "## Full Result",
        "```json",
        json.dumps(result, ensure_ascii=False, indent=2),
        "```",
    ]
    fp = Path(out_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text("\n".join(lines), encoding="utf-8")
    return {"report_path": str(fp)}
