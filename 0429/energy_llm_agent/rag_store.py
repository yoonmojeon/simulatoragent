"""
RAG Store — ChromaDB 기반
=========================
두 가지 컬렉션을 관리합니다.

1. fmu_vars : FMU 변수 이름 + 물리 설명 → 시맨틱 검색용
2. trial_results : 과거 시뮬 trial (params + risk) → 유사 파라미터 prior 검색용

사용 예:
    from rag_store import RagStore
    store = RagStore()
    store.build_fmu_vocab()               # FMU 변수 설명 인덱싱
    store.add_trial(params, result)        # trial 결과 추가
    hits = store.query_trials("low risk winch K_p", n=5)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "rag_db"
DEMO_CASES = ROOT.parent / "demo-cases"

# --- 물리 용어 사전 (해양 공학 → FMU 변수 매핑) ---
DOMAIN_VOCAB: list[dict[str, str]] = [
    {
        "term": "wave_height",
        "description": "Significant wave height (m). Higher value means stronger sea state. Directly drives wave force disturbances on the vessel.",
        "fmu": "linearized_wave_model",
    },
    {
        "term": "peak_frequency",
        "description": "Wave peak frequency (rad/s). Higher frequency means shorter wave period, increasing dynamic loading.",
        "fmu": "linearized_wave_model",
    },
    {
        "term": "k_down",
        "description": "Vertical wave stiffness coefficient. Controls heave motion coupling from wave model to vessel.",
        "fmu": "linearized_wave_model",
    },
    {
        "term": "k_north",
        "description": "North wave stiffness. Controls surge coupling. Higher value amplifies north-south wave force on vessel.",
        "fmu": "linearized_wave_model",
    },
    {
        "term": "k_east",
        "description": "East wave stiffness. Controls sway coupling. Higher value amplifies east-west wave force on vessel.",
        "fmu": "linearized_wave_model",
    },
    {
        "term": "K_p",
        "description": "Winch PID proportional gain (N/m). Governs responsiveness of crane load depth control. Too high causes oscillation.",
        "fmu": "winch",
    },
    {
        "term": "K_d",
        "description": "Winch PID derivative gain (N·s/m). Provides damping to suppress winch speed oscillation.",
        "fmu": "winch",
    },
    {
        "term": "K_i",
        "description": "Winch PID integral gain (N/m·s). Eliminates steady-state depth error. High values risk windup.",
        "fmu": "winch",
    },
    {
        "term": "power_system",
        "description": "Shipboard power bus. Aggregates active/reactive load from thrusters and crane winch. Bus voltage and frequency are outputs.",
        "fmu": "power_system",
    },
    {
        "term": "risk_score",
        "description": "Composite metric: mean_vps + std_vps + (1-success_rate)*failure_penalty. Lower is better.",
        "fmu": "metric",
    },
    {
        "term": "viol_per_step",
        "description": "Fraction of simulation steps where power consumption exceeded power_limit_kw. Indicates unsafe power load.",
        "fmu": "metric",
    },
    {
        "term": "success_rate",
        "description": "Fraction of repeated trials where load depth reached target within tolerance and time limit.",
        "fmu": "metric",
    },
    {
        "term": "energy_j",
        "description": "Total energy consumed by winch motor during the simulation [J]. Key metric for marine energy efficiency.",
        "fmu": "metric",
    },
    {
        "term": "mean_power_kw",
        "description": "Mean power consumption of winch during operation [kW]. Should stay below power_limit_kw.",
        "fmu": "metric",
    },
    {
        "term": "overshoot_abs",
        "description": "Absolute overshoot of load depth from target [m]. Smaller is better for precise crane placement.",
        "fmu": "metric",
    },
]


class RagStore:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._vocab_col = self._client.get_or_create_collection(
            "fmu_vars",
            metadata={"hnsw:space": "cosine"},
        )
        self._trial_col = self._client.get_or_create_collection(
            "trial_results",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # 1) FMU 변수/물리 용어 인덱싱
    # ------------------------------------------------------------------
    def build_fmu_vocab(self, force: bool = False) -> int:
        if self._vocab_col.count() > 0 and not force:
            return self._vocab_col.count()

        docs, ids, metas = [], [], []
        for i, entry in enumerate(DOMAIN_VOCAB):
            term = entry["term"]
            desc = entry["description"]
            fmu = entry["fmu"]
            docs.append(f"{term}: {desc}")
            ids.append(f"vocab_{i:04d}_{term}")
            metas.append({"term": term, "fmu": fmu})

        # FMU 파일에서 변수명 추출해 추가
        base = ROOT.parents[1] / "0422" / "simulatoragent"
        sys.path.insert(0, str(base))
        try:
            from scenarios import SCENARIOS
            from fmpy import read_model_description as rmd

            for sc_key, sc_def in SCENARIOS.items():
                for fspec in sc_def.get("fmus", []):
                    fmu_case = fspec.get("case", "")
                    fmu_name = fspec.get("fmu", "")
                    fmu_path = (
                        DEMO_CASES / fmu_case / "fmus" / f"{fmu_name}.fmu"
                    )
                    if not fmu_path.exists():
                        continue
                    try:
                        md = rmd(str(fmu_path), validate=False)
                        for v in md.modelVariables:
                            vname = v.name
                            vdesc = getattr(v, "description", "") or ""
                            causality = getattr(v, "causality", "")
                            text = f"{fmu_name}.{vname} ({causality}): {vdesc}"
                            uid = f"fmuvar_{sc_key}_{fmu_name}_{vname}"[:200]
                            if uid not in ids:
                                docs.append(text)
                                ids.append(uid)
                                metas.append({"term": vname, "fmu": fmu_name, "scenario": sc_key})
                    except Exception:
                        continue
        except Exception:
            pass

        # 중복 제거
        seen: set[str] = set()
        final_docs, final_ids, final_metas = [], [], []
        for d, i, m in zip(docs, ids, metas):
            if i not in seen:
                seen.add(i)
                final_docs.append(d)
                final_ids.append(i)
                final_metas.append(m)

        batch = 100
        for start in range(0, len(final_docs), batch):
            self._vocab_col.upsert(
                documents=final_docs[start : start + batch],
                ids=final_ids[start : start + batch],
                metadatas=final_metas[start : start + batch],
            )
        return self._vocab_col.count()

    def query_vocab(self, query: str, n: int = 5) -> list[dict]:
        res = self._vocab_col.query(query_texts=[query], n_results=min(n, self._vocab_col.count() or 1))
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({"text": doc, "meta": meta, "distance": round(dist, 4)})
        return hits

    # ------------------------------------------------------------------
    # 2) Trial 결과 인덱싱
    # ------------------------------------------------------------------
    def add_trial(self, trial_id: str, params: dict, result: dict, group: str = "") -> None:
        risk = result.get("risk_score", 9999.0)
        sr = result.get("success_rate", 0.0)
        mean_pwr = result.get("mean_power_kw") or result.get("trials", [{}])[0].get("mean_power_kw", 0)
        energy = result.get("energy_j") or result.get("trials", [{}])[0].get("energy_j", 0)

        flat_params = []
        for fmu, pdict in params.items():
            for p, v in pdict.items():
                flat_params.append(f"{fmu}.{p}={v:.4f}")
        doc = (
            f"group={group} risk={risk:.4f} success_rate={sr:.2f} "
            f"mean_power_kw={mean_pwr:.1f} energy_j={energy:.0f} "
            f"params: {' '.join(flat_params)}"
        )
        self._trial_col.upsert(
            documents=[doc],
            ids=[trial_id],
            metadatas=[{
                "group": group,
                "risk": float(risk),
                "success_rate": float(sr),
                "params_json": json.dumps(params, ensure_ascii=False),
            }],
        )

    def query_trials(self, query: str, n: int = 5, max_risk: float = 9998.0) -> list[dict]:
        total = self._trial_col.count()
        if total == 0:
            return []
        res = self._trial_col.query(query_texts=[query], n_results=min(n * 3, total))
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            if meta.get("risk", 9999.0) <= max_risk:
                try:
                    params = json.loads(meta["params_json"])
                except Exception:
                    params = {}
                hits.append({
                    "text": doc,
                    "risk": meta.get("risk"),
                    "success_rate": meta.get("success_rate"),
                    "params": params,
                    "distance": round(dist, 4),
                })
        return hits[:n]

    def best_params(self, n: int = 3) -> list[dict]:
        """risk 기준 상위 n개 trial params 반환."""
        total = self._trial_col.count()
        if total == 0:
            return []
        res = self._trial_col.get(include=["documents", "metadatas"])
        rows = []
        for doc, meta in zip(res["documents"], res["metadatas"]):
            try:
                params = json.loads(meta["params_json"])
            except Exception:
                params = {}
            rows.append({"risk": float(meta.get("risk", 9999)), "params": params})
        rows.sort(key=lambda r: r["risk"])
        return rows[:n]
