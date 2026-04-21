# House Energy Management — LLM Agent Co-simulation (Scenario 2)

Building HVAC 에너지 관리 시나리오. 7-FMU OSP 코시뮬레이션으로
LLM 에이전트(G0–G4 ablation)가 TempController 파라미터를 최적화하여
에너지 소비 최소화 + 쾌적도(18–24°C) 유지를 동시에 달성.

## 구조

```
house_energy/
  house_sim.py          # 7-FMU 코시뮬 엔진 (fmpy, Celsius 단위)
  run_study.py          # G0–G4 ablation study orchestrator
  configs/
    house_energy_objective.json   # 파라미터 공간 정의
  reports/
    study_rows_*.json             # 원시 trial 데이터
    study_agg_*.json              # 집계 통계
    study_report_*.md             # 자동 보고서
    paper_ready_summary.md        # 논문용 최종 요약
```

## FMU 코시뮬레이션 구조 (OSP demo-cases/house)

```
Clock ──────────────────→ TempController (T_clock)
Room1.T_room ───────────→ TempController (T_room1)
Room2.T_room ───────────→ TempController (T_room2)
TempController.h_room1 →→ Room1 (h_powerHeater)
TempController.h_room2 →→ Room2 (h_powerHeater)
InnerWall.h_wall ───────→ Room1 (h_InnerWall)
InnerWall.h_wall ───────→ Room2 (h_InnerWall)
OuterWall1.h_wall ──────→ Room1 (h_OuterWall)
OuterWall2.h_wall ──────→ Room2 (h_OuterWall)
Room1.T_room ───────────→ InnerWall / OuterWall1
Room2.T_room ───────────→ InnerWall / OuterWall2
```

## 최적화 목표

**Risk** = energy_ratio + comfort_violation + (1−sr)×0.3

- `energy_ratio` = mean_heater_power / (2×20W) — 낮을수록 에너지 효율
- `comfort_violation` = 1 − comfort_rate — 쾌적 이탈 비율
- `sr` = comfort_rate ≥ 0.5 성공 여부

## 실험 실행

```bash
# 기본 (budget=12, n_trials=2, 5 seeds)
python run_study.py

# 커스텀
python run_study.py --budget 15 --n-trials 3 --seeds "11,22,33"
```

## 결과 요약 (budget=12, n_trials=2, 5 seeds)

| Group | mean risk ↓ | std ↓ | comfort ↑ |
|---|---:|---:|---:|
| G0 Random | 1.1677 | 0.0297 | 0.437 |
| G1 Grid | 1.2304 | 0.0000 | 0.372 |
| G2 LLM-only | 0.7554 | 0.4510 | 0.490 |
| G3 LLM+Reflection | 0.6988 | 0.4810 | 0.545 |
| **G4 LLM+RAG+Reflection** | **0.0236** | **0.0183** | **0.978** |

→ G4가 G0 대비 **98% 리스크 감소**, 쾌적도 **97.8%** 달성

## 의존성

FMU 경로: `../../0422/simulatoragent/demo-cases-master/house/`  
RAG DB: `../energy_llm_agent/rag_db/` (공유)
