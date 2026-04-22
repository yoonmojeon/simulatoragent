# House Energy Management (Scenario 2)

7-FMU 건물 HVAC 코시뮬레이션에서 LLM 에이전트가 난방 제어 파라미터를
탐색/최적화하는 실험 모듈입니다.

이 디렉토리는 `0429` 내부 자산(`demo-cases`, `energy_llm_agent/rag_db`)을 직접 사용합니다.

## 목표

- 에너지 소비를 낮추면서(heater power 최소화)
- 쾌적 범위(18~24 degC) 유지율을 높이는 파라미터 탐색
- G0~G4 비교를 통해 LLM/RAG/Reflection 효과 분석

## 주요 파일

- `house_sim.py`
  - 7-FMU 직접 코시뮬레이션 엔진
  - Jacobi coupling으로 입력/출력 교환
  - risk 및 comfort/power 지표 계산
- `llm_agent_house.py`
  - house 시나리오용 tool-calling 에이전트
  - `inspect_scenario`, `query_rag`, `run_simulation`, `generate_report` 툴 루프
- `run_llm_study_house.py`
  - G0~G4 통합 실행 오케스트레이터
- `configs/house_energy_objective.json`
  - 탐색 파라미터 범위 정의

## FMU 구성

- `Clock`
- `TempController`
- `Room1`, `Room2`
- `InnerWall`
- `OuterWall1`, `OuterWall2`

연결 관계(요약):
- room 온도 출력 -> controller 입력
- controller heater 출력 -> room 입력
- wall 열전달 출력 -> room 입력
- room 온도 -> wall 입력

## 최적화 지표

`risk = energy_ratio + comfort_violation + (1 - success) * 0.3`

- `energy_ratio`: 평균 난방 전력 비율(낮을수록 좋음)
- `comfort_rate`: 두 방이 동시에 쾌적 구간에 있는 비율(높을수록 좋음)
- `comfort_violation = 1 - comfort_rate`
- `success = 1` if `comfort_rate >= 0.5` else `0`

## 실행 방법

### 1) 전체 G0~G4 실행

```powershell
cd c:\Users\rlaeh\OneDrive\Desktop\mo\0429\house_energy
python -u run_llm_study_house.py --groups all --budget 12 --n-trials 3 --seeds 11,22,33
```

### 2) G4만 실행

```powershell
python -u run_llm_study_house.py --groups G4 --budget 12 --n-trials 3 --seeds 11,22,33
```

### 3) 에이전트 단독 실행

```powershell
python -u llm_agent_house.py --mode G4 --max-iter 50
```

## 최신 결과

최신 집계 파일: `reports/llm_study_agg_20260422_182737.json`

- G0: 1.1664
- G1: 1.2304
- G2: 1.1681
- G3: 0.3212
- G4: 0.2508

현재 run 기준으로는 G4가 house 시나리오에서 최저 risk를 기록합니다.

## 출력 파일

- `reports/llm_study_agg_*.json`
- `reports/llm_study_report_*.md`
- `reports/llm_agent_report_*.md`

## 경로 의존성

- FMU 경로: `../demo-cases/house/`
- RAG DB: `../energy_llm_agent/rag_db/`
