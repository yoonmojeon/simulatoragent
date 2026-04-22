# 0429 Project Overview

이 저장소는 논문 제출용 LLM 기반 FMU 코시뮬레이션 실험을 위한 단일 워크스페이스입니다.

현재는 아래 두 시나리오를 동일한 실험 프레임으로 운영합니다.

- `energy_llm_agent`: 해양/에너지 결합 vessel 시나리오 (9-FMU)
- `house_energy`: 건물 HVAC 시나리오 (7-FMU)

## 공통 방법론 (G0~G4)

- G0: Random baseline
- G1: Grid baseline
- G2: LLM only
- G3: LLM + Reflection
- G4: LLM + RAG + Reflection (proposed)

두 시나리오 모두 동일한 그룹 체계와 지표 집계 형식을 사용합니다.

## 디렉토리 구조

```text
0429/
  demo-cases/                 # FMU demo assets
  energy_llm_agent/           # Scenario 1 (vessel)
    llm_agent.py
    run_llm_study.py
    fmu_sim.py
    scenarios.py
    reports/
  house_energy/               # Scenario 2 (house)
    llm_agent_house.py
    run_llm_study_house.py
    house_sim.py
    reports/
  external_fmus/              # optional external FMU corpus
```

## 빠른 실행

### Vessel (G0~G4)

```powershell
cd c:\Users\rlaeh\OneDrive\Desktop\mo\0429\energy_llm_agent
python -u run_llm_study.py --groups all --budget 12 --n-trials 3 --seeds 11,22,33
```

### House (G0~G4)

```powershell
cd c:\Users\rlaeh\OneDrive\Desktop\mo\0429\house_energy
python -u run_llm_study_house.py --groups all --budget 12 --n-trials 3 --seeds 11,22,33
```

## 결과 파일

- Vessel: `energy_llm_agent/reports/llm_study_agg_*.json`, `llm_study_report_*.md`
- House: `house_energy/reports/llm_study_agg_*.json`, `llm_study_report_*.md`

## 환경 설정

루트 `.env` 예시:

```env
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3:8b
```

## 참고

- 이 버전은 `0429` 단독 동작을 기준으로 정리되어 있습니다.
- 과거 `0410`, `0422` 의존 경로는 제거되었습니다.
