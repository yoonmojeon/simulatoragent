# 0429 Energy LLM Agent (Marine + Energy FMU Co-simulation)

에너지 저널(ADVEI Special Issue) 투고를 위한 실험 자동화 파이프라인입니다.

핵심 목표:
- 해양 + 에너지 FMU 코시뮬레이션(FMPy 기반)
- LLM 에이전트 스타일 워크플로우(목표 해석 → 코드 생성 → 최적화 → 리포트)
- 재현 가능한 실험 산출물(JSON/Markdown)

## 데이터 소스

- OSP demo-cases (marine reference models)
  - `construction-vessel` (vessel + power_system + thruster + wave/wind + winch)
  - `lars` (launch and recovery)
- Non-OSP FMU corpus
  - `modelica/fmi-cross-check`에서 선별한 에너지/전기 FMU
  - 위치: `external_fmus/cross_check_energy/`
    - `Rectifier_CATIA_R2016x.fmu`
    - `fuelrail_AMESim15.fmu`
    - `ControlledTemperature_CATIA_R2016x.fmu`

현재 워크스페이스 기준:
- FMU 루트: `../demo-cases`

## 빠른 시작

```powershell
cd c:\Users\rlaeh\OneDrive\Desktop\mo\0429\energy_llm_agent
python agent_pipeline.py --goal "연료/전력 위반을 줄이면서 안전한 DP crane operation" --scenario vessel --budget 10
```

실행 결과:
- 생성 코드: `generated/generated_cosim_runner.py`
- 최적화 로그: `reports/optimization_result_*.json`
- 자동 보고서: `reports/auto_report_*.md`

## 아키텍처

1. **Orchestrator**
   - 자연어 목표를 실험 목적함수로 변환
2. **Co-simulation Code Generator**
   - 시나리오 실행용 파이썬 코드를 생성
3. **Parameter Optimizer**
   - FMU 파라미터 탐색 (랜덤 + 국소 탐색)
4. **Automatic Report Generator**
   - 결과를 논문형 Markdown으로 정리

## MCP Toolset 모드

`mcp_toolset.py`에 다음 툴이 포함됩니다.
- `fmu_inspector`: FMU 변수/메타데이터 추출
- `rag_retriever`: 로컬 실험 이력 검색(ChromaDB 자리표시자)
- `simulation_runner`: 생성된 코시뮬 코드 실행
- `report_generator`: 실험 결과 Markdown 생성

실행:
```powershell
python mcp_demo_run.py
```

## G0~G4 비교 실험

```powershell
python run_study.py --budget 6 --n-trials 1 --seeds 11,22
```

출력:
- `reports/study_rows_*.json`
- `reports/study_agg_*.json`
- `reports/study_report_*.md`

## 노트

- 백엔드는 `0429/energy_llm_agent` 내부 `fmu_sim.py`/`scenarios.py`를 사용합니다.
- `SIM_PROTOCOL_PROFILE=paper`를 기본 적용해 난이도를 유지합니다.
