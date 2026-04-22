# 0429 Energy LLM Agent

`construction_vessel_full`(9-FMU) 시나리오를 대상으로, LLM 에이전트가
`FMU inspection → RAG retrieval → code generation → simulation loop → report generation`
까지 수행하는 실험 파이프라인입니다.

이 디렉토리는 `0429` 단독으로 동작하며, 외부 `0422` 의존성 없이 실행됩니다.

## 연구 목적

- 해양+에너지 결합 시스템에서 LLM 기반 자율 최적화 가능성 검증
- G0~G4 ablation 비교를 통해 제안 방법론(G4: LLM+RAG+Reflection) 평가
- 논문 투고용 재현 가능한 결과물(JSON, Markdown report) 자동 생성

## 시나리오

- **Scenario ID:** `construction_vessel_full`
- **구성 FMU:** wave model, wind model, vessel model, reference model, DP controller,
  thrust allocator, thruster model, power system, winch
- **목표:** crane depth 추종을 유지하면서 power violation 기반 risk 최소화

## 핵심 파일

- `llm_agent.py`
  - OpenAI/Ollama tool-calling 기반 메인 에이전트
  - 툴 스키마(`inspect_fmu`, `query_rag`, `generate_cosim_code`, `run_simulation`, `generate_report`)
  - G2/G3/G4 모드 로직 및 reflection 체크포인트
- `run_llm_study.py`
  - G0~G4 실행 오케스트레이터
  - 휴리스틱(G0/G1) + 실제 LLM 에이전트(G2/G3/G4) 통합 실행
- `mcp_toolset.py`
  - FMU inspection/RAG retrieval/simulation/reporting 툴 구현
- `codegen.py`
  - `generated/generated_cosim_runner.py` 자동 생성
- `fmu_sim.py`, `scenarios.py`
  - 코시뮬레이션 엔진/시나리오 정의(현재 디렉토리 자체 포함)
- `rag_store.py`
  - ChromaDB 기반 vocab + trial memory 저장소

## 실행 환경

- Python 3.10+
- `pip install -r requirements.txt`
- `.env` 예시
  - `OPENAI_BASE_URL=http://127.0.0.1:11434/v1`
  - `OPENAI_API_KEY=ollama`
  - `OPENAI_MODEL=qwen3:8b`

## 실행 방법

### 1) 전체 G0~G4 재실험

```powershell
cd c:\Users\rlaeh\OneDrive\Desktop\mo\0429\energy_llm_agent
python -u run_llm_study.py --groups all --budget 12 --n-trials 3 --seeds 11,22,33
```

### 2) G4만 재실험

```powershell
python -u run_llm_study.py --groups G4 --budget 12 --n-trials 3 --seeds 11,22,33
```

### 3) 에이전트 단독 실행

```powershell
python -u llm_agent.py --mode G4 --max-iter 50
```

## 출력 산출물

- `reports/llm_study_agg_*.json`:
  - 그룹별 최종 지표 집계(best/mean risk, llm calls, simulations)
- `reports/llm_study_report_*.md`:
  - 표 기반 요약 리포트
- `reports/llm_agent_report_*.md`:
  - 개별 LLM 최적화 세션 분석 리포트
- `generated/generated_cosim_runner.py`:
  - 코드 생성 결과물

## 최신 실험 결과(요약)

최신 집계 파일: `reports/llm_study_agg_20260422_182737.json`

- G0: 0.7986
- G1: 0.8986
- G2: 0.0203
- G3: 0.2032
- G4: 0.0203

## 메트릭 정의

- `risk_score` (lower is better)
- `success_rate`
- `mean_power_kw`
- `energy_j`

상세 계산식은 `fmu_sim.py`, `scenarios.py`의 profile 설정(`SIM_PROTOCOL_PROFILE=paper`)을 따릅니다.

## 재현성 체크리스트

- `0429/demo-cases` FMU 자산 존재 확인
- `.env` 모델 설정 확인(`qwen3:8b` 권장)
- 동일 seed/budget 사용
- 이전 실험 영향 제거가 필요하면 `reports/*`, `rag_db/*` 정리 후 실행
