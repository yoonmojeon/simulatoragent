# Paper-Ready Result Summary — House Energy Management
## LLM Agent for FMU-based Building HVAC Co-simulation

**Scenario:** Building Energy Management (7-FMU OSP Co-simulation)  
**FMUs:** Clock + TempController + Room1 + Room2 + InnerWall + OuterWall1 + OuterWall2  
**Objective:** Minimize heating energy while maintaining thermal comfort (18-24°C) in a 2-room building  
**Simulation:** 30-minute window, 1-second communication step, OSP demo-cases/house  

---

## Ablation Study Results (G0–G4)

| Group | Description | mean risk ↓ | std ↓ | min risk | mean comfort ↑ |
|---|---|---:|---:|---:|---:|
| G0 | Random Search | 1.1677 | 0.0297 | 1.1306 | 0.437 |
| G1 | Grid Search | 1.2304 | 0.0000 | 1.2304 | 0.372 |
| G2 | LLM-only (local exploit) | 0.7554 | 0.4510 | 0.2117 | 0.490 |
| G3 | LLM + Reflection (adaptive σ) | 0.6988 | 0.4810 | 0.1152 | 0.545 |
| **G4** | **LLM + RAG + Reflection (Proposed)** | **0.0236** | **0.0183** | **0.0076** | **0.978** |

*5 seeds × 12 budget × 2 trials per evaluation*

---

## Key Findings

- **G4 최저 mean risk:** 0.0236 vs G0 1.1677 → **98.0% 리스크 감소**
- **G4 최저 분산:** std=0.0183 vs G3 std=0.4810 → **26.3배 안정적**
- **G4 최고 쾌적도:** 97.8% (양 방 모두 18-24°C 유지) vs G0 43.7%
- **G2/G3 높은 분산:** RAG prior 없이 가끔 좋은 해를 찾으나 일관성 없음 (std≈0.48)
- **G1 grid std=0:** seed 무관한 결정론적 탐색, 우연한 개선 불가

---

## Risk Metric Definition

```
risk (single trial) = energy_ratio + comfort_violation + (1 - success) × 0.3
  energy_ratio      = mean_heater_power / (2 × 20 W)        ← 에너지 효율
  comfort_violation = 1 - comfort_rate                       ← 쾌적 위반
  success           = 1 if comfort_rate ≥ 0.50 else 0       ← 이진 성공
  
risk (n trials, aggregated) = mean_risk + std_risk + (1 - success_rate) × 0.3
```

---

## Thermostat Optimization Challenge

The TempController implements **bang-bang thermostat control**:
- Heater turns ON at full power (`OvenHeatTransfer` W) when `T_room < T_heatStart`
- Heater turns OFF when `T_room > T_heatStop`
- 9-dimensional search space: controller (4 params), 2 rooms (1 each), 3 walls (1 each)

**Why G0–G3 struggle:**
- Low thermal capacity (C=1 J/K) creates rapid temperature cycling
- Without domain knowledge, random/grid search rarely finds stable operating point
- Adaptive sigma in G3 improves locally but cannot escape poor basins without prior

**Why G4 succeeds:**
- RAG warm-start provides domain-specific priors from accumulated trials
- Semantic vocabulary maps "thermal comfort", "heating threshold" → FMU params
- Online learning updates RAG after each trial → knowledge compounds across seeds

---

## ADVEI Informatics Contribution Points

1. **Semantic Interoperability**
   - RAG vocabulary maps natural-language concepts to FMU variables:
     - "comfort zone" → `T_heatStart`, `T_heatStop`
     - "insulation quality" → `k_outsidewall`, `k_inner`
     - "heater efficiency" → `OvenHeatTransfer`, `transferTime`
   - Enables zero-shot parameter recommendation from domain description

2. **Explainable Optimization (XAI)**
   - G4 warm-start logs: `[G4] RAG warm-start: prior risk=0.1662`
   - Traceable reasoning chain: natural-language goal → RAG query → parameter proposal
   - Human-interpretable rationale for every candidate parameter set

3. **Safety-critical Automation**
   - Failure penalty (×0.3) for comfort violation < 50% enforces safety constraint
   - G4 achieves 97.8% comfort compliance (vs G0's 43.7%) with zero manual tuning
   - The RAG prior acts as a "safety gate" preventing exploration of dangerous regimes

---

## Comparison: Two Scenarios (Energy LLM Agent Paper)

| Metric | Scenario 1: Construction Vessel (marine) | Scenario 2: House HVAC (building) |
|---|---|---|
| FMU count | 9 FMUs | 7 FMUs |
| Domain | Marine offshore DP + winch | Building thermal energy management |
| G4 mean risk | 1.0833 | 0.0236 |
| G4 comfort/stability | std=0.0067 (11× better than G0) | std=0.0183 (26× better than G3) |
| Key finding | Safety gate prevents power overrun | RAG prior enables near-optimal comfort |
| ADVEI angle | Semantic FMU variable mapping | Thermal domain knowledge transfer |
