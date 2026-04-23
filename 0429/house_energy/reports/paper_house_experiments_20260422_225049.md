# Paper House Experiments

Run: 20260422_225049

## Repetition Summary

| method | n | mean best risk | std | min | mean trial risk | sims | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| G2_llm_only | 5 | 0.5674 | 0.3831 | 0.2519 | 0.9061 | 14.40 | 16.20 |
| G3_reflection | 5 | 0.6983 | 0.3512 | 0.3476 | 0.9898 | 8.00 | 10.60 |
| G4_full_independent | 5 | 0.3713 | 0.0861 | 0.2612 | 0.7300 | 12.00 | 19.00 |

## G4 Ablation Summary

| variant | n | mean best risk | std | min | mean trial risk | sims | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold_memory | 5 | 0.5168 | 0.3566 | 0.2481 | 0.6889 | 13.00 | 19.40 |
| full | 5 | 0.1738 | 0.0612 | 0.1269 | 0.2372 | 10.60 | 18.80 |
| semantic_only | 5 | 0.4236 | 0.1206 | 0.1834 | 0.9037 | 11.40 | 16.80 |
| warm_memory | 5 | 0.1717 | 0.0853 | 0.0290 | 0.2261 | 11.20 | 18.80 |

## Artifact
- JSON: `paper_house_experiments_20260422_225049.json`