# Experiment Log

| Date | Directory | Thesis | Outcome | Runs |
|------|-----------|--------|---------|------|
| 2026-03-09 | [depth-scaling](./depth-scaling/) | Transformer depth shows diminishing returns for find_return; 4-8 layers matches 16 layers | Confirmed by final reward (4L=97.4% of 16L), but reframing as steps-to-threshold shows depth helps most where it's hardest — 16L reaches reward 15.5 in 1736 steps vs 4L's 4529 (2.6x speedup) | lively-wizard-xmz694 (1L), happy-sun-ytvomm (2L), curious-spirit-3tii0t (4L), quick-dragon-kmtmsf (8L), silly-river-931ykh (16L) |
