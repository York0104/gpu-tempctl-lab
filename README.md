# GPU Temperature Control Experiment

本專案用於驗證 GPU 溫度控制實驗，目標是在固定工作負載下，透過 `power limit` 與 `workload mode` 調整，使 GPU 溫度維持在指定區間內持續一段時間。

## Control Targets
- 60C mode: 55C ~ 65C
- 80C mode: 75C ~ 85C

## Validated Platform
- NVIDIA GeForce GTX 1080 Ti
- Driver: 535.288.01
- Power Limit range: 125W ~ 300W
- Default Power Limit: 250W

## Main Files
- `controller.py`: closed-loop controller
- `workload_torch.py`: GPU heating workload
- `gpu_logger.py`: telemetry logger
- `pl_sweep.py`: open-loop power limit sweep tool

## Notes
Validated results:
- 60C mode achieved 600s in-band hold
- 80C mode achieved 600s in-band hold
