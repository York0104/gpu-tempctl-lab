# GPU Temperature Control Lab

本專案用於建立可重複的 GPU 溫度控制與情境化測試流程，目標是在固定工作負載下，透過 `power limit`、`workload mode`，以及情境層的 fault injection，讓 GPU 進入指定熱行為區間，並輸出可分析的溫度、功耗、效能與事件紀錄。

目前主驗證平台為 **NVIDIA GeForce GTX 1080 Ti**，並已完成：
- 60°C 模式閉環控制驗證
- 80°C 模式閉環控制驗證
- `scenario_runner.py` 情境封裝與 quick smoke 驗證
- `cooling_anomaly` 風扇 fault injection 基本流程驗證

---

## 1. 專案目標

本專案聚焦兩類能力：

1. **閉環溫控（Closed-loop temperature control）**
   - 60°C 模式：目標在 `55~65°C`
   - 80°C 模式：目標在 `75~85°C`

2. **情境化測試（Scenario-based validation）**
   - `natural_high_heat`：自然升溫場景，驗證控制器是否能在目標區間內維持指定秒數
   - `cooling_anomaly`：冷卻異常場景，注入風扇 fault 並記錄 thermal / performance / event 資料

主要用途：
- 建立可重複熱行為資料
- 作為異常偵測 / 負載辨識 / 控制設計的資料來源
- 驗證多層架構：controller → workload → logger → scenario runner

---

## 2. 功能設計

### 2.1 控制架構

目前採用三層設計：

- **控制層**
  - `controller.py`
  - 根據目標溫度模式（60 / 80）執行區間控制

- **負載層**
  - `workload_torch.py`
  - 透過 PyTorch GEMM / matmul 產生固定 GPU 熱源

- **情境層**
  - `scenario_runner.py`
  - 封裝自然高熱與冷卻異常等情境，統一輸出 metadata、事件與結果摘要

---

### 2.2 主要 actuator

目前第一版設計使用兩個主要控制量：

1. **Power Limit**
   - 主要 actuator
   - 透過 `nvidia-smi -pl` 調整 GPU 功耗上限

2. **Workload Mode**
   - 次要 actuator
   - 透過不同矩陣大小改變熱源強度：
     - `low`  -> 4096
     - `mid`  -> 6144
     - `high` -> 8192

### 2.3 Fan 控制定位

目前 fan 不作為一般閉環控制主軸，而是：
- 在 `cooling_anomaly` 中作為故障注入對象
- 作為後續低溫輔助控制或高溫極端情境擴充的保留能力

---

## 3. 溫控策略

### 3.1 60°C 模式

- 目標溫度：`60°C`
- 容許區間：`55~65°C`
- 穩定區間：`58~62°C`
- 初始設定：
  - `PL_init = 150W`
  - `workload_mode = mid`

控制邏輯：
- 優先調整 `power limit`
- 若 `PL` 已經降到下限仍偏熱，切換到 `low` workload mode
- 若溫度偏低，再恢復 `mid`

這使 60°C 模式在 1080 Ti 上不只依賴 PL，也能在 PL 到底後透過 workload mode 補足控制空間。

---

### 3.2 80°C 模式

- 目標溫度：`80°C`
- 容許區間：`75~85°C`
- 穩定區間：`78~82°C`
- 初始設定：
  - `PL_init = 300W`
  - `workload_mode = mid`

控制邏輯：
- 高溫模式為**非對稱控制**
- 若溫度過低，先等待 / 維持高 PL，必要時切換 `high` mode
- 若溫度過高，再降低 PL，必要時由 `high` 降回 `mid`

---

## 4. 程式說明

### `controller.py`
閉環控制器。

功能：
- 根據 `--target 60` 或 `--target 80` 載入不同 profile
- 每秒讀取 NVML telemetry
- 每 3 秒平均溫度做一次控制決策
- 控制 `power limit`
- 視情況切換 `workload mode`
- 成功後輸出 `controller.csv`
- 結束時恢復 default power limit

---

### `workload_torch.py`
GPU 熱源產生器。

功能：
- 使用 PyTorch 在 CUDA 上持續執行 GEMM / matmul
- 支援：
  - `--mode low|mid|high`
  - `--size`
  - `--perf-csv`
  - `--fault-flag-file`
- 可在 scenario 模式下同步記錄效能資料

---

### `gpu_logger.py`
簡單 NVML telemetry logger。

功能：
- 以固定 interval 記錄：
  - 溫度
  - 功耗
  - power limit
  - GPU / memory utilization
  - fan
  - clocks

適用於：
- 單點驗證
- 手動開環觀測
- 快速記錄 raw telemetry

---

### `pl_sweep.py`
開環掃描工具。

功能：
- 依序掃描多個 power limit
- 每個測點先 cooldown，再執行固定 workload
- 輸出：
  - `sweep_raw.csv`
  - `sweep_summary.csv`
  - `metadata.json`

用途：
- 建立 `PL -> steady-state temperature` 對照表
- 作為閉環控制器初始參數依據

---

### `perf_logger.py`
工作負載效能記錄器。

功能：
- 記錄 workload 的效能資訊：
  - `iters_total`
  - `iters_last_interval`
  - `iter_per_sec`
  - `avg_iter_ms`
- 用於分析 fault / workload 模式切換時效能變化

---

### `fan_fault_injector.py`
風扇手動控制 / 恢復工具。

功能：
- 使用 NVML low-level API 設定 manual fan %
- 可恢復 default fan policy
- 主要提供 `scenario_runner.py` 的 `cooling_anomaly` 使用

---

### `scenario_runner.py`
情境執行器。

目前支援：
- `natural_high_heat`
- `cooling_anomaly`

功能：
- 建立 run directory
- 記錄 metadata v2
- 啟動 controller / workload / logger / fault injection
- 執行後自動輸出簡短 summary
- 將 requested / effective / artifacts / result 分開記錄，方便後續分析

---

## 5. 輸出檔案

### 5.1 `controller.csv`
閉環控制輸出。

主要欄位：
- `ts`
- `elapsed_s`
- `target_c`
- `phase`
- `temp_c`
- `temp_avg_3s`
- `power_w`
- `power_limit_w`
- `gpu_util_percent`
- `mem_util_percent`
- `fan_percent`
- `graphics_clock_mhz`
- `mem_clock_mhz`
- `controller_pl_w`
- `controller_mode`
- `decision_action`
- `in_band_consecutive_s`

用途：
- 檢查控制器是否成功收斂
- 看最終穩態 PL / mode
- 看控制動作是 `PL±5`、`PL±10`、`MODE_LOW`、`MODE_HIGH` 等

---

### 5.2 `thermal.csv`
scenario 中的 thermal telemetry。

主要用途：
- 記錄異常場景中溫度 / 功耗 / power limit / fan 行為
- 作為 anomaly 場景輸出的主要 thermal trace

---

### 5.3 `perf.csv`
工作負載效能資料。

主要欄位：
- `scenario`
- `fault_active`
- `workload_mode`
- `matrix_size`
- `iters_total`
- `iters_last_interval`
- `iter_per_sec`
- `avg_iter_ms`

用途：
- 分析 fault 前後的效能變化
- 對照 thermal 行為看 GPU 是否受冷卻異常影響

---

### 5.4 `events.csv`
scenario 事件紀錄。

常見事件：
- `power_limit_set`
- `workload_start`
- `thermal_logger_start`
- `fault_start`
- `fault_end_restore_auto`
- `fault_auto_restored_by_safety`
- `safety_temp_hit`
- `scenario_complete`

用途：
- 重建 scenario 的事件時間線
- 判斷 fault 是否真的注入、是否成功恢復、是否觸發 safety stop

---

### 5.5 `metadata.json`
scenario metadata v2。

結構分為：
- `requested`
- `effective`
- `artifacts`
- `result`
- `gpu`
- `host`
- `timestamps`

用途：
- 記錄本次執行的請求參數
- 記錄實際生效控制配置
- 記錄輸出檔案位置
- 記錄結果與最終 summary 欄位

---

### 5.6 `controller_stdout.log`
controller 的 stdout/stderr 轉存檔。

用途：
- 快速確認是否達成：
  - `Success: stayed within [...]`
  - `Restored default PL = ...`
- 作為 scenario_runner 失敗時的第一個 debug 入口

---

## 6. 輸出解讀

### 6.1 如何判讀 `controller.csv`

#### 60°C 模式成功特徵
- `temp_c` 長時間落在 `55~65`
- `in_band_consecutive_s` 最後達到目標秒數
- `decision_action` 從調整逐步變成 `HOLD_STABLE`
- `controller_pl_w` 可能收斂到低值（例如 125W）
- `controller_mode` 在需要時可能切成 `low`

#### 80°C 模式成功特徵
- `temp_c` 長時間落在 `75~85`
- `controller_pl_w` 常接近高值（例如 300W）
- `controller_mode` 在需要時可能切成 `high`

---

### 6.2 如何判讀 `metadata.json`

#### `requested`
代表使用者要求的 scenario 參數，例如：
- target
- hold_seconds
- max_run_seconds
- fault_start
- fault_seconds

#### `effective`
代表實際生效的控制設計，例如：
- controller band
- `pl_init_w`
- `workload_mode`
- fault plan

#### `result`
最重要的欄位：
- `status`
- `return_code`
- `success_condition`
- `final_temp_c`
- `final_controller_pl_w`
- `final_controller_mode`
- `fault_started`
- `fault_restored`
- `safety_stop`

---

### 6.3 如何判讀 `events.csv`

#### `natural_high_heat`
主要看：
- controller 是否正常結束
- 不一定會有 events.csv（依實作不同）

#### `cooling_anomaly`
主要看：
- 是否出現 `fault_start`
- 是否出現 `fault_end_restore_auto`
- 或是否被 `fault_auto_restored_by_safety` 提前終止

若：
- `fault_start = yes`
- `fault_restore = yes`
- `safety_stop = false`

則代表 anomaly scenario 正常執行並恢復。

---

## 7. 測試指令

### 7.1 直接測控制器

#### 60°C 模式
```bash
sudo -E $(which python) controller.py \
  --target 60 \
  --hold-seconds 600 \
  --max-run-seconds 1800 \
  --output ctl_60.csv
```

#### 80°C 模式
```bash
sudo -E $(which python) controller.py \
  --target 80 \
  --hold-seconds 600 \
  --max-run-seconds 1800 \
  --output ctl_80.csv
```

### 7.2 單純 workload 測試
```bash
python workload_torch.py --mode mid
```

或：

```bash
python workload_torch.py --mode high --perf-csv logs/perf_test.csv
```

### 7.3 單純 logger 測試
```bash
python gpu_logger.py --output logs/gpu_log.csv --gpu-index 0 --sample-interval 1
```

### 7.4 Power Limit 開環掃描
```bash
python pl_sweep.py --pls 150,175,200,225,250,275,300 --run-seconds 180 --cooldown-target 40
```

若 cooldown 太慢可放寬：

```bash
python pl_sweep.py --pls 150,175,200,225,250,275,300 --run-seconds 180 --cooldown-target 45
```

### 7.5 Scenario quick smoke

#### natural_high_heat + 60°C
```bash
sudo -E $(which python) scenario_runner.py \
  --scenario natural_high_heat \
  --target 60 \
  --hold-seconds 20 \
  --max-run-seconds 180 \
  --output-root logs/quick_smoke_all
```

#### natural_high_heat + 80°C
```bash
sudo -E $(which python) scenario_runner.py \
  --scenario natural_high_heat \
  --target 80 \
  --hold-seconds 20 \
  --max-run-seconds 180 \
  --output-root logs/quick_smoke_all
```

#### cooling_anomaly
```bash
sudo -E $(which python) scenario_runner.py \
  --scenario cooling_anomaly \
  --workload-mode mid \
  --power-limit 250 \
  --run-seconds 120 \
  --fault-start 30 \
  --fault-seconds 30 \
  --fault-fan-percent 50 \
  --sample-interval 1 \
  --safety-temp 83 \
  --output-root logs/quick_smoke_all
```

### 7.6 Scenario 正式驗證

#### 60°C 長時間驗證
```bash
sudo -E $(which python) scenario_runner.py \
  --scenario natural_high_heat \
  --target 60 \
  --hold-seconds 600 \
  --max-run-seconds 1800 \
  --output-root logs/validation_all
```

#### 80°C 長時間驗證
```bash
sudo -E $(which python) scenario_runner.py \
  --scenario natural_high_heat \
  --target 80 \
  --hold-seconds 600 \
  --max-run-seconds 1800 \
  --output-root logs/validation_all
```

#### cooling anomaly 正式驗證
```bash
sudo -E $(which python) scenario_runner.py \
  --scenario cooling_anomaly \
  --workload-mode mid \
  --power-limit 250 \
  --run-seconds 900 \
  --fault-start 300 \
  --fault-seconds 300 \
  --fault-fan-percent 50 \
  --sample-interval 1 \
  --safety-temp 83 \
  --output-root logs/validation_all
```

## 8. 目錄結構

建議以目前 branch 實作為基礎，整理成以下結構：

```text
gpu-tempctl-lab/
├─ README.md
├─ controller.py
├─ scenario_runner.py
├─ workload_torch.py
├─ gpu_logger.py
├─ pl_sweep.py
├─ perf_logger.py
├─ fan_fault_injector.py
├─ ctl_60.csv
├─ ctl_60_v2.csv
├─ ctl_80.csv
├─ ctl_80_v2.csv
├─ gpu_log_pl250.csv
├─ .gitignore
└─ logs/
   ├─ pl_sweep_YYYYMMDD_HHMMSS/
   │  ├─ metadata.json
   │  ├─ sweep_raw.csv
   │  └─ sweep_summary.csv
   ├─ quick_smoke_all/
   │  ├─ natural_high_heat_YYYYMMDD_HHMMSS/
   │  │  ├─ command.json
   │  │  ├─ controller.csv
   │  │  ├─ controller_stdout.log
   │  │  └─ metadata.json
   │  └─ cooling_anomaly_YYYYMMDD_HHMMSS/
   │     ├─ thermal.csv
   │     ├─ perf.csv
   │     ├─ events.csv
   │     ├─ metadata.json
   │     ├─ workload_stdout.log
   │     └─ logger_stdout.log
   └─ validation_all/
      └─ ...
```
