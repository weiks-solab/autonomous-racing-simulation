# 互動 GP 模型登記表

完整 pkl 含 n×n 分解矩陣，檔案很大且被 Git 忽略（`/models/gp/`），只在本機。
`models/compact/` 的精簡檔只存標準化參數、訓練輸入與答案、kernel 數值，可放進 Git。
載入時以固定超參數重新計算分解，預測與原 pkl 一致，而且讀取不經 pickle。

由 pkl 產生精簡檔，並自動比對預測（差異需 <1e-9，否則刪除輸出）：

```powershell
.\.venv\Scripts\python.exe interaction_gp.py export --model models/gp/<模型目錄>/model.pkl --output models/compact/<名稱>.npz
```

`main.py` 的 `INTERACTION_GP_MODEL_PATH` 可指向 `.npz` 或 `.pkl`，換版本只改這一行。

## v1：M1-30（Git tag `interaction-gp-v1`）

| 項目 | 內容 |
| --- | --- |
| 精簡檔 | `models/compact/interaction_gp_v1_m1_30.npz`（375 KB，SHA-256 `dbc983c1…95aa7e`） |
| 原 pkl（本機） | `models/gp/speed06_20261124_c_m1_30/model.pkl`（131 MB，SHA-256 `93f3c182…5068e9`） |
| 精簡檔核對 | 500 筆訓練輸入的預測差最大 1.0e-13 m；兩個閉迴路場景 8 s 內決策逐步相同 |
| 輸入／答案 | 13 維互動歷史特徵（`strategy/interaction_features.py`）→ 前車 0.5 s 後的 Δd（m） |
| 訓練資料 | `history_gp_20261124` 舊 12 場（1704 筆）＋ `history_speed06_20261124` 新訓練 8 場（2336 筆）；共 4040 筆 |
| 資料雜湊 | samples.csv：`48d2cd40…37ac5f9`（history_gp）、`ddd76542…2ec2ea`（speed06） |
| GP 設定 | StandardScaler＋Constant×ARD RBF＋WhiteKernel、normalize_y、無 restart；擬合 140 s、無警告 |
| 適用範圍 | 圓角矩形；車距約 1.6～6.8 m、前車約 1.4～6.9 m/s；d_target 只有 0、±0.3、±0.45、±0.6 |
| 主要指標 | 新測試 8 場的 0.6 m 試探 MAE 0.039／0.041 m（0～15 s／15～30 s），±2σ 涵蓋 79.6／77.0%；舊保留 8 場整體 MAE 0.0229 m |
| 重建 | 需本機兩份資料；執行 [compare.py](../reports/experiments/2026-11-24_speed_amplitude_probe/compare.py)（輸出目錄已存在時須先改名） |
| 紀錄 | [速度 × ±0.6 m 實驗](../reports/experiments/2026-11-24_speed_amplitude_probe/README.md)、[interaction_overtake 評估](../reports/experiments/2026-11-24_interaction_overtake/README.md) |

限制：並排與超越（車距 <1.6 m）不在訓練範圍，interaction_overtake 會退回定速等 d 預測。
