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

## v2：加入近距離攻防資料（使用者已設為 main 預設；未打 tag）

| 項目 | 內容 |
| --- | --- |
| 精簡檔 | `models/compact/interaction_gp_v2_close.npz`（455 KB，SHA-256 `f9d3dd89…0add4`） |
| 原 pkl（本機） | `models/gp/close_range_20261124_v2/model.pkl`（201 MB，SHA-256 `71f8779f…31f`）；精簡檔預測差 0 |
| 訓練資料 | `close_range_20261124` 訓練 24 場中車距 −2～3 m 的 2416 筆，加 v1 訓練樣本隨機 2584 筆；共 5000 筆 |
| 適用範圍 | 車距約 −2～6.8 m（範圍由訓練輸入自動取得）；其餘同 v1 |
| 離線指標 | 近距離測試 12 場 MAE：0～0.8 m 0.036、0.8～1.6 m 0.030 m（v1 為 0.146、0.101）；舊保留 8 場 0.0238（v1 0.0229） |
| 閉迴路 | 20 組成對：v2 各候選只超車 1／20、平均換目標 87.7 次；v1 與無 GP 都 20／20（見紀錄） |
| 重建 | 需本機三份資料；執行 [train_compare.py](../reports/experiments/2026-11-24_close_range_gp/train_compare.py) |
| 紀錄 | [近距離資料與 v2](../reports/experiments/2026-11-24_close_range_gp/README.md) |

main.py 目前用 v2；要改回 v1，把 `INTERACTION_GP_MODEL_PATH` 改成 `models/compact/interaction_gp_v1_m1_30.npz`。

## 論文格式一步 GP v1（paper_rollout_overtake 使用；未打 tag）

輸入輸出不同於上面的互動 GP，載入類別是 `strategy/paper_gp.py` 的 `PaperGpModel`，路徑設定為 `main.py` 的 `PAPER_GP_MODEL_PATH`。

| 項目 | 內容 |
| --- | --- |
| 精簡檔 | `models/compact/paper_gp_v1.npz`（1.9 MB，SHA-256 `80b6c70c…743333`） |
| 原 pkl（本機） | `models/gp/paper_format_20261125/model.pkl`（803 MB，SHA-256 `60615c53…dfcc39`）；500 列訓練輸入的 4 個輸出預測差都是 0 |
| 輸入／答案 | 14 維（論文 z_k 10 維＋v2 歷史 4 欄，`PAPER_FEATURES`）→ 前車 0.1 s 後的 [Δs, Δd, Δe_ψ, Δv]，4 個獨立 GP；不含 d_target |
| 訓練資料 | 選取規則同 v2：`close_range_20261124` 訓練場車距 −2～3 m 的 2540 筆，加試探資料訓練場 2460 筆；共 5000 筆 |
| GP 設定 | 同 v2；擬合 Δs 331 s、Δd 346 s、Δe_ψ 473 s、Δv 212 s。Δs、Δe_ψ、Δv 有界限警告，未重新調整 |
| 適用範圍 | 圓角矩形、單一防守設定；車距 −1.99～6.79 m、前車 1.38～6.90 m/s、後車−前車速度 −1.79～1.47 m/s（由訓練輸入自動取得） |
| 離線指標（一步） | 近距離 0≤gap≤2 m（803 筆）：Δd MAE 0.0050 m（等速 0.0250）、Δs 0.0021 m、Δe_ψ 0.0131 rad；Δv 在所有測試集都比「車速不變」差 |
| 閉迴路 | 單場示範（rear 7／front 6 m/s，20 s）：k=2 未超車、換目標 184 次、碰撞重疊 122 步；只用平均（k=0）7.54 s 超車。k=2 決策時前車 d 的 ±2σ 涵蓋 +1 s 為 57%；每次決策平均 0.79 s |
| 重建 | 需本機三份資料；執行 [train_eval.py](../reports/experiments/2026-11-25_paper_format_gp/train_eval.py)，再用上面的 export 命令產生精簡檔 |
| 紀錄 | [論文格式 GP 第一版](../reports/experiments/2026-11-25_paper_format_gp/README.md)、[paper_rollout_overtake 示範](../reports/experiments/2026-11-25_paper_rollout_overtake/README.md) |
