# GP 模型訓練

## 互動歷史 GP：試探反應預測

`interaction_gp.py collect --features history` 收集 13 欄，答案仍為前車 `d(t+0.5)−d(t)`（使用時加回 front_d 即絕對位置）。
d 左正；歷史以 t−0.4、t−0.2、t 三幀切兩段。

| 欄位 | 定義 | 單位 |
| --- | --- | --- |
| gap | 前車 s − 後車 s | m |
| rear_d／front_d | 當下橫向位置 | m |
| rear_heading_error／front_heading_error | 車頭角 − 所在中心線線段切線角，包裝到 ±π | rad |
| rear_speed／front_speed | 車身速度（運動學模型無側滑） | m/s |
| front_curvature | 前車所在中心線曲率，左彎正 | 1/m |
| rear_dd_early／front_dd_early | d(t−0.2) − d(t−0.4) | m |
| rear_dd_late／front_dd_late | d(t) − d(t−0.2) | m |
| d_target | 後車當下實際採用的目標 d | m |

歷史不足 0.4 s 的樣本略過；歷史窗或標籤窗內有碰撞矩形重疊也略過。
`--randomize-probes` 讓每場隨機先左或先右、左右交替，每次幅度取 0.3／0.45／0.6 m，整場 hold 取 1.0～2.5 s；
使用獨立亂數，起點、車距、速度抽樣與不加此選項時相同。不加時 probe_follow 行為不變。

```powershell
.\.venv\Scripts\python.exe interaction_gp.py collect --features history --randomize-probes --episodes 20 --seconds 15 --seed 42 --output data/training/my_history
# 消融：同一資料、同一切分，只改輸入欄位
.\.venv\Scripts\python.exe interaction_gp.py train --data data/training/my_history --max-train 0 --output models/gp/h_basic --feature-subset gap rear_d rear_heading_error rear_speed front_d front_heading_error front_speed front_curvature
.\.venv\Scripts\python.exe interaction_gp.py train --data data/training/my_history --max-train 0 --output models/gp/h_history --feature-subset gap rear_d rear_heading_error rear_speed front_d front_heading_error front_speed front_curvature rear_dd_early front_dd_early rear_dd_late front_dd_late
.\.venv\Scripts\python.exe interaction_gp.py train --data data/training/my_history --max-train 0 --output models/gp/h_action
```

控制條件的收集（需 `--randomize-probes`）：

- `--probe-amplitudes 0.6`：側向試探幅度候選，預設 0.3／0.45／0.6 m。
- `--speed-strata 3 4 5 6 7`：前車巡航速度依區間輪流分配。同一區間依序為（先左、靜止）（先右、靜止）（先左、移動）（先右、移動），場數須為 4×區間數的倍數。
- `--episode-prefix speed06`：場次 ID 前綴，合併不同資料集時避免重複。

不加這三項時，同一 seed 產生的資料與之前相同。

```powershell
.\.venv\Scripts\python.exe interaction_gp.py collect --features history --randomize-probes --probe-amplitudes 0.6 --speed-strata 3 4 5 6 7 --episodes 16 --seconds 30 --seed 43 --episode-prefix speed06 --output data/training/my_speed06
```

攻方實際超車收集（近距離、並排與超越），後車改用 `relative_overtake` 對 `defend`：

- 攻方候選與橫移距離同 `interaction_overtake`：7 個訓練過的目標、3.0 m。
- 每場輪流使用 `--deviation-weights`；前車速度依 `--speed-strata` 平衡；攻方比前車快 `--speed-advantage`；初始車距 `--initial-gap`。兩車都移動中起步。
- 碰撞矩形重疊的樣本仍會被排除，所以真正接觸的時段不在資料中。

```powershell
.\.venv\Scripts\python.exe interaction_gp.py collect --features history --rear-strategy relative_overtake --speed-strata 3 4 5 6 7 --deviation-weights 0 1 3 --speed-advantage 0.5 1.5 --initial-gap 1.5 4 --episodes 36 --seconds 20 --seed 44 --episode-prefix close --output data/training/my_close
```

完整 model.pkl 很大（例如 M1-30 為 131 MB），只在本機。可轉成可放進 Git 的精簡檔，預測與原檔一致，版本紀錄見 [models/registry.md](models/registry.md)：

```powershell
.\.venv\Scripts\python.exe interaction_gp.py export --model models/gp/<模型目錄>/model.pkl --output models/compact/<名稱>.npz
```

不給 `--feature-subset` 時使用全部欄位。評估子集與等速基準一律由全部欄位計算，三組可直接比較；
歷史資料的等速基準為 `front_dd_late/0.2×0.5`，另報告後車橫移中（|rear_dd_late|>0.02 m）與保持中的誤差。
model.pkl 記錄實際輸入名稱、單位與歷史長度；此模型尚未接入規劃器。

## 試探互動第一階段：0.5 秒反應預測

新增獨立入口 `interaction_gp.py`，重用正式雙車步進與既有 GP kernel。
這個入口使用圓角矩形、rear=probe_follow、front=defend；不載入 GP 控制策略。
防守配置固定，初始位置、2～5 m 車距、3～7 m/s 前車巡航速度依 seed 改變，
後車巡航為前車＋2 m/s，各場交替使用靜止／移動中起步；目前試探順序固定左、回中、右、回中。

```powershell
.\.venv\Scripts\python.exe interaction_gp.py collect --episodes 20 --seconds 15 --seed 42 --output data/training/interaction_phase1
.\.venv\Scripts\python.exe interaction_gp.py train --data data/training/interaction_phase1 --seed 42 --max-train 800 --output models/gp/interaction_phase1
```

輸出目錄必須不存在。原始狀態每 0.02 s 保存於 raw_states.csv，後車同時刻決策及控制保存於 actions.csv；
samples.csv 每 0.1 s 抽樣，答案為 0.5 s 後前車 d 減當下 d。
實際步長取自 main，metadata 保存設定及完整中心線；標籤時間須為步長整數倍。
跨碰撞區間、第一筆缺少歷史資料及末尾缺少未來資料的樣本會略過，原因與數量保留。

七項輸入依序為：gap（前車 s−後車 s）、rear_d、front_d、relative_speed（後車−前車）、
front_speed、front_lateral_speed、front_curvature。
橫向速度用過去至多 0.1 s 差分；曲率用已知中心線前後各 0.5 m 三點估計，左彎為正。
注意 gap 的符號與舊 relative_s 相反，前車在前方時為正；這版只適用短距跟車的最短有號車距。
所有輸入可由當下／過去觀測及地圖取得，前車內部策略資訊不作輸入。
後車動作本階段只記錄供診斷，尚未用於比較不同假設動作的後果。

依 episode 約 60／20／20% 切分訓練、驗證、測試，scaler 與 GP 僅擬合訓練資料。
預設最多隨機取 800 筆訓練樣本以控制完整 GP 計算量，樣本索引與 seed 存入 report.json。
本版固定模型設定，驗證集先作獨立評分，未依驗證結果調參；測試集不參與擬合。
比較 GP、固定 Δd=0、當下橫向速度×0.5 三者的 MAE／RMSE，並分直線、彎道、側向位置及策略模式報告。
report.json 另含兩倍標準差的實際涵蓋率、樣本分布、kernel、擬合警告與版本；
model.pkl、predictions.csv、evaluation.png 分別保存模型、預測及測試散點圖。

此模型是離線互動預測，與原本 s→d 路線模型的格式不同，不能直接填進 GP_MODEL_PATHS。
第一階段尚未接入規劃器，也未提供完整未來軌跡或碰撞保證。
原始資料與模型沿用 .gitignore，本次小型結果摘要放在 reports/experiments/2026-11-17_interaction_gp/。

## 既有路線模型與通用訓練入口

`train_gp.py` 讀取收集程式輸出的資料夾，訓練一個答案欄位，並保存模型與測試結果。
沿用原有 `samples.csv` 與 `metadata.json` 格式，不需要重新收集或修改 `gp_demo.py`。

目前收集程式預設學習 progress 對手的同時刻 **s → d**。
訓練完成後，可在 `main.py` 的 `GP_MODEL_PATHS` 指定輸出的 `model.pkl`，
接入相對進度進攻；操作與預測假設見 [README](README.md#在進攻策略使用-gp)。
`python plot_gp_route.py` 可離線輸出整圈路線圖；`python main.py --realtime` 會顯示碰撞檢查使用的 GP 預測。
先在 `training_data/fields.py` 選擇或註冊欄位，再執行：

```powershell
.\.venv\Scripts\python.exe generate_training_data.py --episodes 5 --seconds 20 --stride-steps 5
.\.venv\Scripts\python.exe train_gp.py --data data/training/生成的資料夾名稱
```

只有一個答案時不用再寫 `--target`；即使你在 fields.py 新增答案名稱，訓練程式也會自動讀取。
同時刻模式會明確顯示「同時刻對應」，並將 `target_timing="current"` 與時間間隔 0 寫入報告及模型。

## 使用方式

在專案根目錄使用原有 `.venv`：

```powershell
.\.venv\Scripts\python.exe train_gp.py --data data/training/20261105T065547Z_270c5e64
```

請將 `--data` 後面換成你要使用的資料夾。程式不開視窗，預設將結果存到
`models/gp/<UTC時間戳與識別碼>/`。原始資料不會被修改，已有模型目錄不會被覆寫。
讀取與輸出路徑都限制在本專案內。

指定結果存放位置：

```powershell
.\.venv\Scripts\python.exe train_gp.py --data data/training/20261105T065547Z_270c5e64 --output models/gp/first_run
```

訓練需要既有環境中的 NumPy、Matplotlib 與 scikit-learn。
`requirements-gp.txt` 記錄包含 scikit-learn 1.7.2 的相依套件清單；程式不會自行安裝任何套件或 Python。
只使用基本模擬功能時，仍可沿用原本的 `requirements.txt`。

## 訓練做了什麼

1. 讀取輸入欄位、答案欄位與預測時間。欄位順序依 metadata，不固定寫死四個輸入。
2. 依整場模擬的 `episode_id` 分成訓練與測試資料，同一場的不同配對不會被拆開。
3. 只使用訓練資料建立標準化器，讓不同輸入尺度便於比較。
4. 使用「變化幅度 × RBF 相似程度 + 雜訊」的 GP 學習指定答案。
5. 對未參與訓練的 episode 預測平均值及標準差，計算誤差並儲存。

輸入標準化器與 GP 使用同一個 `Pipeline`，保存後能一起載入。
答案透過 `normalize_y=True` 標準化，輸出的平均值、標準差與誤差已轉回原本單位。
GP 的 kernel 參數只使用訓練集最佳化；測試集只評分。
保存的是這個接受評估的模型，不會再把測試資料加入重訓。

原有資料預測的是 **0.02 秒後**的位移，訓練程式會自動沿用；新的同時刻資料則使用時間間隔 0。
若要學習 0.1 秒後的答案，需要先用適當的預測時間生成或重建資料，
不能只在訓練時把時間名稱改成 0.1 秒。

## 最常用的參數

| 命令列選項 | 預設 | 意義 |
| --- | --- | --- |
| `--data` | 必須指定 | 收集程式輸出的資料夾，至少需要 samples.csv 與 metadata.json |
| `--target` | 唯一答案自動選用 | 多答案資料優先相容舊的 opponent_delta_d，否則需明確指定 |
| `--test-fraction` | `0.2` | 保留作測試的 episode 比例，而非逐筆資料比例 |
| `--seed` | `42` | 固定切分與參數搜尋，方便比較結果 |
| `--output` | 自動建立新目錄 | 模型與評估輸出位置，必須尚不存在 |

例如 5 個有有效樣本的 episode，預設用 4 場訓練、1 場測試。
只有一場有效 episode 時會報錯；測試比例若使訓練集沒有任何一場，也會報錯。
只有一場測試的結果容易受該場情境影響，應視為初步評估。
比較設定時請固定 seed；若反覆根據同一測試集選參數，之後需另外保留未用過的測試資料。

若資料中也收集了 `opponent_delta_speed`，可另存一個速度變化模型：

```powershell
.\.venv\Scripts\python.exe train_gp.py --data data/training/你的資料夾 --target opponent_delta_speed
```

第一版每次訓練一個答案。輸入自動使用資料中的全部 feature，新增 feature 後不需修改訓練流程。
資料若沒有指定答案，程式會列出可用答案，不會自動把輸入當成答案。

## 八輸入、動作條件式互動 GP

`train_action_gp.py` 沿用執行當下的 `main.py` 賽道、車輛、策略、控制器與碰撞矩形。
各場從 d=0、靜止起步，只隨機平移整組車輛的賽道起點，保留原車距與目標速度。
固定收集 20 場，以 episode 切成 12 場訓練、4 場驗證、4 場測試，預設每場 25 s。
樣本間距 0.1 s，答案為前車 `d(t+0.5)-d(t)`；排除未來不足、無歷史速度與標籤窗內碰撞矩形重疊的樣本。

| 輸入名稱 | 定義 | 單位 |
| --- | --- | --- |
| `gap` | 前車 s − 後車 s，封閉賽道最短有號距離 | m |
| `rear_d` | 後車橫向位置 | m |
| `front_d` | 前車橫向位置 | m |
| `relative_speed` | 後車車身速度 − 前車車身速度 | m/s |
| `front_speed` | 前車車身速度 | m/s |
| `front_lateral_speed` | 前車 d 在過去 0.1 s 的差分速度 | m/s |
| `front_curvature` | 前車所在中心線的有號曲率，前後各 0.5 m 三點估計 | 1/m |
| `d_target` | 後車當下真正採用的候選目標 d | m |

```powershell
.\.venv\Scripts\python.exe train_action_gp.py --data data/training/my_action_gp --output models/gp/my_action_gp --pair vehicle-1 vehicle-2 --seconds 25 --seed 42
```

資料與模型目錄必須是新路徑。此入口使用全部 12 場的有效訓練樣本，不套用舊七輸入入口的 800 筆上限。
沿用 StandardScaler + Constant×ARD RBF + WhiteKernel，GP 正規化答案，無額外 optimizer restart；
超參數僅依訓練資料擬合，驗證與測試只用來評估，不用於選參數。
若先加 `--collect-only` 收集，可接續：

```powershell
.\.venv\Scripts\python.exe interaction_gp.py train --data data/training/my_action_gp --output models/gp/my_action_gp --max-train 0
```

`d_target` 使用同一時間步控制前的已選動作；不會把所有未執行候選複製成相同答案。
後續 0.5 s 內策略仍可改目標，因此此資料不能直接當成「固定某候選 0.5 s」的反事實實驗。
本模型沒有自動接入規劃器，亦不能直接替換只接受單一 s 的既有 GP 路線模型。
原七輸入資料、原收集命令及原 s→d 流程保留。

## 基本 GP 的少量設定

設定集中在 `train_gp.py` 頂端的 `TrainingConfig`：

| 設定 | 預設 | 如何理解 |
| --- | --- | --- |
| `length_scale` | `1.0` | 標準化後的相似程度初始尺度，每個輸入都能分別學習 |
| `noise_level` | `0.02` | 標準化答案的雜訊變異數初始值，不是 0.02 公尺 |
| `optimizer_restarts` | `2` | 額外換初始值嘗試搜尋，增加時通常會花更多時間 |

這些初始值會在程式中的界限內最佳化。最後的值會印在終端機，並寫入 report.json。
若最佳化碰到界限，相關提示會保留在報告的 `fit_warnings`，不會自動改變設定或重新訓練。
GP 使用全部訓練樣本；增加資料量與搜尋次數都可能顯著增加訓練時間。

## 如何看結果

| 指標 | 意義 |
| --- | --- |
| MAE | 平均每次差多少；位移答案若是 0.005 m，即平均差 5 mm |
| RMSE | 對較大的錯誤更敏感，與答案使用相同單位 |
| 固定預測 0 的誤差 | 簡單比較基準。對 `opponent_delta_d`，相當於永遠猜對手不橫移 |
| 固定預測訓練平均值的誤差 | 比較 GP 是否優於每次都猜同一個平均數；平均值只從訓練集計算 |

其他答案的零值基準只是「固定猜 0」的參考，不一定代表不移動。
對同時刻 d，零值基準表示永遠猜對手在中心線上。
若 GP 的測試 MAE 沒有優於兩個簡單基準中較好的結果，終端機會明確提醒。
標準差描述模型假設下的不確定程度，不保證等於實際誤差。

結果資料夾包含：

- `model.pkl`：標準化器與 GP、輸入名稱與順序、答案名稱與單位、預測時間及套件版本。
- `report.json`：設定、訓練／測試 episode 與樣本數、MAE／RMSE、學到的參數、調參提示及訓練時間。
- `predictions.csv`：測試樣本 ID、加上 `feature_` 前綴的輸入欄位、actual、predicted_mean、predicted_std、error。
  `error` 定義為預測值減實際值；答案名稱與單位記錄在 report.json。
- `evaluation.png`：實際值與預測值的散點圖，越接近斜線越準。
  單一輸入且為同時刻模式時，另加一張輸入對應答案的曲線圖，例如 s 對 d。
  曲線與不確定範圍由模型在整段輸入上計算，並分別標示訓練觀測與測試實際值。

## 載入自己的模型預測

```python
from pathlib import Path
import pickle
import numpy as np

with Path("models/gp/first_run/model.pkl").open("rb") as stream:
    saved = pickle.load(stream)

# 名稱依你訓練時選用的輸入而定；用名稱排列，避免欄位順序放錯。
observation = {
    "opponent_s": 12.0,
    "relative_s": -3.0,
    "relative_d": 0.4,
    "relative_speed": 2.0,
    "opponent_d": -0.1,
}
X = np.array([[observation[name] for name in saved["feature_names"]]], dtype=float)
mean, std = saved["pipeline"].predict(X, return_std=True)
print(saved["target_spec"], saved["label_horizon_seconds"])
print("預測：", mean[0], "標準差：", std[0])
```

輸入保留原始物理單位，Pipeline 會套用訓練時保存的縮放。
若未來增加輸入欄位，observation 也需要提供該欄位。
pickle 請只載入自己產生且信任的模型，並使用相容的套件版本。

## 測試

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gp_training
```

測試涵蓋整場資料切分、測試集不參與標準化、擴充輸入及選擇答案、
模型儲存與重讀、誤差計算及完整命令列流程。
測試暫存檔全部放在專案內 `.cache/`。
