# GP 模型訓練

`train_gp.py` 讀取收集程式輸出的資料夾，訓練一個答案欄位，並保存模型與測試結果。
沿用原有 `samples.csv` 與 `metadata.json` 格式，不需要重新收集或修改 `gp_demo.py`。

目前收集程式預設學習 progress 對手的同時刻 **s → d**。
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

## GP 的少量設定

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
