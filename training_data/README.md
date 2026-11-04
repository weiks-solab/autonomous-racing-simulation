# 訓練資料生成模組

本模組收集模擬車輛的實際狀態，建立「當下觀察 → 未來反應」的數值迴歸資料。
不負責訓練 GP，不需要 scikit-learn，也不會變更 `gp_demo.py`。

原有 `main.py`、`core/`、策略與視覺化程式維持原樣。新增的無視窗模擬入口
`sim/data_generation.py` 直接使用既有策略管理器、Pure Pursuit 與運動模型。
`generate_training_data.py` 讀取 `main.py` 的車輛、賽道及控制／策略設定，但不執行其主程式。

## 執行

在專案根目錄、已安裝 `requirements.txt` 的 Python 環境中：

```powershell
python generate_training_data.py
```

若使用專案的 `.venv`，也可以明確指定：

```powershell
.\.venv\Scripts\python.exe generate_training_data.py --episodes 10 --seconds 6 --seed 42
```

預設生成 5 個各 4 秒的 episode，將結果放在專案內新的
`data/training/<UTC時間戳與隨機識別碼>/`。不開啟視窗，不會覆寫已有輸出目錄。
`--seconds` 必須是車模步長的整數倍，且足以涵蓋答案的預測時間。

```powershell
python generate_training_data.py --episodes 3 --seconds 4 --horizon-steps 5 --stride-steps 2 --output data/training/example
```

以目前 `main.py` 的步長 `0.02 s` 而言，這個例子每 `0.04 s` 取一筆輸入，
答案則是對手在 `0.1 s` 後相對現在的 `d` 變化。

未指定配對時，優先從 `defend` 指派推導出 `(被防守的車, 防守車)`；
若沒有防守指派，則收集每台進攻車與其他車的有向配對。
也可明確指定一組或多組，不限於兩台車：

```powershell
python generate_training_data.py --pair vehicle-1 vehicle-2
```

`--pair EGO_ID OPPONENT_ID` 可以重複指定。兩個 ID 必須不同且存在於 `VEHICLE_SPECS`。
`--no-defense` 將防守車改為既有的中心線跟隨模式，仍會收集相同配對的資料。

## 預設欄位與形狀

每筆資料的識別欄位為 `episode_id, step_index, time_s, ego_id, opponent_id`，
不會自動放入模型輸入。

| 類別 | 欄位 | 定義 | 單位 |
| --- | --- | --- | --- |
| 輸入 | `relative_s` | 我方 s 減對手 s；封閉賽道取最短有號距離 | m |
| 輸入 | `relative_d` | 我方 d 減對手 d | m |
| 輸入 | `relative_speed` | 我方 `VehicleState.velocity` 減對手的值 | m/s |
| 輸入 | `opponent_d` | 對手當下相對中心線的 d | m |
| 答案 | `opponent_delta_d` | 對手 `d(t + Δt) - d(t)` | m |

位置使用後軸參考點，d 沿中心線前進方向左正右負。
`relative_s > 0` 表示我方在前，封閉賽道的範圍為 `[-L/2, L/2)`；
恰好相距半圈時映射到負值。開放賽道直接相減。
相對速度是兩車標量速度的差，沒有投影成沿賽道的 s 速度。

預設 `label_horizon_steps=1`、`sample_stride_steps=1`。
答案是車模更新後的實際 d 差，不是策略目標、候選路徑或橫向速度。
單一 episode 的所有車會先完成同一步更新，再收集一次同步觀察，包含初始與最後狀態。

內部輸入固定為 `X.shape == (N, F)`，答案固定為 `Y.shape == (N, T)`。
預設為 `(N, 4)` 與 `(N, 1)`；沒有有效樣本時也保留正確的空陣列形狀。

```python
from training_data import load_dataset

dataset = load_dataset("data/training/example")
X_train = dataset.X
y_train = dataset.Y[:, 0]  # 單一答案時，可傳給目前的 GP 迴歸器
```

這不會自動修改或替換 `gp_demo.py` 中的示範資料。
若要縮放特徵，應在訓練端只用訓練集合擬合縮放器，並保存供推論使用。

## 輸出檔案

每個輸出目錄包含：

- `samples.csv`：識別欄位、選定輸入及答案，順序與 schema 一致。
- `raw_states.csv`：每步每台車的狀態、s/d、幾何尺寸、橫移描述與碰撞紀錄。
- `metadata.json`：格式版本、欄位順序與單位、符號定義、步長及預測時間、
  完整中心線、排除統計、seed、初始條件與控制／策略設定。

CLI 的初始條件會依固定 seed 改變整體起始位置、各車縱向位置、d、實際車速及目標速度。
策略參數維持固定。車體起始位置會檢查碰撞及賽道邊界。
各 episode 重新建立車輛、控制器與獨立策略物件。
程式介面可透過 `RandomizationConfig` 調整隨機範圍。

預設排除預測區間內我方或對手涉及的碰撞，包含與第三台車的碰撞。
其他兩台車之間的碰撞不會排除這組樣本。`--keep-collisions` 可保留碰撞區間。
缺少配對車輛、沒有足夠未來狀態，以及提取值為 NaN/Inf 的樣本會記錄排除原因。
不補零、不跨 episode 配對、不替換樣本中的對手。沒有有效樣本時 CLI 會報錯。
目前不自動排除模擬途中出界的狀態，可用原始資料另行分析與篩選。

輸出與載入路徑都限制在本專案內。修改欄位或預測時間後，請使用新的輸出目錄。
目前 CSV/JSON 不保存提取器的 Python 實作；自訂欄位的提取器需要由呼叫端管理版本。

## 擴充欄位

內建另有 `ego_speed`、`opponent_speed` 輸入，以及 `opponent_future_d`、
`opponent_delta_speed` 答案，可直接選用：

```powershell
python generate_training_data.py --features relative_s relative_d relative_speed opponent_d opponent_speed --targets opponent_delta_d opponent_delta_speed
```

新增欄位時註冊一個提取器，再設定輸出順序即可。
輸入提取器收到的 `FeatureContext` 沒有未來狀態；
答案提取器的 `TargetContext` 才包含 `current` 與 `future`。
第一版的輸入介面僅提供當下快照，歷史特徵未來可透過擴充 context 加入。
所有欄位目前都是數值標量；分類或向量資料應先拆欄或另外定義編碼與模型。

下例直接讀取原始紀錄，新增輸入並重新生成多答案資料，不必重新模擬：

```python
from training_data import (
    DatasetSchema, FieldSpec, VehiclePair, build_dataset,
    create_default_registry, export_dataset, load_recordings,
)

track, sampling, episodes = load_recordings("data/training/example")
registry = create_default_registry()
registry.register_feature(
    FieldSpec("opponent_yaw", "rad", "Current opponent world-frame yaw."),
    lambda context: context.opponent.state.yaw,
)
schema = DatasetSchema(
    feature_names=("relative_s", "relative_d", "relative_speed", "opponent_d", "opponent_yaw"),
    target_names=("opponent_delta_d", "opponent_delta_speed"),
)
dataset = build_dataset(
    episodes, track, (VehiclePair("vehicle-1", "vehicle-2"),),
    sampling, schema, registry,
)
export_dataset(dataset, episodes, track, "data/training/example_extended")
```

也可以用 `registry.register_target(FieldSpec(...), extractor)` 新增答案。
每個 registry 都獨立，未知或重複欄位會報錯。
若新欄位需要原始資料尚未記錄的資訊，仍需擴充收集並重新模擬。

## 訓練時的資料切分

以 `episode_id` 分組切分訓練、驗證與測試集合；同一次模擬的不同配對也應放在同一組。
避免把時間相鄰且重疊的樣本隨機分散到不同集合，造成評估過於樂觀。
若對手的策略參數改變，需決定分開建模或新增能描述差異的輸入。

目前四個輸入未包含航向誤差、橫向動作歷史與策略內部狀態，
相同輸入可能有不同答案。模組負責忠實收集資料，不保證這四個特徵能唯一決定反應。

## 測試

```powershell
python -m unittest tests.test_training_data tests.test_training_data_generation
python -m unittest
```

測試涵蓋座標符號、跨圈、時間配對、多車配對、區間碰撞、欄位擴充、
CSV/JSON 重讀與原始資料重建，以及既有車模／策略的無視窗執行。
測試暫存資料位於專案內 `.cache/`。
