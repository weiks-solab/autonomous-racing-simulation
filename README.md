# Racing Overtake Simulator

輕量的 Python 2D 賽車模擬器。每輛車各自使用運動學自行車模型與 Pure
Pursuit 控制器，車輛數量可自由增加。

目前示範三輛車：`vehicle-1` 是超車的 ego，`vehicle-2` 是可啟用防守的
對手車，`vehicle-3` 持續跟隨中心線。策略只決定橫向目標與平滑路徑；
方向盤角度及速度控制仍由 Pure Pursuit 負責。

## 安裝與執行

需要 Python 3.10 以上、NumPy 與 Matplotlib，沒有新增其他依賴。

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

預設模擬 21 秒，再顯示中心線、賽道邊界、各車軌跡與車身。即時觀看：

```bash
python3 main.py --realtime
```

停用防守時，所有指派 `defend` 的車改為跟隨中心線，其他策略保持原設定：

```bash
python3 main.py --realtime --no-defense
```

若環境使用 `python` 指令，將上述 `python3` 換成 `python` 即可。

## 想調整參數，先看 main.py 開頭

常用設定集中在 `main.py` 的「常用參數」區，每個防守參數都有簡短中文註解。

| 設定 | 用途 |
| --- | --- |
| `VEHICLE_SPECS` | 每輛車的 ID、起始位置、目標速度；新增一列就能增加車輛 |
| `STRATEGY_ASSIGNMENTS` | 指定每台車使用什麼策略，以及防守誰 |
| `DEFENSE_ENABLED` | 是否啟用防守 |
| `SIMULATION_SECONDS` | 模擬多久 |
| `TRACK_*` | 橢圓大小、點數與賽道寬度 |
| `MODEL_CONFIG` / `VEHICLE_GEOMETRY` | 時間步長、軸距、速度限制與車身尺寸 |
| `LOOKAHEAD_DISTANCE` | Pure Pursuit 的前視距離 |
| `OVERTAKING_CONFIG` | 所有進攻車使用的超車參數 |
| `DEFENSE_CONFIG` | 所有防守車使用的成本、觸發條件與路徑參數 |
| `OFFSET_PATH_CONFIG` | 臨時路徑的取樣間距與長度 |

建議一次改一個防守參數，先從以下幾個開始：

| `DEFENSE_CONFIG` 參數 | 增大時的效果 |
| --- | --- |
| `q_block` | 更重視與 ego 的橫向位置對齊；設為 0 停止主動防守 |
| `center_weight` | 更偏好中心線 |
| `smooth_weight` | 更不喜歡預測路徑中突然的橫向變化 |
| `transition_length` | 橫移需要更長距離，動作較緩慢 |
| `minimum_gap` | 更早停止追著近距離車輛的橫向位置移動 |
| `hold_time` | 原路徑仍可行時，更久才允許換目標 |
| `switch_improvement` | 新候選需要改善更多成本，才值得切換 |

左右候選由 `candidate_offsets` 設定；策略也會加入目前位置、目前目標及
ego 當下的橫向位置。候選超出可用寬度時會被排除。

`create_display_options()` 緊接在參數區下方，可以切換軌跡、車身、控制目標、
策略狀態與臨時路徑的顯示。

## 替每台車指定策略

策略演算法分別放在 `strategy/`，此次模擬要如何分配，則在 `main.py` 設定：

```python
STRATEGY_ASSIGNMENTS = {
    "vehicle-1": StrategyAssignment("overtake"),
    "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
    "vehicle-3": StrategyAssignment("centerline"),
}
```

| 策略名稱 | 功能 | 演算法檔案 |
| --- | --- | --- |
| `centerline` | 跟隨中心線，不主動進攻或防守 | `strategy/centerline.py` |
| `overtake` | 找前方較慢的車，選擇左右超車路徑 | `strategy/overtaking.py` |
| `defend` | 依成本選擇橫向防守位置 | `strategy/defending.py` |

例如想改成第二台車進攻，第三台車防守第二台，只需改成：

```python
STRATEGY_ASSIGNMENTS = {
    "vehicle-1": StrategyAssignment("centerline"),
    "vehicle-2": StrategyAssignment("overtake"),
    "vehicle-3": StrategyAssignment("defend", target_vehicle_id="vehicle-2"),
}
```

策略不會替你改速度或初始位置。若希望進攻車能追上防守車，也要在
`VEHICLE_SPECS` 調整相對位置及目標速度。

沒有列在對照表裡的車，預設使用 `centerline`。對照表中的車輛 ID 必須出現在
`VEHICLE_SPECS`；防守對象必須存在，且不能是自己。防守對象不必使用進攻策略，
只是它必須符合後方追近的觸發條件，防守行為才會啟動。

同一種策略可以分配給多台車；每台車各自保有狀態，不會共用已選側邊、動作起點
或目標鎖定計時。目前同一種類的車共用設定參數，仍各自獨立決策，沒有聯合規劃。
多車同時進攻或防守也不代表能保證避免碰撞。

`StrategyManager` 會先讓所有一般／進攻車更新，再把它們新選的路徑交給防守車。
同一階段的車都使用相同觀測快照，不會因車輛在設定表中的排列順序而輪流看到
其他同階段車的新路徑。進攻車仍看到上一個模擬步留下的防守路徑。

## 未來新增自己的策略

`strategy/base.py` 定義共用介面與 `StrategyDecision`：

```python
def update(self, vehicle, others, track, time_seconds) -> StrategyDecision:
    ...
```

輸入只有自己的觀測、其他車的觀測、賽道與目前時間。輸出包含狀態、`target_d`
及可選的 `LateralManeuver`，不包含方向盤或加速度命令。`maneuver=None` 代表
使用共同中心線；要追蹤非零橫向目標，必須同時提供對應的平滑動作。

新增策略時，把演算法放進 `strategy/` 的獨立檔案，在 `strategy/assignment.py`
的策略名稱檢查與 `create_strategy()` 加入名稱及建構方式，再於對照表選用。
模擬迴圈使用相同介面，不需要增加車輛 ID 的特殊判斷。

既有進攻與防守演算法的輸出格式不同，因此 `assignment.py` 用兩個簡單的包裝類別
轉成共用格式，保留原來演算法。`sim/strategy_history.py` 負責記錄每台車的狀態
與橫向目標，讓執行結果也能跟著分配表改變。

## 防守策略怎麼判斷？

閱讀順序建議是 `strategy/defending.py` 的 `update()` →
`_candidate_targets()` → `_evaluate_candidate()` → `_check_path()`。

1. 指定 ego 在後方、距離夠近且速度較快，才啟用防守成本。
2. 產生幾個平滑橫移候選，使用固定速度作短時間幾何預測。
3. 排除預測車身角點超出賽道邊界餘量，或預測會與任何其他車重疊的候選。
4. 比較可行候選的成本，選擇最小者，輸出 `target_d` 與 `LateralManeuver`。
5. 鎖定目標一小段時間，避免每次更新都左右換邊；原路徑不可行時可提前換目標。
6. 太近或並排時停止追逐橫向位置；威脅解除後，選擇可行的返回路徑。

防守成本對應：

```text
J_O = J_race + q_block × sum((d_opponent[k] - d_ego_now)²)

J_race = center_weight × sum(d_opponent[k]²)
       + smooth_weight × sum((d_opponent[k] - d_opponent[k-1])²)
```

`k=1..N` 是未來的預測時間點，`d_opponent[0]` 為對手車目前位置。
`N` 是預測步數，不是車輛數量。`d_ego_now` 在一次成本評估中固定，沒有預測
ego 未來意圖。`J_race` 是偏好中心線與平順移動的簡化成本，並非圈速最佳化。

防守項只在偵測到追近威脅時啟用。所有成本都是取樣點的加總，改變
`prediction_step` 也會改變成本尺度與平順項的相對影響，需要重新觀察權重效果。

狀態顯示為 `CRUISE`（正常行駛）、`DEFEND`（防守）、`HOLD`（保持／暫停換位）
與 `RETURN`（返回中心線）。圖上也會顯示選定路徑的 `race` 與 `block` 成本；
終端會列出狀態轉換、橫向目標與碰撞紀錄。

## 預測與限制

`strategy/prediction.py` 共用幾何預測：車輛保持目前速度，依照已選定的橫向
動作前進；未提供動作時保持目前 `d`。防守成本仍使用 ego 當下的位置，
已選定的路徑只供碰撞檢查使用。ego 評估超車候選時，也能使用對手已選定的路徑。
這是模擬器內直接提供的路徑資訊，沒有感測器或通訊模型。

這些檢查只涵蓋有限預測時間及離散取樣點，沒有模擬追蹤誤差與未來加速度，
不能保證任何參數組合都不碰撞。ego 開始超車後仍保留原有的側邊鎖定行為。
若全部防守候選都不可行，策略會保留上一條路徑，並在決策的 `reason` 回報
`no feasible candidate`；沒有緊急控制或碰撞處理。

對手車維持固定的目標速度，不會故意煞車。碰撞偵測只回報矩形車身重疊，
不會改變車輛運動。沒有博弈、MPC、動態車輛物理或賽事規則。

## 賽道座標與程式分工

- `s` 從中心線第一點開始累積，沿點序增加，在封閉賽道上繞回起點。
- `d` 向行車方向的左側為正，右側為負。
- `core/`：車輛狀態、車輛模型、幾何與 Pure Pursuit，不依賴視覺化。
- `strategy/`：超車、防守與高階路徑，不接收整個模擬器物件。
- `sim/`：每輛車的執行狀態與 Matplotlib 視覺化。
- `main.py`：參數、組裝與執行模擬。

## 測試

```bash
python3 -m unittest
```

沒有圖形介面的環境可在專案內存放 Matplotlib 快取並執行：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib python3 -m unittest
MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib python3 main.py
MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib python3 main.py --no-defense
```
