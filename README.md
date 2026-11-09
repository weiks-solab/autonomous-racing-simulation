# Racing Overtake Simulator

輕量的 Python 2D 賽車模擬器。每輛車各自使用運動學自行車模型與 Pure
Pursuit 控制器，車輛數量可自由增加。

目前示範由 `vehicle-1` 使用成本式進攻，`vehicle-3` 使用純進度策略。
車數與策略可在 `main.py` 自行調整，未指派策略的車持續跟隨中心線。
策略只決定橫向目標與平滑路徑；
方向盤角度及速度控制仍由 Pure Pursuit 負責。

## 安裝與執行

基本模擬需要 Python 3.10 以上、NumPy 與 Matplotlib。
GP 訓練另外使用 scikit-learn，相依清單見 `requirements-gp.txt`。

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

預設在圓角矩形賽道模擬 21 秒，再顯示賽道邊界、各車軌跡與車身。即時觀看：

```bash
python3 main.py --realtime
```

停用防守時，所有指派 `defend` 的車改為跟隨中心線，其他策略保持原設定：

```bash
python3 main.py --realtime --no-defense
```

若環境使用 `python` 指令，將上述 `python3` 換成 `python` 即可。

## 收集資料與訓練 GP

在專案根目錄使用既有 Python 環境，先生成資料，再指定輸出的資料夾訓練：

```bash
python3 generate_training_data.py --episodes 5 --seconds 20 --horizon-steps 5
python3 train_gp.py --data data/training/生成的資料夾名稱
```

目前時間步長為 0.02 s，`--horizon-steps 5` 表示答案為 0.1 s 後的位移。
訓練依整場模擬切分資料，保存標準化器、GP、誤差報告與評估圖。
資料及模型保留在本機，不納入 Git；程式不會自動安裝套件。
詳細操作見 [資料生成說明](training_data/README.md) 與 [GP 訓練說明](GP_TRAINING.md)。
本次功能與設定調整記錄於 [CHANGELOG.md](CHANGELOG.md)。

## 想調整參數，先看 main.py 開頭

常用設定集中在 `main.py` 的「常用參數」區，每個防守參數都有簡短中文註解。

| 設定 | 用途 |
| --- | --- |
| `VEHICLE_SPECS` | 每輛車的 ID、起始位置、目標速度；新增一列就能增加車輛 |
| `STRATEGY_ASSIGNMENTS` | 指定每台車使用什麼策略，以及防守誰 |
| `DEFENSE_ENABLED` | 是否啟用防守 |
| `SIMULATION_SECONDS` | 模擬多久 |
| `TRACK_*` | 地圖選擇、直道與彎道尺寸、點數與賽道寬度 |
| `MODEL_CONFIG` / `VEHICLE_GEOMETRY` | 時間步長、軸距、速度限制與車身尺寸 |
| `LOOKAHEAD_DISTANCE` | Pure Pursuit 的前視距離 |
| `OVERTAKING_CONFIG` | 原本規則式 `overtake` 使用的超車參數 |
| `COST_OVERTAKING_CONFIG` | 新增 `cost_overtake` 的進度、超車獎勵與預測參數 |
| `PROGRESS_CONFIG` | `progress` 的候選目標、橫移長度、決策週期與自車預測時長 |
| `DEFENSE_CONFIG` | 所有防守車使用的成本、觸發條件與路徑參數 |
| `OFFSET_PATH_CONFIG` | 臨時路徑的取樣間距與長度 |

建議一次改一個防守參數，先從以下幾個開始：

| `DEFENSE_CONFIG` 參數 | 增大時的效果 |
| --- | --- |
| `q_block` | 更重視與 ego 的橫向位置對齊；設為 0 關閉對齊成本，碰撞開關獨立 |
| `progress_weight` | 更重視預估前進進度，設為 0 關閉進度獎勵 |
| `smooth_weight` | 更不喜歡預測路徑中突然的橫向變化 |
| `transition_length` | 橫移需要更長距離，動作較緩慢 |
| `detection_range` | 在更遠的後方範圍內啟用對齊成本 |
| `decision_period` | 更久才重新比較候選成本 |
| `path_sample_spacing` | 幾何路徑取樣較疏；單位為公尺，較小可提高距離換算精度 |
| `collision_horizon` | 防守碰撞條件檢查更長的未來時間；不改變成本評估時長 |

`collision_check_enabled=False` 可停用防守碰撞條件，回到原本只依成本選路徑。

固定候選由 `candidate_offsets` 設定，中心線 `0` 一定列入；策略不會額外加入
目前位置、目前目標或 ego 的位置。可用橫向範圍為
`|d| <= track.width / 2 - vehicle.width / 2 - safety_margin`，超出範圍的候選
會在評分前排除。賽道窄到無法容納車寬與邊界留距時，會回報設定錯誤。

`create_display_options()` 緊接在參數區下方，可以切換軌跡、車身、控制目標、
策略狀態、臨時路徑與候選快照的顯示。

## 切換賽道與調整直道、彎道

在 `main.py` 頂端設定 `TRACK_NAME`：

```python
TRACK_NAME = "rounded_rectangle"  # 四段直道＋四個 90 度圓弧
# TRACK_NAME = "oval"             # 切回原本的橢圓
```

預設圓角矩形的中心線跨度為 20 × 12 m，彎道半徑 3 m，路寬 2 m。
水平直道各長 14 m、垂直直道各長 6 m，一圈約 58.85 m。
中心在原點，從下方直道左端出發，沿逆時針行駛。
可調整 `TRACK_RECTANGLE_LENGTH`、`TRACK_RECTANGLE_HEIGHT`、
`TRACK_CORNER_RADIUS`；長寬是中心線跨度，不包含兩側路寬。
彎道直徑必須小於兩個跨度，且半徑必須大於半個路寬，以保留四段直道及有效內側邊界。

兩種地圖共用 `TRACK_WIDTH` 與 `TRACK_POINT_COUNT`，預設 500 點。
圓角矩形沿幾何弧長等距取樣，直道與圓弧的切線方向連續；原本橢圓仍沿參數角取樣。
`oval` 尺寸由 `TRACK_SEMI_MAJOR_AXIS` 與 `TRACK_SEMI_MINOR_AXIS` 決定。
賽道建立邏輯在 `core/track_factory.py`，未來新增形狀可統一從 `create_track()` 接入。

模擬與 `generate_training_data.py` 共用 `create_demo_track()`，切圖會同時套用至兩者；
資料匯出仍保存實際中心線、路寬與封閉設定。車輛位置更新、`s/d` 投影、Pure Pursuit
與每車策略介面沿用原有方式。改變彎道尺寸後仍應檢查追蹤誤差，必要時調整速度與前視距離。

## 查看攻防候選路徑

使用即時模式，可以看到進攻與防守決策的候選曲線：

```powershell
.\.venv\Scripts\python.exe main.py --realtime
```

在 `create_display_options()` 設定 `show_candidate_paths` 可開關候選顯示，
`show_candidate_labels` 可開關候選文字與評估時間；`show_temporary_paths`
則控制目前交給 Pure Pursuit 追蹤的路徑。

| 圖上樣式 | 意義 |
| --- | --- |
| 車輛同色的細虛線 | 未選中的候選快照 |
| 車輛同色的粗實線 | 選中的候選／目前追蹤路徑 |
| 灰色點線 | 進攻車被既有可行性判斷排除的候選 |
| 淡色實線 | 車輛實際走過的軌跡 |

防守的可行候選標籤顯示 `d` 與總成本 `J`，選中者加上 `*`；既有策略文字顯示
選中路徑的 `race`、`block` 分項。進攻候選顯示 `LEFT`／`RIGHT`、`d` 與
可行性結果或排除原因，適用原本的規則式 `overtake`。
新策略 `cost_overtake` 則顯示五條 bicycle model 預測軌跡，通過碰撞篩選的
候選標籤列出 `progress`、`pass` 成本與總成本 `J`。被剔除的候選畫成灰色點線，
標示碰撞對象與相對該次評估的未來秒數，不顯示成本或選中標記。

`candidates @ ...s` 表示評估時的模擬時間。候選曲線保留該時刻的位置與速度，
只畫各策略評估時間範圍內的候選。防守與原規則式進攻畫的是幾何參考路徑；
防守啟用碰撞條件且指定對象存在時，曲線使用 `collision_horizon` 的檢查範圍，
停用條件或對象不存在時則使用成本的 `prediction_horizon`。
`cost_overtake` 直接畫控制器與 bicycle model 推進出的狀態，不用理想橫移曲線
代替實際預測結果。防守與成本式進攻在重新評分時更新快照；規則式進攻則在
準備超車時更新左右候選，鎖定側邊後保留，返回或跟隨時清除。

目前追蹤路徑仍隨車輛前進，前方長度由 `OFFSET_PATH_CONFIG.path_length` 設定。
候選快照的座標與評估
結果保持固定，不會以新車位重畫舊決策。非即時模式只顯示模擬最後一幀，
已返回或跟隨中心線的規則式進攻車不會顯示先前超車候選。

候選資料由策略共用輸出單向送往視覺化，不會傳入其他車的觀測或控制器，
也不增加防守碰撞判斷或改變既有決策頻率。

## 替每台車指定策略

策略演算法分別放在 `strategy/`，此次模擬要如何分配，則在 `main.py` 設定：

```python
STRATEGY_ASSIGNMENTS = {
    "vehicle-1": StrategyAssignment("cost_overtake"),
    "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
    "vehicle-3": StrategyAssignment("centerline"),
}
```

| 策略名稱 | 功能 | 演算法檔案 |
| --- | --- | --- |
| `centerline` | 跟隨中心線，不主動進攻或防守 | `strategy/centerline.py` |
| `overtake` | 找前方較慢的車，選擇左右超車路徑 | `strategy/overtaking.py` |
| `cost_overtake` | 五候選 bicycle 預測，以進度與超車獎勵選路徑 | `strategy/cost_overtaking.py` |
| `progress` | 五候選 bicycle 預測，只選最大進度，不預測其他車或篩選碰撞 | `strategy/progress.py` |
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
只要它在後方且位於偵測距離內，就會啟用對齊成本，沒有速度差門檻。

同一種策略可以分配給多台車；每台車各自保有狀態，不會共用已選側邊、動作起點
等資料。目前同一種類的車共用設定參數，仍各自獨立決策，沒有聯合規劃。
多車同時進攻或防守也不代表能保證避免碰撞。

`StrategyManager` 讓所有車使用同一模擬步的觀測快照，更新結果不會在這一步
傳給其他車。防守車收到的其他車觀測會移除 `maneuver`，只提供當下狀態，
因此不會取得對方的候選目標、已選目標或規劃路徑。規則式進攻仍可讀取上一個
模擬步留下的其他車路徑；成本式進攻只依其他車的當下狀態預測，不讀取其路徑。

## 成本式進攻與自行車模型預測

`strategy/cost_overtaking.py` 的 `CostBasedOvertakingStrategy` 獨立實作共用
`update(vehicle, others, track, time_seconds)` 介面，沒有模擬器或 Matplotlib 依賴。
預設每 0.2 秒評估固定五個橫向目標，依可用路寬篩選後，分別產生平滑參考線。
目標相同時沿用原動作起點，切換目標時從目前位置建立新動作。
完成橫移後保留等偏移參考線，避免繞下一圈時重新開始過渡。

每條候選從實際 `(x, y, yaw, velocity)` 開始，用既有 Pure Pursuit 算出轉向與
加速度，再由 `KinematicBicycleModel` 以實際模型時間步長向前推進，預設預測
5 秒。每一步都依預測位置重建前方參考路徑，與實際模擬採用相同追蹤流程。
最後不足一個時間步時，僅縮短最後一步，使所有候選在同一預測時間終點評分。

前車選擇偵測範圍內最近的一台，不要求速度較慢。前車預測維持其**當下速度**，
令加速度為零，以當下的 `d` 作為等偏移參考線，再用該車自己的 Pure Pursuit、
軸距與模型限制預測轉向；不模擬前車將採取哪一個防守目標。所有其他車都採用
相同的定速、等偏移假設，每台每輪只預測一次，供五條候選共用。各車的模型與
控制參數均由 `main.py` 明確傳入。

候選先進行碰撞硬性篩選，通過後才計算成本。檢查涵蓋所有其他車，
不受超車獎勵的 `detection_range` 限制。比較同一預測時間的旋轉車身矩形，
包含 `t=0` 與預測終點；車身接觸也視為碰撞。任一時間點重疊就整條剔除。
使用實際車身尺寸，`safety_margin` 仍只作用於候選橫向目標的路寬限制。
如果車輛預測步長不同，使用雙方時間點的聯集，對缺少的狀態插值，
yaw 沿最短角度銜接後再比較，避免直接用相同陣列索引配對不同時間。

成本只有兩項：

```text
J_progress = -progress_weight × 預測終點累積進度
J_pass     = -overtaking_weight × 超車獎勵
J          = J_progress + J_pass
```

進度由相鄰預測狀態的縱向位移連續累積，跨越封閉賽道起終點不會跳變。
超車獎勵為 0 或 1：以本輪起點為共同基準，當進攻車終點進度大於
「前車初始縱向距離 + 前車預測進度」時給 1；沒有前車時為 0。
這是縱向位置超過的獎勵，不檢查完整車身是否通過。

每次只從通過碰撞篩選的候選中選總成本最低者。碰撞是剔除條件，
不作為額外成本，因此再高的超車獎勵也不能讓被剔除的候選重新入選。
沒有速度差門檻、側邊鎖定、中心線成本、平順成本或強制返回中心線。
`ATTACK` 表示本輪有前車，`PROGRESS` 表示只有進度項。

若全部候選被剔除，顯示 `NO_FEASIBLE_PATH`，沿用上一版的 `target_d`
與完整 `maneuver`，保留原動作起點，模擬繼續執行。首次尚未選過路徑時，
沿用原本的共同中心線。新一輪拒絕原因仍會更新到候選快照，五條都不標為選中。
下一次評估有可行候選時，自動恢復成本選擇。

此時沿用的舊路徑也可能存在預測碰撞。碰撞檢查只涵蓋離散預測時間點，
其他車實際決策也可能偏離定速、等偏移假設，因此不保證實際模擬完全無碰撞。

調整 `COST_OVERTAKING_CONFIG` 的 `progress_weight`、`overtaking_weight`、
`candidate_offsets`、`prediction_horizon`、`decision_period` 與 `transition_length`
即可改變評分與預測範圍。預測模型使用 `MODEL_CONFIG`，控制器使用各車實際設定。

### 純進度策略

在 `STRATEGY_ASSIGNMENTS` 使用 `StrategyAssignment("progress")`，即可只比較自車的
五條候選軌跡。例如保留後車超車、讓前車只爭取自己的進度：

```python
STRATEGY_ASSIGNMENTS = {
    "vehicle-1": StrategyAssignment("cost_overtake"),
    "vehicle-3": StrategyAssignment("progress"),
}
```

`progress` 使用各車自己的 Pure Pursuit、速度設定與自行車模型，選擇預測終點累積
縱向進度 `delta_s` 最大的候選，圖上成本為 `J = -delta_s`。不找超車對象、不計算
超車獎勵，也不預測其他車或剔除預測碰撞。其他車的位置、速度與已選路徑不影響決策。
候選目標仍受路寬、車寬與 `safety_margin` 限制；這只限制目標位置，不檢查整條軌跡出界。
主迴圈的實際碰撞偵測仍會記錄重疊，不改變車輛運動。

參數集中於 `PROGRESS_CONFIG`：預設候選 `(-0.6, -0.3, 0, 0.3, 0.6)` m、過渡長度
3 m、決策週期 0.1 s、預測時長 1.5 s。預測步長沿用 `MODEL_CONFIG.time_step`。
相同目標沿用既有橫移動作；每台車保有獨立狀態。候選圖只列出 `progress` 成本。
無視窗訓練資料入口也支援此策略，並保存其設定至 metadata。

### 隨時替換策略

在 `STRATEGY_ASSIGNMENTS` 將 `cost_overtake` 換成 `overtake`，即可回到原本的
左右規則式超車；改為 `centerline` 則只追蹤中心線。防守車配置可保持原設定。

新增其他策略時，將實作放在 `strategy/` 的獨立檔案，回傳 `StrategyDecision`，
並在 `main.py` 的 `strategy_factories` 註冊工廠：

```python
strategy_factories = {
    "my_attack": lambda vehicle_id: MyAttackStrategy(...),
}
```

再將某台車指派為 `StrategyAssignment("my_attack")`。工廠每次都應建立新物件，
各車才能保有獨立狀態；`vehicle_id` 可用於選擇該車的明確模型與控制參數。
策略名稱會在建立 `StrategyManager` 時依註冊表檢查，未註冊名稱會在模擬前報錯。
不需要修改模擬迴圈或策略管理器來新增自訂名稱。

## 未來新增自己的策略

`strategy/base.py` 定義共用介面與 `StrategyDecision`：

```python
def update(self, vehicle, others, track, time_seconds) -> StrategyDecision:
    ...
```

輸入只有自己的觀測、其他車的觀測、賽道與目前時間。輸出包含狀態、`target_d`
及可選的 `LateralManeuver`，不包含方向盤或加速度命令。`maneuver=None` 代表
使用共同中心線；要追蹤非零橫向目標，必須同時提供對應的平滑動作。

新增策略時，把演算法放進 `strategy/` 的獨立檔案，透過 `strategy_factories`
註冊名稱與建構方式，再於對照表選用。模擬迴圈使用相同介面，
不需要增加車輛 ID 的特殊判斷。

進攻與防守演算法的輸出格式不同，因此 `assignment.py` 用兩個簡單的包裝類別
轉成共用格式。`sim/strategy_history.py` 負責記錄每台車的狀態
與橫向目標，讓執行結果也能跟著分配表改變。

## 防守策略怎麼判斷？

閱讀順序建議是 `strategy/defending.py` 的 `update()` →
`_candidate_targets()` → `_evaluate_candidate()`。

1. 指定 ego 在後方且位於偵測距離內時，啟用對齊成本；否則該項為零。
2. 從固定候選目標中，排除超出可用路寬的點。
3. 為各目標產生自身平滑橫移路徑，使用自己的目前速度作幾何預測。
4. 啟用碰撞條件時，先剔除與指定防守對象預測重疊的候選，再計算成本。
5. 選可行候選中總成本最低者，輸出 `target_d` 與 `LateralManeuver` 給 Pure Pursuit。

每隔 `decision_period` 重新比較成本，沒有近距離停止橫移、目標鎖定時間
或切換改善門檻。若最低成本目標相同，沿用原動作起點，
讓 Pure Pursuit 繼續追蹤已選的平滑路徑。

### 可開關、可替換的幾何碰撞條件

在 `main.py` 的 `DEFENSE_CONFIG` 調整：

```python
collision_check_enabled=True,
collision_horizon=1.0,
prediction_step=0.1,
```

防守車先將候選路徑轉成世界座標，累加相鄰取樣點的距離，再找出沿路徑走滿
`目前車身速度 × t` 的位置，插值取得對應的 `s`、`d`，車頭方向取路徑線段切線。
路徑長度不足時繼續取樣；跨越起終點時，縱向進度不繞回。
指定對象保持當下的 `d` 與車身速度，用相同方法預測；每輪只預測一次供五條
候選共用，不讀取它的候選或已選路徑。這些都是幾何近似，不執行 Pure Pursuit
或 bicycle rollout，沒有模擬追蹤誤差。

碰撞時間範圍由 `collision_horizon` 決定，取樣間隔不大於 `prediction_step`，
包含當下與終點。使用實際車身的旋轉矩形比較同一時間點，接觸也視為碰撞。
只檢查指派的防守對象，與啟用對齊成本的距離、前後位置與 `q_block` 條件分開。
即使已並排、對方已通過或對齊成本關閉，指定對象仍會接受碰撞檢查。

`strategy/defense_conditions.py` 的 `has_predicted_collision()` 是獨立條件：
回傳 `None` 表示通過，回傳碰撞資訊則剔除候選。修改此函式即可替換判斷方式，
不需要改成本公式或 Pure Pursuit；停用開關時不產生碰撞預測，也不呼叫此條件。

只有可行候選參與成本比較。若全部被剔除，保留上一版目標與完整路徑起點，
顯示 `NO_FEASIBLE_PATH` 並繼續模擬；首次尚未選過路徑時沿用共同中心線。
圖上五條候選都不標為新選中，灰色點線列出碰撞對象與未來秒數，
不顯示被剔除候選的成本。下一輪有可行候選時恢復正常選擇。

`collision_horizon` 與成本的 `prediction_horizon` 分開，調整碰撞時長不會改變
原本的成本取樣或權重。預設 1 秒、間隔 0.1 秒時，五條候選最多比較 55 次車身重疊。

防守成本對應：

```text
J_O = J_race + q_block × sum((d_opponent[k] - d_ego_now)²)

J_race = -progress_weight × estimated_progress
       + smooth_weight × sum((d_opponent[k] - d_opponent[k-1])²)

estimated_progress = s_end - s_start
s_end = 沿候選路徑走滿 max(v, 0) × prediction_horizon 後的賽道位置
```

`k=1..N` 是未來的預測時間點，`d_opponent[0]` 為對手車目前位置。
`N` 是預測步數，不是車輛數量。`d_ego_now` 在一次成本評估中固定，沒有預測
ego 未來意圖。`v` 是當下車身速度，不直接當成 `ds/dt`。幾何路徑依
`path_sample_spacing` 取樣（預設 0.2 公尺），以距離累加與線性插值取得各時間點
的位置。成本、碰撞檢查與候選圖使用同一套位置計算，不再使用 `D² / L` 的平均比例。
靜止時進度為 0；此處仍是理想沿路徑行駛的估計，不包含實際追蹤誤差。

對齊成本關閉時，只有進度獎勵與平順成本參與選擇，不再偏好或強制返回中心線。
平順與對齊項仍為取樣點的加總，進度獎勵只計算一次；改變 `prediction_step`
仍會影響成本的相對尺度，需要重新觀察權重效果。

狀態顯示為 `CRUISE`（跟隨共同中心線）、`DEFEND`（啟用對齊成本）
與 `RETURN`（沿用既有名稱，表示對齊成本關閉但仍在追蹤偏移路徑）；全部候選被剔除時顯示
`NO_FEASIBLE_PATH` 並保留舊路徑。其餘狀態只用於顯示，
不會覆蓋成本選出的目標。選中 `0` 且已接近中心線時，以 `return_tolerance`
判斷是否接回共同中心線。圖上也會顯示 `race` 與 `block` 成本；終端會列出
狀態轉換、橫向目標與碰撞紀錄。

## 預測與限制

`strategy/prediction.py` 共用幾何預測：車輛保持目前速度，依照已選定的橫向
動作前進；未提供動作時保持目前 `d`。防守只取樣自己的候選路徑，對齊位置
固定為 ego 當下的 `d`。ego 評估超車候選時，仍能使用其他車上一個模擬步
留下的路徑做既有碰撞檢查。這是模擬器直接提供的資訊，沒有感測器或通訊模型。

防守的路寬限制只作用於候選目標，不逐點檢查車身是否越界或追蹤誤差。
其可關閉的碰撞條件採用幾何近似，只檢查指定對象，不能保證實際車輛完全不碰撞；
全部候選被剔除時沿用的舊路徑也可能存在碰撞。進攻策略仍保留既有的有限時間碰撞檢查
與超車側邊鎖定行為，僅適用 `overtake`；`cost_overtake` 使用上述 bicycle
前向預測、車身碰撞硬性篩選與兩項成本，沒有側邊鎖定判斷。

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
