# 研究紀錄

此文件供未參與開發對話的 ChatGPT 與報告撰寫者使用。記錄規則見
[AGENTS.md](../AGENTS.md#研究紀錄與報告交接)；目前架構與操作見
[README.md](../README.md)，簡短變更見 [CHANGELOG.md](../CHANGELOG.md)。
後續依日期新增紀錄；同日同主題可合併。不可把未確認建議寫成使用者意圖或既定計畫。

## 2026-11-11｜上次推送後變更與本次提交檢查

### 使用者明確說明與版本範圍

- 使用者明確要求：「請把距離上次 push 後我新增的東西紀錄並 push GitHub」，
  並於中斷後要求繼續。本次獲授權將既有程式差異、文件與小型證據一起提交及推送。
- 分支：`master`。遠端基準：`d3910ddc11e9ef31c5eceaa70afd71c920507e34`。
  本次透過 GitHub connector 讀取遠端 master，並以 `git fetch origin master` 核對，結果一致；
  本機 HEAD 也在同一版本，沒有另待推送的本機 commit。
- 基準 commit 的提交日期為 2026-11-09，但精確 push 時間未記錄，不能把提交時間當成推送時間。
- 以下為**基於 `d3910ddc11e9ef31c5eceaa70afd71c920507e34`，含未提交修改**的提交前檢查。
  最終提交 SHA 可由本節所在的 Git commit 取得；推送是否完成以遠端分支 ref 為準。
- 各項歷史演算法修改的使用者研究動機／假設：使用者未說明。本節依差異與實際輸出整理，
  不將工程新增描述為已證實的研究改善。

### 上次遠端版本之後新增與修改的內容

詳細公式、目前參數與三組 GP 結果沿用下方「目前研究狀態基線」，本節只彙整推送範圍。

| 主題 | 可查證的差異與主要檔案 |
| --- | --- |
| 相對進度進攻 | 新增 `strategy/relative_overtaking.py` 的 `RelativeOvertakingStrategy`；只最小化預測終點有號間距，保留負成本、跨圈累積、超車後保留原比較對象，並篩選所有其他車的預測碰撞；新增對應單元及整合測試 |
| 每車指派與場景 | `main.py` 預設改為 vehicle-1 `relative_overtake`、vehicle-2 `progress`；新增獨立九候選設定（0.1 s 決策、0.8 s 預測）；成本式進攻示範亦改為九候選、0.8 s 預測、權重 2／10；最大加速度 3→5 m/s² |
| 碰撞尺寸 | `core/vehicle_geometry.py` 新增 `CollisionConfig`，碰撞矩形與實體車身共用中心但可有不同長寬；主程式目前設 0.60×0.40 m，實體車身仍 0.55×0.30 m；預測、實際重疊判斷及相關測試同步支援 |
| 可編輯資料欄位 | 新增 `training_data/fields.py` 的 `InputField`／`OutputField` 清單與選擇；預設 `opponent_s`→同時刻 `opponent_current_d`；保留舊四輸入／未來答案，驗證 current／future 與 horizon 一致，新增 `--list-fields` 及對應測試 |
| 資料生成與相容性 | `generate_training_data.py`、`sim/data_generation.py` 支援相對進度策略、progress 優先配對及設定保存；`training_data/` 支援 horizon=0、最後一幀、target_timing、可選碰撞尺寸匯出，仍可讀舊檔 |
| GP 訓練與評估介面 | `train_gp.py` 支援唯一答案自動選用、current／future 註記、固定訓練平均值基準，以及單輸入同時刻對應圖；更新 `GP_TRAINING.md` 與資料說明；沒有將 GP 接入策略決策 |
| 研究交接 | AGENTS.md 新增長期規則；README 與 CHANGELOG 同步現況；建立本研究紀錄及三組既有 GP 的設定／報告／逐筆預測／圖表封存、來源雜湊與檢查日誌 |

這些程式差異在本次提交整理前已存在。本次未修改演算法；77 個既有程式與相依設定檔的
SHA-256 仍與首次盤點一致。另修正首次整理時三個 JSON 說明欄位的亂碼，
只修正描述文字，沒有改動實驗數值或原樣封存的報告、CSV 與圖表。
新增 `.gitattributes`，讓 `reports/experiments/` 保留原始位元組，避免 Git 換行轉換造成
封存雜湊失配；此設定只管理證據檔案，不改變模擬行為。

### 本次實際驗證與限制

使用專案內既有 Python 3.10.20、NumPy 2.2.6、Matplotlib 3.10.9、scikit-learn 1.7.2，
環境設定與下方首次整理的選定測試相同（PYTHONPATH 指向 `.venv/Lib/site-packages`、
Matplotlib 使用 Agg，快取與測試暫存均位於專案內）。本次沒有修復失效的 `.venv` 啟動器。

實際執行的完整測試 argv：

```powershell
& ./.cache/python/cpython-3.10.20-windows-x86_64-none/python.exe -m unittest -v
```

- 完整 discovery 的原始輸出為 **`Ran 264 tests in 21.436s`、`FAILED (errors=1)`**，exit code 1。
  唯一錯誤為 `tests.test_defending_integration.DefendingIntegrationTests.setUpClass` 的
  `ValueError: path_length must cover the complete transition`；該類別的方法未執行。
  場景使用 6 m 規則式過渡，而 main.py 參考路徑長 4 m；本次未更動此設定。
  原始輸出：[unittest.log](experiments/2026-11-11_push_review/unittest.log)。
- 重新執行既有封存的 `verify_evidence.py`，14 個來源封存檔 SHA-256 與三組共 1352 筆預測的
  誤差核對通過，exit code 0。輸出：[evidence_verification.log](experiments/2026-11-11_push_review/evidence_verification.log)。
- 完整環境、命令、遠端基準、exit code 與驗證範圍：
  [validation.json](experiments/2026-11-11_push_review/validation.json)。
- 測試含測試專用短場景及微型 GP 擬合，不是對既有研究模型重新訓練；
  未重新執行大型實驗，也未新增多 seed 超車成功率、碰撞率或圈速量測。
- 單元測試及既有預測誤差核對，不能證明策略效果、跨場景泛化或無碰撞；
  三組 GP 的不同答案時間與場景仍不可當成單一受控改善比較。

### 交接內容與下一步

本次納入程式、測試、操作文件、研究紀錄與小型證據，讓 GitHub 上的讀取者能一併查看。
`data/training/` 的完整 samples／raw_states、`models/gp/` 的 model.pkl，以及環境和快取
仍維持原有忽略規則；只有 GitHub 的環境仍無法重新擬合歷史模型或重新播放完整原始軌跡。

使用者已決定的工作是完成本次紀錄、commit 與 push；未另指定後續演算法修改。
首次基線中列出的 1～3 項驗證建議仍屬「Codex 分析／建議，未經使用者確認」。

## 2026-11-11｜目前研究狀態基線

本節保留首次整理當時的狀態；「未執行完整測試」「尚未提交」等敘述均指首次整理階段。
後續完整測試與提交授權見上方「上次推送後變更與本次提交檢查」。

### 版本、範圍與資料可信度

- 整理日期：2026-11-11，Asia/Taipei。分支：`master`。
- 盤點版本：**基於 `d3910ddc11e9ef31c5eceaa70afd71c920507e34`，含未提交修改**。
  本次開始前已有 20 個 tracked 檔案修改及 5 個 untracked 程式／測試檔；不是本次實作。
  初始狀態、8 筆 commit 紀錄、程式 SHA-256 與封存檔來源見
  [source_manifest.json](experiments/2026-11-11_baseline/source_manifest.json)。
- 本次僅附加紀錄規則、整理基線、同步 README 的現況描述及封存小型證據；沒有修改演算法。
- 歷史實驗的分支、commit、實際命令未保存，均為**未記錄／待確認**。
  不能將本次基底 commit 或目前參數套用到歷史結果。產物 ID 含 UTC 時間標記，
  但沒有獨立保存的起訖時間，不據此補出精確執行時間。
- 本機 `origin/master` tracking ref 指向上述 commit；本次未 fetch，未查核 GitHub 即時狀態。
  沒有自動 commit 或 push。

### 使用者明確說明

- 本專案目標是輕量 Python 2D 多車超車／防守模擬，保留日後在 ROS2 Humble、
  F1TENTH 類車輛重用演算法核心的可能性。來源：既有 [AGENTS.md](../AGENTS.md)。
  這不表示已實作 ROS2。
- 本次整理目的：使用者會讓 ChatGPT 透過 GitHub 分析研究結果，撰寫給教授看的進度報告；
  要求紀錄準確、可追溯，且本次不修改演算法。來源：2026-11-11 使用者本次指示，
  已納入 AGENTS.md「研究紀錄與報告交接」。
- 歷史各項策略、GP 特徵與參數修改的研究動機或待驗證假設：**使用者未說明**。
  下文只記錄程式與既有輸出能確認的內容。

### 可查證的實作事實與完成程度

| 項目 | 已實作的內容與來源 | 已有證據及界限 |
| --- | --- | --- |
| 車輛／控制／賽道 | `core/vehicle_state.py` dataclasses、`core/vehicle_model.py` 後軸運動學 bicycle model、`core/pure_pursuit.py`，`core/track_factory.py` 可選賽道 | 有測試與歷史 commit；本次未重新量測圈數或中心線誤差 |
| 多車與策略分配 | `strategy/assignment.py` 的 `StrategyManager`、`main.py` 的 `STRATEGY_ASSIGNMENTS`；每車獨立策略物件、同步觀測 | 已實作 N 車介面；本次沒有新做 N 車效能或碰撞率評估 |
| 高階策略 | `centerline`、`overtake`、`cost_overtake`、`progress`、`relative_overtake`、`defend`，輸出橫向目標／平滑路徑，低階仍為 Pure Pursuit | 本次選定測試包含相對進度策略單元測試；不代表超車成功率已驗證 |
| 預測與碰撞 | `strategy/bicycle_prediction.py`、`strategy/prediction.py`、`strategy/collision_check.py`；旋轉矩形重疊只回報，不改變運動 | 有離散預測篩選；沒有 collision response 或無碰撞保證 |
| 資料生成 | `generate_training_data.py`、`sim/data_generation.py`、`training_data/`；保存實際狀態、metadata 及可選特徵／答案 | 三組本機資料與模型評估已存在，本次讀取並核對，未重新生成這三組資料 |
| GP 訓練 | `train_gp.py` 以 episode 分組切分、標準化、單一答案 GP，輸出報告／預測／圖表 | 三組既有測試誤差見下文；GP 尚未接入策略決策 |
| GP 示範 | `gp_demo.py` 使用人工函數與 40 筆合成資料 | 是獨立示範，不能當作真實策略反應資料的評估；本次未執行 |
| 排除範圍 | 既有規範排除博弈、Nash／Stackelberg、RL、MPC、ROS2 實作、感測器與動態車輛物理 | 不列為已完成研究成果；本次沒有新增這些功能 |

主要入口：[main.py](../main.py)、[策略介面](../strategy/base.py)、
[策略分配](../strategy/assignment.py)、[資料說明](../training_data/README.md)、
[GP 操作](../GP_TRAINING.md)。完整操作沿用既有文件，不在此重寫。

### 歷史版本索引（依 Git 可確認的事實）

日期為 Git commit 的 +08:00 日期，表示提交時間，不等同實驗執行日。
歷史分支名稱未記錄；下列 commit 均可由本次 `master` 的歷史讀取。
來源為 source_manifest.json 的完整 SHA 與時間；可用 `git show <commit> --stat` 核對。

| 提交日期 | Commit | 提交差異可確認的功能 |
| --- | --- | --- |
| 2026-07-21 | `9a44729` | 單車 Pure Pursuit、運動學模型、橢圓賽道與視覺化 |
| 2026-08-31 | `242ef4f` | 多車模擬 |
| 2026-08-31 | `3412668` | 賽道 s/d、車身幾何與碰撞回報 |
| 2026-08-31 | `63aa9a4` | Milestone 4 規則式超車 |
| 2026-11-02 | `a5736bf` | 每車策略分配與成本式防守 |
| 2026-11-03 | `3a85f8f` | 成本式超車、候選預測碰撞篩選與候選視覺化 |
| 2026-11-04 | `b644d11` | 沿幾何路徑距離預測、資料收集與 GP 合成示範 |
| 2026-11-09 | `d3910dd` | GP 訓練入口、圓角矩形賽道、progress 策略 |

### 本次盤點的工作區差異（不是本次新實作）

來源：本次開始時 `git diff`、目前程式，以及 source_manifest.json 的檔案雜湊。
修改日期與使用者修改理由未確認；不將盤點日寫成實際修改日。

| 範圍 | 基底 commit → 目前工作區的差異 |
| --- | --- |
| 示範車輛／指派 | vehicle-1 `cost_overtake`、vehicle-3 `progress` → vehicle-1 `relative_overtake`、vehicle-2 `progress` |
| 相對進度策略 | 新增 `strategy/relative_overtaking.py` 及對應單元／整合測試、主程式與資料生成註冊 |
| 碰撞幾何 | 新增 `CollisionConfig`；實體車身仍為 0.55 × 0.30 m，主程式碰撞矩形另設 0.60 × 0.40 m；原始狀態匯出可記錄碰撞尺寸 |
| 速度控制 | `main.py` 最大加速度由 3.0 改為 5.0 m/s² |
| cost_overtake 示範設定 | 進度權重 1→2、超車獎勵權重 20→10、五候選→九候選、預測 1.5→0.8 s；決策週期仍 0.1 s |
| 資料欄位 | 新增 `training_data/fields.py` 與欄位註冊；CLI 預設改為 `opponent_s` → `opponent_current_d`，答案為同時刻、horizon=0；保留舊未來答案模式 |
| GP 報告 | 支援唯一答案自動選取、答案時間、訓練平均值基準及單輸入同時刻對應圖 |

### 目前主程式設定（靜態盤點，非歷史實驗設定）

以下取自本次開始時的 [main.py](../main.py)，**不代表本次重新執行了此 21 s 場景**。

| 設定 | 目前值 |
| --- | --- |
| 賽道 | `rounded_rectangle`，中心線跨度 20 × 12 m、半徑 3 m、500 點、寬 2 m、封閉；仍可選 `oval` |
| 場景 | 2 車，模擬 21 s；vehicle-1：s=0 m、目標 11 m/s、relative_overtake；vehicle-2：s=4 m、目標 9 m/s、progress |
| 初始狀態 | 兩車 d=0 m、速度 0 m/s；yaw 沿中心線切線，由 `_create_vehicle()` 建立 |
| 車模 | dt=0.02 s、軸距 0.33 m、轉角上限 50°、速度範圍 0～10 m/s；11 m/s 目標仍受上限限制 |
| 控制器／幾何 | 前視 2 m、速度增益 1.5、最大加速度 5 m/s²；車身中心距後軸 0.12 m；實體 0.55 × 0.30 m、碰撞 0.60 × 0.40 m |
| relative_overtake | 九目標 -0.6～+0.6 m、間距 0.15 m；決策 0.1 s、預測 0.8 s、過渡 3 m、偵測 6 m、邊界留距 0.1 m |
| progress | 五目標 (-0.6,-0.3,0,0.3,0.6) m；決策 0.1 s、預測 1.5 s、過渡 3 m、邊界留距 0.1 m |
| cost_overtake | 目前未指派；九目標、決策 0.1 s、預測 0.8 s、過渡 3 m、偵測 6 m，權重 progress=2、overtaking=10 |
| defend | 目前未指派；q_block=15、progress_weight=2、smooth_weight=1，五目標、決策 0.2 s、成本預測 1 s、取樣 0.1 s、過渡 2 m；幾何間距 0.2 m；碰撞篩選關閉（時長設定仍為 1.5 s） |
| 路徑／防守開關 | 參考路徑長 4 m、間距 0.2 m；`DEFENSE_ENABLED=True`，但當前無 defend 車；`--no-defense` 只替換 defend 指派 |

### 成本、限制與對手預測的實作摘要

這是基線索引；之後只記錄新增或變更部分。輸入為明確的自身／他車觀測、track 與 time_seconds；
輸出為 `StrategyDecision`（橫向目標及可選的 `LateralManeuver`），不是轉向命令。

| 策略／來源 | 目標函數與選擇方式 | 預測、硬性條件與剔除 |
| --- | --- | --- |
| [relative_overtake](../strategy/relative_overtaking.py) | 最小化 `J = gap_now + Δs_other - Δs_self`；全部為 m，不取絕對值，負值保留；無其他權重或正規化 | 自車 Pure Pursuit+bicycle rollout；所有他車定當下速度、追蹤當下 d 的等偏移路徑，時長 0.8 s；先剔除與任一他車預測碰撞的候選 |
| [progress](../strategy/progress.py) | 最大化自車 `Δs_self`（m），圖上 `J=-Δs_self`；無權重或正規化 | 自車 rollout 1.5 s；不預測他車，不作預測碰撞篩選 |
| [cost_overtake](../strategy/cost_overtaking.py) | 最小化 `J=-w_p Δs_self-w_o I_pass`；`I_pass∈{0,1}` 表示預測終點縱向超過前車；目前 w_p=2、w_o=10 | 自車與所有他車 rollout 0.8 s；他車定速等偏移；先檢查所有他車碰撞，超車獎勵對象才受 6 m 偵測限制 |
| [defend](../strategy/defending.py) | 最小化 `J=-w_p Δs + w_s Σ(d[k]-d[k-1])² + q Σ(d[k]-d_ego_now)²`，對齊項 k=1..N；目前權重 2、1、15 | 以當下車身速度沿自身候選幾何路徑行進，非 bicycle rollout；指定對象在後方 -6≤gap<-1 m 且 q>0 才啟用對齊項；d_ego_now 固定。可選碰撞條件僅檢查指定對象，與對齊觸發分開 |

`Δs` 是累積賽道縱向進度（m），`d` 是中心線左正右負偏移（m）。defend 平順／對齊項為 m²
取樣加總，沒有乘 dt 或依樣本數平均；cost_overtake 的獎勵為無單位指標。
這兩種加權成本在程式中沒有明定權重物理單位或項目正規化，不能將總成本逕稱為某個物理單位。
改變防守取樣間距會改變加總項尺度。

共用的候選目標路寬條件為 `|target_d| ≤ width/2 - body_width/2 - safety_margin`。
這只限制目標，不代表整條軌跡或車身永不出界。碰撞是**候選剔除條件**，不是成本懲罰；
離散時間的旋轉碰撞矩形接觸也算重疊，包含當下與預測終點。
relative_overtake／cost_overtake 全部候選被剔除時沿用舊目標與路徑，並無安全停止方案。
防守僅在開啟碰撞條件時作對應檢查，目前主程式該條件為關閉。

relative_overtake 優先選最近前車；無新前車時保留舊對象，超車後仍以累積有號間距比較。
對象消失且無新前車時停用相對成本、仍篩碰撞；同分優先保留原可行目標。
由公式可直接確認：同一輪對手預測對所有候選相同，故有對象時，在相同可行集合內，
相對成本排序與最大化自車進度一致；這不是經實驗量測的策略改善結論。
上述預測均未使用訓練好的 GP，也沒有讀取對手將來的策略決策。

### 既有 GP 實驗設定與量化結果

本次只讀取既有產物、重算已保存預測的誤差，未重訓這些模型。
封存日期皆為本次盤點日；歷史執行分支、程式版本及原始命令未記錄。
完整原始設定、每個 episode 的實際 s/d、初始速度／目標速度與指派都保留在各組 metadata.json。
不使用目前 main.py 回填這些歷史欄位。

| 封存組別 | 原資料 ID → 原模型 ID | 任務／實際取樣 | 資料與場景 |
| --- | --- | --- | --- |
| [gp_future_0p02s](experiments/2026-11-11_baseline/gp_future_0p02s/metadata.json) | `20261105T065547Z_270c5e64` → `20261109T084348Z_1b728413` | 四輸入 relative_s、relative_d、relative_speed、opponent_d；答案 opponent_delta_d，0.02 s 後；dt=0.02 s，horizon=1，stride=1 | 5×4 s；1000 筆；兩車 cost_overtake／defend；500 點封閉中心線跨度 20×10 m，長約 48.4419 m，路寬 2 m（型別名稱未記錄） |
| [gp_future_1s](experiments/2026-11-11_baseline/gp_future_1s/metadata.json) | `20261109T090105Z_79f9bb58` → `20261109T090300Z_5d40b9b6` | 同四輸入／同位移答案；1 s 後，dt=0.02 s，horizon=50，stride=1 | 5×20 s；4755 筆；兩車 cost_overtake／defend；同上中心線 |
| [gp_current_sd](experiments/2026-11-11_baseline/gp_current_sd/metadata.json) | `20261111T063545Z_f6eb6a97` → `20261111T064250Z_dbc5bc95` | opponent_s→opponent_current_d；同時刻，非未來位移；dt=0.02 s，horizon=0，stride=5（每 0.1 s） | 5×20 s；1005 筆；兩車 cost_overtake／progress；500 點封閉中心線跨度 20×12 m，長約 58.8483 m，路寬 2 m（型別名稱未記錄） |

三組資料生成 seed 均為 42；GP 切分 seed=42、test_fraction=0.2；皆以 episode 分組，
訓練 episode 為 00000、00002、00003、00004，測試只有 00001。
每組的 00001 是各自的模擬，不應視為三組條件相同的比較實驗。
三組 metadata 都設定排除相關碰撞；前兩組因未來長度不足分別排除 5、250 筆，第三組無排除。

隨機範圍均記錄為：縱向 jitter=1.5 m、橫向 limit=0.5 m、目標速度 jitter=0.5 m/s、
起始速度 deficit=2 m/s、邊界 margin=0.15 m、最多嘗試 100 次；完整實際初始條件以 scenarios 為準。
三組車模步長 0.02 s、軸距 0.33 m、速度上限 10 m/s；前視距離 2 m。
前兩組加速度上限 3 m/s²、防守過渡 4 m、決策 0.3 s；第三組是 5 m/s²、2 m、0.2 s。
三組的 cost_overtake 都是五候選、權重 1／20、決策 0.1 s、預測 1.5 s；
defend 的碰撞條件均關閉；第三組 progress 為五候選、決策 0.1 s、預測 1.5 s。
三組都未記錄後來新增的碰撞尺寸覆寫，不能當成目前 0.60 × 0.40 m 場景。

GP 採 `ConstantKernel × RBF + WhiteKernel`，初始 length_scale=1、noise_level=0.02、
optimizer_restarts=2；輸入只用訓練集標準化，答案 `normalize_y=True`，每個輸出單位均為 m。
報告記載 Python 3.10.10、NumPy 2.2.6、scikit-learn 1.7.2；這是歷史環境，不是本次檢查環境。
目前訓練流程的說明見 [GP_TRAINING.md](../GP_TRAINING.md)；歷史 kernel、參數及警告以原報告為準。

下表為原報告數值（本次從 predictions.csv 重算 GP／零值基準，與報告一致；第三組訓練平均值
另用本機訓練資料計算核對）。MAE／RMSE 單位均為 m：

| 組別／原報告 | 訓練／測試樣本 | GP MAE／RMSE | 固定零值 MAE／RMSE | 固定訓練平均值 MAE／RMSE |
| --- | --- | --- | --- | --- |
| [0.02 s](experiments/2026-11-11_baseline/gp_future_0p02s/report.json) | 800／200 | 0.007320／0.010424 | 0.008498／0.011523 | 原報告未記錄 |
| [1 s](experiments/2026-11-11_baseline/gp_future_1s/report.json) | 3804／951 | 0.120623／0.179379 | 0.108631／0.160076 | 原報告未記錄 |
| [同時刻 s→d](experiments/2026-11-11_baseline/gp_current_sd/report.json) | 804／201 | 0.039131／0.117732 | 0.568663／0.579660 | 0.101691／0.165749 |

**不利結果與失敗案例：**1 s 預測的 GP MAE 和 RMSE 都比零值基準差；原報告亦有
signal variance 達上界、第一個 length scale 達下界的警告。0.02 s 與同時刻 s→d 在各自的
單一測試 episode 內優於表列基準，但不足以宣稱跨場景或跨賽道泛化。
1 s 組最大絕對誤差約 0.904936 m（episode-00001，t=0.02 s）；同時刻組仍有約
0.833086 m 誤差（episode-00001，t=0 s，actual=-0.436183 m、predicted=+0.396903 m）。
逐筆列與最大誤差索引見 [summary.json](experiments/2026-11-11_baseline/summary.json) 及各組 predictions.csv。

**目視觀察：**本次檢視封存圖，0.02 s 組有大量預測接近固定小正值，1 s 組也有大量預測
接近零、偏離對角線；同時刻圖的大部分點靠近對角線，仍有明顯離群點。這些是圖面描述，
不替代量化評估或誤差成因分析。原圖連結：
[0.02 s 圖](experiments/2026-11-11_baseline/gp_future_0p02s/evaluation.png)、
[1 s 圖](experiments/2026-11-11_baseline/gp_future_1s/evaluation.png)、
[同時刻 s→d 圖](experiments/2026-11-11_baseline/gp_current_sd/evaluation.png)。

本次讀取三組本機 raw_states.csv，記錄碰撞的同步 frame 數均為 0；分別有 2010、10010、10010 列
車輛狀態。這只描述這三組舊場景的紀錄，不能推論目前策略無碰撞或所有情況碰撞率為零。

### 其他既有輸出與歷史測試紀錄

- [CHANGELOG.md](../CHANGELOG.md) 的既有 2026-11-09、11-10、11-11 段落分別記載完整測試
  225、239、264 項，均提及防守整合初始化錯誤。這些是**舊文件敘述**，未找到相應日期的完整
  原始測試日誌；本次沒有重新執行這些完整測試數量。
- 既有 2026-11-11 段落記載 21 s relative_overtake／progress 場景最後成本約 -15.91 m 且有碰撞。
  未找到能確定對應該次執行的完整設定與逐步輸出，精確數值及可重現性**待確認**；不作已量化的
  超車成功率或碰撞率。不得用本次封存的 cost_overtake GP 資料代替該場景證據。
- 既有 2026-11-10 段落記載 3×2 s、63 筆、42／21 筆切分的流程驗證。本機 `.cache/progress-line-*`
  有相符數量的產物；原命令、分支、commit 與執行日未完整保存，僅作舊流程紀錄，
  不與本次保留的 1005 筆同時刻資料混用。重複的 `-verified` 產物與 smoke 副本未重複封存。
- 本機 `.cache/migration/unittest.log` 有 `Ran 106 tests ... OK`，但未記錄可確認的日期與版本；
  不能用它取代目前測試結果。其他 `.cache/*preview.png` 也缺少完整設定／版本，未當成當前場景圖。
- 封存兩個小型未定日期診斷檔，以免現有結果遺失：
  [collision_benchmark.json](experiments/2026-11-11_baseline/undated_diagnostics/collision_benchmark.json)
  與 [stutter_profile.json](experiments/2026-11-11_baseline/undated_diagnostics/stutter_profile.json)。
  前者記錄 4000 組矩形／圓形比較，圓形相對矩形有 445 個 false positives、0 個 false negatives；
  後者記錄固定設定下不繪圖／逐步 Agg 繪圖的耗時。兩者的分支、commit、硬體、完整場景、
  執行命令與日期未記錄，僅保留診斷證據，不稱為已採用的優化或目前版本效能。

### 本次實際驗證與未執行項目

1. 讀取目前程式、README、CHANGELOG、GP_TRAINING、資料說明、8 筆 Git commit、工作區差異及
   上述既有產物。盤點未使用網路，沒有重訓既有模型或執行大型模擬。
2. 以 Python 標準函式庫讀取 CSV／JSON，核對三組樣本數、測試／訓練 episode 不重疊、
   預測 actual 與原 samples 答案一致，並重新計算誤差；封存 14 個來源檔案及其 SHA-256。
   來源資料 CSV 的 SHA-256 保留於 summary.json，原檔未搬動或刪除。
3. 本機 `.venv/Scripts/python.exe` 啟動失敗，顯示原 Python 3.10 安裝不存在。本次改用
   `.cache/python/cpython-3.10.20-windows-x86_64-none/python.exe`，搭配現有
   `.venv/Lib/site-packages`；實際版本為 Python 3.10.20、NumPy 2.2.6、Matplotlib 3.10.9、
   scikit-learn 1.7.2，未安裝或重建環境。
4. 執行下列選定測試，原始輸出為 **`Ran 62 tests in 3.316s`、`FAILED (errors=1)`**。
   62 個實際執行的方法列為 ok，另有 `DefendingIntegrationTests.setUpClass` 失敗，該類別的方法
   未執行。錯誤為 `ValueError: path_length must cover the complete transition`，觸發處為
   `strategy/offset_path.py`；該整合場景選用 6 m 規則式過渡，但 main.py 參考長度為 4 m。
   這是本次重新確認的既有錯誤，本次未修正。日誌：
   [unit_tests.log](experiments/2026-11-11_baseline/unit_tests.log)，環境、argv 與 exit code：
   [validation.json](experiments/2026-11-11_baseline/validation.json)。

實際選定測試命令（PowerShell，於專案根目錄）：

```powershell
$env:PYTHONPATH = "$PWD/.venv/Lib/site-packages"
$env:PYTHONUTF8 = "1"
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = "$PWD/.cache/matplotlib"
$env:TMP = "$PWD/.cache/research_handoff"
$env:TEMP = "$PWD/.cache/research_handoff"
& ./.cache/python/cpython-3.10.20-windows-x86_64-none/python.exe -m unittest -v tests.test_vehicle_geometry tests.test_collision_detection tests.test_training_data tests.test_field_registration tests.test_relative_overtaking tests.test_defending_integration
```

測試由本次暫存檢查腳本透過 subprocess 執行相同 argv 與環境，完整輸出寫入上述日誌。
欄位介面測試內含測試專用資料收集及微型 GP 擬合，產物在測試暫存目錄；
沒有使用或覆寫既有 data/training／models/gp 的資料與模型，不能把它當成研究模型重現。

封存證據可在只有 Git 檔案的環境核對，不需要 scikit-learn、原始資料或模型：

```powershell
python reports/experiments/2026-11-11_baseline/verify_evidence.py
```

本次使用專案內 Python 3.10.20 加上 `-X utf8 -S` 實際執行該腳本；
結果為 14 個封存檔雜湊、3 組共 1352 筆預測與報告誤差相符。
並核對 Markdown 相對檔案連結、`git diff --check`、報告檔案不受 `.gitignore` 排除，
以及本次前後既有程式 SHA-256 一致。檢查輸出見
[handoff_checks.json](experiments/2026-11-11_baseline/handoff_checks.json)。

**未執行：**完整 `python -m unittest`、main.py 的 GUI／即時模擬、兩圈與追蹤誤差量測、
既有 GP 重訓、超車成功率／碰撞率／圈速／多 seed 統計。單元測試與誤差重算不能替代這些驗證。
本次沒有量化驗證 GP 導入策略後的效果，因為目前尚未接入。

### 重現與 GitHub 交接限制

- 各封存組別包含原樣複製的 `report.json`、`metadata.json`、小型 `predictions.csv` 與
  `evaluation.png`。來源路徑、大小與 SHA-256 見 source_manifest.json；Windows 路徑字串
  在原報告中保留原樣，實際文件連結使用相對路徑。
- Git 版證據足以重新計算已保存測試預測的 MAE／RMSE與比較基準、讀取設定並檢視圖表。
  它不足以重新擬合歷史模型、重新產生完整狀態軌跡或驗證舊環境的模型推論。
- 完整重現還缺以下未納入 Git 的本機檔案：`data/training/` 下上述三個資料 ID 的
  `samples.csv`、`raw_states.csv`，以及 `models/gp/` 下上述三個模型 ID 的 `model.pkl`。
  它們維持既有忽略規則，沒有解除整個資料／模型目錄的忽略。
- 原始歷史命令未記錄。`GP_TRAINING.md` 中命令是操作範例；不能聲稱它們就是當時執行命令。
  即使補回原始資料，仍需確認當時程式版本；目前預設車輛、碰撞尺寸與策略已不同。
- 本次新增報告與證據目前為 untracked，AGENTS／README／CHANGELOG 為工作區修改；
  既有欄位註冊與相對進度策略等程式也尚未提交。**尚未 commit／push 的內容無法由 GitHub 讀取。**
  後續依使用者指示納入版本；本次不自動提交。

### 下一步：使用者已決定的工作

本次明確決定的紀錄機制與首次基線整理已完成；使用者沒有指定下一個演算法實作或實驗計畫。
因此不把下列建議記為已排定工作。

### Codex 分析／建議，未經使用者確認

以下尚未執行，亦不是使用者已確認的研究動機或計畫：

1. **測試設定一致性：**確認規則式策略 6 m 過渡與 4 m 參考長度要如何配置；若另行授權修改，
   用相同固定場景比較修改前後防守整合初始化結果，再執行完整測試。先解決目前不能完整驗證的問題。
2. **GP 同任務評估：**若要判斷模型是否具有穩定效益，固定同一答案時間、賽道、資料切分及策略，
   比較 GP、零值與訓練平均值，另留未用於選參的 episode；單獨列出初始暫態與其餘時段誤差。
   現有 s→d 大誤差是否與初始暫態／同一 s 對應多個 d 有關，仍是待驗證分析。
3. **當前相對進度場景：**若要評估目前 relative_overtake，保存當次實際設定與逐步輸出，
   在相同候選與碰撞條件下和 progress／cost_overtake 比較；先定義超車成功與碰撞統計單位，
   再量測結果。現有 GP 離線誤差不能支持這項策略比較結論。
