# Racing Overtake Simulator

## Project goal

Build a lightweight Python 2D racing simulator for studying
multi-vehicle overtaking and defense algorithms.

The simulator must eventually support an arbitrary number of vehicles
and allow the algorithm core to be reused in ROS2 Humble on a real
F1TENTH-style vehicle.

## Current milestone: Milestone 4 plus a cost-based defense extension

Provide a simple rule-based overtaking strategy for vehicles assigned to attack.
Strategy code must live in `strategy/` and emit high-level lateral targets,
never steering commands. Pure Pursuit continues to provide low-level control.

Assign strategies per vehicle through STRATEGY_ASSIGNMENTS in main.py.
Keep strategy implementations separate in strategy/ and use a shared update
interface with explicit observations, track data and simulation time.
Each assigned vehicle must own independent strategy state, even when multiple
vehicles use the same strategy type. The default scene has one attacker,
one defender and one centerline follower.
Vehicles assigned to defend may choose defensive lateral targets by comparing
a small set of smooth paths with a lateral-alignment cost.
Unassigned vehicles continue independently following the shared centerline.
Disabling defense must preserve the passive-opponent overtaking mode.
Preserve arbitrary N-vehicle support and all Milestone 1-3 behavior.

Do not implement game theory, Nash, Stackelberg,
reinforcement learning, MPC, ROS2, race rules, collision response, dynamic
vehicle physics, sensors, or multi-agent strategic optimization.

Keep tunable demo parameters together near the top of main.py. Use short,
plain comments to explain the defense decisions and cost calculation.

## Technical requirements

- Use Python 3.
- Use type hints.
- Use dataclasses for vehicle state and control commands.
- Use NumPy for numerical calculations.
- Use Matplotlib for visualization.
- Keep functions small and readable.
- Add docstrings where the purpose is not obvious.
- Handle invalid inputs when reasonable.
- Do not add heavy dependencies.

## Architecture

Create this structure:

racing_overtake_sim/
├── AGENTS.md
├── README.md
├── requirements.txt
├── main.py
├── core/
│   ├── __init__.py
│   ├── vehicle_state.py
│   ├── vehicle_model.py
│   ├── track.py
│   └── pure_pursuit.py
├── sim/
│   ├── __init__.py
│   └── visualizer.py
└── tests/
    ├── __init__.py
    └── test_vehicle_model.py

## Architecture rules

- core/ must not import ROS2, Matplotlib, Pygame, or simulator-specific objects.
- sim/ may import Matplotlib.
- main.py should only assemble and run the simulation.
- Do not pass the entire simulator into controllers or planners.
- Algorithms should receive explicit vehicle states and track data.
- Do not write code specifically for exactly two vehicles.
- Preserve the possibility of supporting N vehicles later.

## Vehicle state

VehicleState should contain:

- x in metres
- y in metres
- yaw in radians
- velocity in metres per second

## Vehicle model

Use a rear-axle kinematic bicycle model with:

- steering angle in radians
- acceleration in metres per second squared
- configurable wheelbase
- configurable simulation time step
- steering and velocity limits

## Pure Pursuit

Pure Pursuit should:

- locate the nearest track point
- search forward for a lookahead point
- work across the end of a closed path
- calculate steering angle
- limit steering angle
- regulate toward a configurable target speed

## Definition of done

The task is complete when:

1. `python main.py` runs without errors.
2. One vehicle follows an oval track for at least two laps.
3. A Matplotlib figure shows the centerline and vehicle trajectory.
4. The vehicle remains reasonably close to the centerline.
5. Tests pass using `python -m unittest`.
6. README.md explains installation and execution.

# Workspace Rules

This repository is the only workspace.

Never access files outside this repository.

Never modify files outside this project.

Do not assume the existence of other projects.

All generated files must remain inside this repository.
回答時使用繁體中文

## 研究紀錄與報告交接

### 目的與適用範圍

使用者會讓 ChatGPT 透過 GitHub 讀取本專案，分析研究結果並撰寫給教授看的進度報告。
應留下準確、可追溯的實作與實驗紀錄，讓未看過對話的 AI 也能理解目前進度。
紀錄使用繁體中文，保留必要的英文技術名稱；避免把工程修改誇大成研究成果。

### 紀錄可信度規則（優先於其他紀錄要求）

使用者在開發時不一定說明研究目的。不得為填滿紀錄，自行補出使用者的動機、假設或預期成果，
也不必為填寫紀錄反覆詢問。所有紀錄須區分：

1. **使用者明確說明**：只有使用者在對話或文件中明確表達的研究目的、修改理由與決策，
   才能記為使用者意圖；摘要保留原意，不擴大解讀。
2. **可查證的實作事實**：依程式、差異與執行輸出，記錄修改前後的行為、參數與結果。
   可以說明程式直接造成的作用，但不能據此推定研究動機。
3. **待驗證的分析**：可能原因、研究價值、預期改善與下一步建議，如需保留，
   放在獨立的「Codex 分析／建議，未經使用者確認」欄位，不混入事實紀錄。

使用者未說明研究目的或修改理由時，寫「使用者未說明」或省略該欄位。
不得從函式名稱、修改內容或結果反推出意圖。例如：

- 可記錄：「防守決策週期由 0.3 秒改為 0.2 秒。」
- 可說明：「設定的決策更新頻率因此提高。」
- 不可自行寫：「為改善防守反應遲緩，縮短決策週期。」
- 未實測時不可寫：「修改後防守效果改善。」

「下一步」須區分使用者已決定的工作與 Codex 建議，不得把建議寫成確定的研究計畫。
第一次整理歷史紀錄也適用；既有文件未記錄的動機，不得自行補齊。

### 一、紀錄位置與分工

- `README.md`：維護目前架構、功能與操作方式。
- `CHANGELOG.md`：簡短記錄各日期的功能新增、行為變更、修正與已知問題。
- `reports/research_log.md`：按日期記錄研究目的、方法變更、實驗結果、限制與下一步，
  作為 ChatGPT 撰寫進度報告的主要來源。
- `reports/experiments/<日期_實驗名稱>/`：保存值得保留的實驗設定、數據摘要與圖表。

優先沿用既有文件與輸出，不建立內容重複的文件。大量原始資料、模型權重與暫存檔
沿用現有管理方式；報告所需的小型摘要與圖表應能納入 Git。

### 二、何時更新

完成影響演算法、模擬行為、資料處理、評估方式或研究結論的工作後，更新相關紀錄。
同一天同一主題可合併，保留重要決策與結果。
純排版、註解或不影響行為的小修改，不必寫成研究進展。
不要為補齊紀錄而自行啟動耗時實驗。

### 三、研究紀錄內容

每次有實質研究進展時，依適用情況記錄下列內容；沒有資料時寫「未測試」「未記錄」
或「待確認」，不要硬填。可信度規則優先於欄位完整性。

#### 1. 研究目的

- 本次要解決什麼問題？
- 修改前有什麼現象或限制？
- 希望驗證什麼假設？

#### 2. 方法與修改

- 修改前後的做法，以及修改理由。
- 涉及的程式路徑、主要類別或函式。
- 演算法的輸入、輸出與決策流程。
- 若修改成本函數，記錄公式、符號、權重、單位、正規化方式，以及取最小值或最大值。
- 區分成本懲罰、硬性約束與候選軌跡剔除條件。
- 若涉及對手預測，說明採用的假設與預測範圍。
- 若涉及 GP 等學習模型，說明輸入特徵、預測目標、預測時間、資料切分與比較基準。
- 只詳細說明本次新增或變更的部分，避免反覆重寫整套架構。

#### 3. 實驗設定與重現方式

- 實際執行的命令與必要環境資訊。
- 賽道、車輛數、策略配置、初始條件與關鍵參數。
- 模擬時間步長、策略更新週期、預測時間與候選數量等相關設定。
- 隨機種子、episode 數、資料量及訓練／測試切分方式（若適用）。
- 記錄實際使用的設定，不可以目前預設值推測歷史實驗設定。
- 若重現依賴未上傳的本機資料，明確說明缺少哪些資料。

#### 4. 結果與證據

- 列出實際量測數值、單位、樣本數與比較基準。
- 策略實驗可包含超車成功率、碰撞率、圈速、進度差或計算時間，僅記錄確實量測的項目。
- 模型實驗可包含 MAE、RMSE、基準模型誤差及評估範圍。
- 記錄失敗案例與不利結果，不只記錄成功案例。
- 連結到專案內實際存在的數據摘要、日誌與圖表。
- 清楚分開「量化結果」「目視觀察」與「推測原因」。

#### 5. 驗證與限制

- 記錄實際執行的測試命令、通過／失敗情況與未執行的檢查。
- 引用舊測試紀錄時標示來源與日期，不得描述成本次重新驗證。
- 單元測試通過不代表超車策略或模型效果已獲驗證。
- 區分已實作、已執行、已量化驗證與尚在規劃中的功能。
- 說明結果能支持哪些結論，以及尚不能支持哪些結論。

#### 6. 下一步

- 提出與目前問題直接相關的 1～3 個具體驗證項目。
- 說明各項要驗證的問題與預計比較方式。
- 標示為建議或待辦，不寫成已完成工作；區分使用者已決定工作與未確認的建議。

### 四、版本與證據規則

- 每筆紀錄標示日期、分支，以及可確認的程式版本。
- 結果來自未提交工作區時，標示「基於 <commit>，含未提交修改」，不得捏造最終 commit SHA。
- 歷史資料不足時，不推測確切日期、參數、數值或修改動機。
- 不把後續修改過的程式設定，當成先前實驗實際使用的設定。
- 用專案相對路徑連結證據，避免只有本機絕對路徑。
- 確認預計交接的摘要與圖表未被 `.gitignore` 排除；不要因此將整個資料集或模型資料夾解除忽略。
- 這套規則不授權自動 commit 或 push；依使用者當次指示處理。

### 五、完成工作時的回覆

簡短列出：

1. 本次完成的實質修改。
2. 實際執行的驗證與主要結果。
3. 更新了哪些研究紀錄及證據檔案。
4. 哪些必要結果仍只在本機、尚未納入 Git 或尚未推送，會導致 ChatGPT 無法讀取。

### 首次設定與歷史基線整理

1. 將本章規則附加至根目錄 `AGENTS.md`，保留原有內容。
2. 讀取目前程式、`README.md`、`CHANGELOG.md`、`GP_TRAINING.md` 與相關 Git 紀錄。
3. 建立 `reports/research_log.md`，整理「目前研究狀態基線」。
4. 如有相關實驗輸出，讀取設定與結果，選取必要的小型摘要與圖表放入
   `reports/experiments/`；不搬動或刪除原始資料。
5. 無法確認的歷史結果標示待確認，不為填寫文件重新訓練模型或執行大型實驗。
6. 首次設定僅建立紀錄機制與整理現況，不修改演算法。
