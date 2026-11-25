# 變更紀錄

## 2026-11-25

### 進度權重 progress_weight（類別預設 1，既有策略行為不變）

- `RelativeOvertakingConfig.progress_weight`：乘在終點間距 gap_T（＝前車 s − 自車 s）上，越大越重視自車進度。
  - 候選成本項 relative_gap 顯示加權後的值。
- `main.py` ④ `RELATIVE_OVERTAKING_CONFIG` 新增 `progress_weight`（先為 1.0，之後改為 2.0）；paper_rollout_overtake 沿用同一值。
  - 改為 2.0 後，relative_overtake、interaction_overtake、rollout_overtake 被指派時的行為也會改變。
  - 2.0 與換目標成本 0.3 的單場結果見 [換目標成本](reports/experiments/2026-11-25_paper_switch_cost/README.md)。
- 新增 1 項測試；全套 472 項，2 項失敗與基線同名。
- 11 個等價場景相同（vehicle-1 指派在記憶體中改回後比對；[紀錄](reports/experiments/2026-11-25_progress_weight/equivalence.log)）。

### 換目標成本 switch_weight（類別預設 0，既有策略行為不變）

- `RelativeOvertakingConfig.switch_weight`（m）：候選的目標 d 與目前決策不同時，成本加上固定值；說明與候選成本項會顯示 switch。
- `main.py` 只在 `PAPER_ROLLOUT_OVERTAKING_CONFIG` 設定（先設 0.5，之後改為 0.3），relative_overtake 維持 0。0.3 與 0.5 的差別見實驗紀錄。
- 新增 1 項測試；全套 471 項，2 項失敗與基線同名；11 個等價場景相同（vehicle-1 指派在記憶體中改回後比對）。
- 單場示範見 [換目標成本](reports/experiments/2026-11-25_paper_switch_cost/README.md)：
  - k=2 換目標 184→8 次、碰撞 122→0 步，前車 d 的 ±2σ 涵蓋（+1 s）57→79%。
  - k=2 與 k=0 都停在前車後方約 1 m，不再超車。

### 新增進攻策略 paper_rollout_overtake：論文方式接入超車決策（既有策略行為不變）

- `strategy/paper_rollout_overtaking.py`：
  - 沿上一步選中候選的軌跡，25 個樣本每 0.1 s 查論文格式一步 GP，並從 N(μ, σ²) 抽樣推演前車。
  - 樣本平均當作前車預測；避撞責任的 r_col 沿賽道、橫向各放大 kσ，成為橢圓（k=2）。
  - 候選與成本同 relative_overtake。
  - 第 0 步不可用時退回定速等 d；之後超出範圍的樣本改為等速。
- `strategy/paper_gp.py`：
  - 載入 4 個輸出的模型（pkl 或 npz），核對欄位、時域、歷史段與 ζ，並由訓練輸入取得適用範圍。
  - 新的精簡格式 `paper_gp_compact_v1`。
- 共用程式的擴充，原呼叫方式都不變：
  - `strategy/interaction_features.py` 新增 `paper_feature_rows`，一次算 Q 列。
  - `strategy/responsibility.py` 的 `PairPrediction` 新增可選的 `PositionUncertainty`，有值時圓改成橢圓。
  - `strategy/relative_overtaking.py` 新增 `_pair_uncertainty` hook，預設回傳 None。
  - `strategy/interaction_gp.py`：`predict_profile_trajectory` 新增可選參數 `progress`；精簡檔的讀寫抽成 `pipeline_arrays`、`rebuild_pipeline`。
- `interaction_gp.py export` 依 `model_kind` 匯出論文格式模型。新增 `models/compact/paper_gp_v1.npz`（1.9 MB，4 個輸出的預測差 0），並登記到登記表。
- 組裝：
  - 工廠與 `SimulationSettings` 註冊 `paper_rollout_overtake`。
  - `main.py` 新增 ⑤-4；有車指派這個策略時才載入模型。
  - 場景與預設指派未改。
- 新增 17 項測試；全套 470 項，2 項失敗與基線同名。11 個等價場景全部相同。
- 單場示範見 [paper_rollout_overtake 示範](reports/experiments/2026-11-25_paper_rollout_overtake/README.md)：
  - k=2 未超車，換目標 184 次，碰撞重疊 122 步。
  - 只用平均（k=0）在 7.54 s 超車，與 relative_overtake 相近。
  - 每次決策平均 0.79 s。

### 論文格式一步 GP 第一版（只有離線，既有策略行為不變）

- `strategy/interaction_features.py`：
  - 新增 `PAPER_FEATURES`（論文 z_k 10 維＋v2 歷史 4 欄）與 `PAPER_TARGETS`（前車 Δs、Δd、Δe_ψ、Δv）。
  - 新增 `paper_state_features`、`front_state_delta`，訓練與之後的線上推演共用。
- `training_data/interaction.py`：
  - 新增 `build_paper_dataset`，標籤時域依 `SamplingConfig`，第一版使用 0.1 s。
  - 窗內缺車與碰撞的檢查抽成共用函式，既有 basic／history 結果不變。
- 新增 4 項測試。
- 新增實驗 [論文格式 GP 第一版](reports/experiments/2026-11-25_paper_format_gp/README.md)：
  - 由三份既有原始紀錄重建資料，4 個獨立 GP。
  - 近距離 0≤gap≤2 m 的 Δd MAE 0.0050 m（等速 0.0250）。
  - Δv 比「車速不變」差；gap<−2 m 在訓練範圍外。
- 尚未接入策略。模型只在本機，未匯出精簡檔。

### 新增進攻策略 rollout_overtake（既有策略行為不變）

- `strategy/rollout_overtaking.py`：取上一步選中候選的預測軌跡，沿它每 0.1 s 用互動 GP 推演前車軌跡，所有候選依這條軌跡選擇。
  - 候選、成本、避撞責任與 ±2σ 邊界檢查沿用 interaction_overtake。
  - 第一節點不可用時退回定速等 d；後段超出範圍時維持。
- 工廠與 `SimulationSettings` 註冊 `rollout_overtake`。
- `main.py` 新增 ⑤-3 `ROLLOUT_OVERTAKING_CONFIG`；有車指派 interaction_overtake 或 rollout_overtake 時才載入 GP。場景與指派未改。
- 新增 9 項測試；全套 449 項，2 項失敗與基線同名。
- 單場示範見 [rollout_overtake 示範](reports/experiments/2026-11-25_rollout_overtake/README.md)：未超車、換目標 146 次、無碰撞。

## 2026-11-24

### 攻方近距離資料收集與 GP v2（使用者已將 main 切換為 v2）

- `interaction_gp.py collect` 新增 `--rear-strategy relative_overtake`：
  - 攻方實際超車收集，攻方候選同 interaction_overtake。
  - `--deviation-weights` 每場輪流，`--speed-advantage`、`--initial-gap` 設定範圍，兩車移動中起步。
  - 預設 probe_follow 收集不變（回歸逐筆相同）。
- 新增 v2 精簡檔 `models/compact/interaction_gp_v2_close.npz` 與登記表 v2 條目；新增 1 項測試，全套 440 項 2 項失敗（同名）。
- 評估見 [近距離資料與 v2](reports/experiments/2026-11-24_close_range_gp/README.md)：
  - 離線近距離誤差大幅下降。
  - 閉迴路 v2 使攻方來回切換、超車變少。
- 使用者之後自行把 main 的 `INTERACTION_GP_MODEL_PATH` 改為 v2 精簡檔；v1 仍可由 tag `interaction-gp-v1` 或改回路徑取得。

### 互動 GP 精簡檔與模型登記（tag `interaction-gp-v1`）

- 使用者要求保存目前模型版本。M1-30 的 pkl 有 131 MB，超過 GitHub 單檔 100 MB。
- 新增 `interaction_gp.py export`：把 pkl 轉成 `.npz` 精簡檔，只存標準化參數、訓練輸入與答案、kernel 數值。
  - `InteractionGpModel.load()` 讀精簡檔時不經 pickle，以固定超參數重建。
  - v1 精簡檔 375 KB；與原 pkl 的預測差最大 1.0e-13 m；兩個閉迴路場景決策逐步相同。
- `main.py` 的 `INTERACTION_GP_MODEL_PATH` 改指向 `models/compact/interaction_gp_v1_m1_30.npz`，其他電腦也能直接執行。
- 新增 [models/registry.md](models/registry.md) 記錄版本、雜湊、資料、指標與重建方式。
- 本次提交同時收錄使用者在 main.py 的測試設定：vehicle-1 用 interaction_overtake、7/6 m/s、relative_overtake 的 deviation_weight=0.0。

### 互動 GP 超車策略 interaction_overtake（既有策略行為不變）

- 新增 `interaction_overtake`（`strategy/interaction_overtaking.py`）：繼承 `relative_overtake`，前車改用 13 維互動歷史 GP 預測。
  - `conditioning="candidate"`：每個候選用自己的目標查詢。
  - `conditioning="previous"`：用上一次目標查詢。
  - 另檢查 平均 ± `band_sigma`·σ 的邊界軌跡。
  - 超出 GP 訓練範圍或歷史不足時退回定速等 d。
- 新增 `strategy/interaction_features.py`（訓練與線上共用特徵、歷史緩衝區）與 `strategy/interaction_gp.py`（模型載入、範圍檢查、前車軌跡）。
  `training_data/interaction.py` 改呼叫共用函式，重建資料特徵逐位元相同。
- `GpPredictionSnapshot` 加可選欄位：Δd 平均、標準差、時域、d_target；畫面標籤顯示 μ±σ。
- `main.py`：⑤-2 設定（`INTERACTION_GP_MODEL_PATH`、`INTERACTION_OVERTAKING_CONFIG`）與 `--interaction-overtake` 示範場景；預設場景不變。
- 新增 14 項測試；全套 439 項，只剩先前同名的 3 項失敗。11 個等價場景與修改前逐位元相同。
- 第一版評估（20 組成對）：三組都沒有超車，GP 只在 6.6% 決策啟用。見 [評估](reports/experiments/2026-11-24_interaction_overtake/README.md)。

### 互動 GP 收集的控制條件（不改變既有資料）

- `interaction_gp.py collect` 新增 `--speed-strata`、`--probe-amplitudes`、`--episode-prefix`：
  - 前車速度依區間平衡分配，先行側與起步狀態也平衡。
  - 可固定試探幅度，並為場次 ID 加前綴。
- 不加這三項時資料不變：舊參數重收的前 5 場場景設定與樣本逐筆相同。
- 新增 3 項測試；全套 425 項，只剩先前同名的 3 項失敗。
- 涵蓋率診斷與速度 × ±0.6 m 實驗見 [診斷](reports/experiments/2026-11-24_coverage_diagnosis/README.md)、[實驗](reports/experiments/2026-11-24_speed_amplitude_probe/README.md)。

### 互動歷史 GP（不改變既有模擬結果）

- `interaction_gp.py collect` 新增 `--features history`：13 欄 = gap、雙車 d／heading_error／speed、前車曲率、過去 0.4 s 兩段雙車 Δd、d_target。歷史不足或歷史窗內有碰撞的樣本略過。
- 新增 `--randomize-probes`（每場隨機試探序列與 hold，獨立亂數）與 train 的 `--feature-subset`（依名稱選欄做消融）。
- train 的評估子集與等速基準改為依欄位名稱查找；舊 7／8 輸入資料照常訓練。
- `ProbeFollowingConfig.probe_sequence`：自訂試探目標序列，預設 None 等同原本左、中、右、中。
- `core/track_geometry.track_tangent_angle()`：取得中心線切線角。
- 三組消融測試 MAE 0.102／0.038／0.030 m，見 [實驗紀錄](reports/experiments/2026-11-24_history_gp/README.md)。全套 422 項，只剩先前同名的 3 項失敗。

### 決策週期容差與橫移接續（會改變部分模擬結果）

- 依使用者決定，修正先前列為已知問題的兩項：
  - B2：`decision_due` 預設容差改為 1e-9。`decision_period=0.1` 時，原本在 5／6 步之間跳動的決策，改為每 5 步準時一次。
  - B3：defend 與規則式 overtake 完成橫移後，改用從目前位置起算的等偏移線。同一目標保持超過一圈時，參考線不再重播過渡。
- 新增 4 項測試，修正前全部失敗：決策次數 55≠60；保持一圈後路徑起點 d 為 0.073／0.001 m，而不是 0.6 m。
- 量測：
  - 預設 relative＋defend 場景（0.05 s 週期）在 6 s 與 21 s 都與修正前完全相同。
  - progress、cost_overtake 的決策由 55→60 次／6 s。
  - cost_overtake＋defend 6 s 的碰撞步數 13→0，完全來自 B2 改變了決策時間點；這只是單一場景，不代表更安全。
  - 資料生成的車輛狀態不變，但記錄的 maneuver 起點欄位在完成橫移後改為接續點。
- 使用 0.1 s 週期策略的結果，不可直接與修正前比較。
- 全套 413 項只剩先前同名的 3 項失敗。見 [修正紀錄](reports/experiments/2026-11-24_decision_period_and_maneuver_fix/README.md)。

### 平順播放與賽道幾何快取（模擬結果不變）

- 使用者回報即時畫面卡頓。實測預設場景：每 3 步重新決策一次，該步策略計算平均約 214 ms，其餘步約 0.03 ms；每幀繪圖約 10 ms。卡頓來自邊算邊畫，不是繪圖。
- 新增 `--replay`（`sim/replay.py`）：先不開視窗算完整場、每 10% 印進度，再依模擬時間播放；跟不上時跳幀維持速度，最後一幀一定顯示。
  - `--speed` 調倍率，預設 `PLAYBACK_SPEED=1.0`。
  - Windows 播放期間暫時把計時器改為 1 ms：實測 Tk 的 `after(20)` 原本約 31 ms，改後約 20.2 ms。
  - `SimulationVisualizer.wait()` 讓等待期間仍處理視窗事件。
  - `--realtime` 維持原本邊算邊畫。
- 即時、最後一幀與播放改用同一份畫面資料組裝（`record_frame`／`frame_kwargs`）。
- 新增 `--track`、`--seconds`：只覆寫這次執行，不必改檔。
- `Track` 快取線段、累積長度與總長（`Track.segments`），座標轉換不再每一步重算整條中心線，算式不變。
  - 預設場景 6 s、各 2 次：進攻每次決策 162.6→106.9 ms、防守 47.0→31.0 ms，總耗時 21.1→13.9 s。
- 驗證：
  - 11 個等價場景與前一次重構後的結果逐位元相同。
  - 全套 409 項（新增 8 項）只剩先前同名的 3 項失敗。
  - Agg 下幀間隔：`--realtime` 中位數 12.9 ms、p95 171 ms，34% 超過 50 ms；`--replay` 中位數 20.0 ms、p95 20.0 ms。
  - 實際 TkAgg 視窗未目視確認。
  - 見 [播放與快取紀錄](reports/experiments/2026-11-24_replay_and_track_cache/README.md)。
- README 移除已過時的現值描述，改為指向 `main.py` 參數區：vehicle-2 的策略、GP 預設開關、候選數、`deviation_weight`、`collision_horizon` 等。

### 架構整理（模擬結果不變）

- 依使用者要求檢查架構，這次只做「策略去重複」與「main 參數區整理」，不改變模擬結果。
- 新增 `strategy/common.py`：集中 5 個策略各自複製的輸入檢查、決策時機、候選橫向目標、前車選擇與跨圈累積間距。另新增 `offset_path.start_or_continue_maneuver()`、`bicycle_prediction.constant_offset_maneuver()`／`prediction_config_for()` 取代複製版本。
- `strategy/assignment.py` 新增 `StrategyConfigs` 與 `create_strategy_factories()`；`main.py` 與 `sim/data_generation.py` 共用同一份策略工廠與參考路徑長度規則。新增策略的步驟見 README「隨時替換策略」。
- 修正：`main.py` 指派規則式 `overtake` 時，參考路徑 4 m 短於過渡 6 m 而報錯。改為與資料生成入口相同，自動延長到涵蓋被指派策略的 `transition_length`；預設場景的路徑長度不變。
- 修正：`--probe-follow` 搭配下載圖名稱時，不再把下載圖的 PNG 背景畫在圓角矩形上。只影響畫面。
- `main.py` 參數區分成 7 區並附目錄：
  - `VEHICLE_SPECS` 改用具名的 `VehicleSpec`，仍可當 tuple 拆解。
  - 顯示設定上移為 `DISPLAY_OPTIONS`。
  - 修正過時註解：候選數、`collision_horizon` 說明、`leader_penalty_weight` 適用模式、`prediction_horizon=1` 改為 `1.0`。
  - 常數名稱與數值不變。
- 新增 `core/validation.py`：8 個設定類別的錯誤訊息改為寫出類別與欄位，例如 `RelativeOvertakingConfig.decision_period must be positive and finite (got 0.0)`。未知策略、車輛 ID、防守對象與地圖名稱會列出可用選項。
- 驗證：
  - 重構前後以 11 個場景比對逐步狀態、最後畫面資料與終端輸出的 SHA-256。10 個相同；`overtake` 不 patch 路徑長度的場景由報錯改為可執行，結果與基線中 patch 為 6 m 的場景相同。
  - 測試中 4 處 patch 目標改為 `strategy.assignment`（策略類別改由該處建立），斷言不變。
  - 全套 401 項（Agg）只剩先前同名的 3 項失敗。
  - 見 [等價驗證](reports/experiments/2026-11-24_refactor_equivalence/README.md)。
- 已知問題（當時使用者決定不改，修正會改變模擬結果；同日稍後依使用者決定修正，見上方「決策週期容差與橫移接續」）：
  - 決策週期浮點漂移：dt=0.02、`decision_period=0.1` 時，實際間隔在 5 步與 6 步之間跳動，6 s 內 55 次而非 60 次。probe_follow 有 1e-9 容差，其他策略沒有；目前 main 的 0.05 s 不受影響。
  - defend 未呼叫 `continue_lateral_maneuver`：同一非零目標保持超過一圈時，`wrapped_forward_distance` 繞回，參考線會重播整段橫移。規則式 overtake 也是同樣寫法。

## 2026-11-23

- 依使用者要求，`relative_overtake` 啟用速度候選時，改為每條橫向候選都搭配巡航與各低速要求（最多 9×4 條）；成本可加安全距離項 `safety_weight`／`safety_distance`。類別預設 0 維持舊成本，main 設為 10／0.5 m。
- 全部候選不可行時，不再沿用舊決策，改選最晚碰撞的候選；同時間選低速，再比進度。
- `defend` 取得 `BicyclePredictionConfig` 時，自車候選改用 Pure Pursuit＋自行車模型 rollout；main 與資料產生入口都會傳入。對齊成本、後車預測與碰撞條件不變。`BicyclePrediction` 新增 `offsets`。
- 預設場景量測：碰撞步數 36→2，但進攻車未完成超車；進攻每次決策 76→255 ms，防守 0.8→31 ms。360 項測試有 2 項失敗，與 11/18 紀錄同名。詳見 `reports/experiments/2026-11-23_speed_cost_fallback_bicycle_defense/`，未 commit／push。
- 量測包含使用者先前已改、未提交的 `LOOKAHEAD_DISTANCE=1.0`。

### 偏離中心線成本

- 依使用者要求，`relative_overtake` 成本加入 `deviation_weight × mean(d_k²)`，`defend` 的 J_race 加入 `deviation_weight × Σd_k²`；類別預設 0，main 兩者設 1.0，尚未調參。
- 修正候選標籤：移除上一輪多放的 `total` 成本項，避免 J 重複加總。
- 預設場景（含使用者後續的碰撞框 0.4 m 寬、7 個候選偏移、防守 `gap < 0` 等未提交參數）：仍完成超車；平均 |d| 攻／守由 0.52／0.52 降為 0.41／0.13 m，碰撞步數 30→20，目標切換 8／6→45／29，耗時不變。
- 365 項測試有 3 項失敗：防守並排判定測試的 2 個子案例來自使用者的 `gap < 0` 修改，另 1 項是既有失敗。詳見 `reports/experiments/2026-11-23_centerline_deviation_cost/`，未 commit／push。

### 避撞責任約束

- 依使用者提供的論文截圖與選定形式（圓形距離、硬約束），新增 `strategy/responsibility.py`：`g=‖Δc‖²−r_col²−ℓ(Δs)≥0`，ℓ 依 Δs（對方 s − 自車 s）以 sigmoid 分配責任，後車負責、並排兩車都負責、領先者放寬。正負號依論文 Fig. 2 與內文，截圖印出的式子相反。
- `relative_overtake`、`defend` 新增 `responsibility` 設定（類別預設 None，維持原行為）：攻方取代矩形重疊剔除；守方改為逐對檢查所有其他車，取代 `collision_check_enabled`。`RearPrediction` 新增 `progress`，`PredictedCollision` 新增 `kind`。
- main 新增 `RESPONSIBILITY_ENABLED=True` 與 `RESPONSIBILITY_CONFIG(r_col=0.6, a=5, b=4.5)`，未調參；`generate_training_data.py` 也會套用。
- 預設場景單次量測：關閉時與修改前完全相同；開啟時 SAT 碰撞步數 78→0、攻方 no_feasible 28→0，攻方首次領先 4.34→3.62 s；守方每次決策 44→70 ms。
- 385 項測試（Agg）有 3 項失敗，與先前紀錄同名；新增 20 項全部通過。不設 Agg 時 `test_visualizer` 另有 3 項與後端有關的失敗。詳見 `reports/experiments/2026-11-23_collision_responsibility/`，未 commit／push。
- 依使用者要求只調參數，讓守方不主動避讓：main 的 `RESPONSIBILITY_CONFIG.slope` 5→40（前車只在完全並排時負責）、`DEFENSE_CONFIG.collision_horizon` 1.5→0.2 s。預設場景攻方仍超車，但 SAT 碰撞步數 0→22；測試結果不變。程式邏輯未改。
- 使用者之後自行把守方 `decision_period`、`collision_horizon` 都改為 0.1 s。
- 依使用者提議新增 `ResponsibilityConfig.leader_soft`，與 `relative_overtake`／`defend` 的 `leader_penalty_weight`：決策當下在後或並排維持硬約束；在前改為 `w × mean max(0, r_col − ‖Δc‖)²` 成本懲罰。類別預設關閉。
- 預設場景量測：
  - 基準：碰撞 37 步，7.12 s 超車。
  - w=0：碰撞 47 步，超車時間相同。
  - w>0：碰撞 53～179 步，超車延後或沒有超車。
  - 沒有改善，所以 main 維持 `leader_soft=False`、權重 0，行為與修改前相同。
- 394 項測試有 3 項失敗，與先前同名。詳見 `reports/experiments/2026-11-23_responsibility_leader_penalty/`，未 commit／push。

### 重疊責任規則

- 使用者之後把 main 改為 `leader_soft=True`（權重 0）。
- 依使用者要求新增 `ResponsibilityConfig.overlap_rule`，與 `leader_soft` 只能擇一：
  - 依決策當下狀態，縱向重疊中（中心 |Δs| < 碰撞框長平均）兩車都負責。
  - 脫離重疊後只剩後車負責。
  - 負責者要保持 ‖Δc‖ ≥ r_col（不套 ℓ），不負責者只計 leader 懲罰。
- main 改為 `leader_soft=False`、`overlap_rule=True`，其餘使用者設定不變；修正了與實際值不符的註解。
- 預設場景量測：
  - 基準（leader_soft、0.1 s）：碰撞 51 步，7.12 s 超車。
  - 重疊規則、守方 0.1 s（目前 main）：碰撞 73 步，10.40 s 超車。
  - 重疊規則、守方 1.0 s：碰撞 26 步，5.52 s 超車。
- 401 項測試有 3 項失敗，與先前同名。詳見 `reports/experiments/2026-11-23_overlap_responsibility/`，未 commit／push。

## 2026-11-18

- 本次提交彙整八輸入 GP、額外低速候選、碰撞顯示與參數紀錄；保留 352 項測試中兩個場景失敗的日誌。原始資料、模型與環境不納入 Git。
- AGENTS.md 加入文件精簡規則：以修改、設定、結果、限制為重點，詳細證據用連結引用。
- 新增 `train_action_gp.py`，依當下 main 的 relative_overtake／defend、0.6×1.2 m 碰撞矩形等設定收集八輸入資料；各場只隨機平移起點，保留原車距、目標速度與策略，20 場各 25 秒。
- `training_data/interaction.py` 可選擇加入後車當下已執行候選的 `d_target`；`interaction_gp.py` 同時支援七／八輸入，`--max-train 0` 使用全部訓練樣本。原七輸入預設、主程式與策略保持原行為。
- 本次共 4026 筆，按整場 12／4／4 切分，2309 筆訓練、910 筆驗證、807 筆測試；八輸入 GP 的驗證 MAE 0.075690 m、測試 MAE／RMSE 0.085968／0.130142 m，測試固定零位移基準為 0.351976／0.468345 m。
- 8 項相關測試通過；全套 352 項有與先前紀錄同名的兩個場景失敗。本次已核對全部標籤、d_target、切分、train-only scaler 及模型重讀，未安裝套件或改動 main。
- 模型尚未接入規劃器；54.20% 樣本在標籤窗內改過目標，不能視為固定任意候選的反事實驗證，最大測試誤差仍有 0.640401 m。紀錄與小型證據見 `reports/experiments/2026-11-18_action_gp/`；完整資料與模型仍在本機，未 commit／push。

- 新增實驗參數紀錄資料夾，保存「加入低速後選後進攻車與防守車互動模擬，碰撞箱調大」的當下設定與程式差異；碰撞箱為 0.6×1.0 m，未重跑模擬或測試。

- 畫面將實體車身碰撞標為紅色 COLLISION，只有判定矩形重疊則標橘色 ENVELOPE OVERLAP。只改視覺化，規劃與碰撞紀錄沿用原規則。
- 新增 show_envelope_overlap 顯示開關，可在 main 關閉橘色提示，保留實體碰撞標示；19 項視覺化測試通過。
- 視覺化 18 項測試通過；全套 348 項有兩個場景測試失敗，用舊視覺化也能重現，未改使用者場景來讓測試通過。

- relative_overtake 可用 `extra_speed_candidate_deltas` 增加速度差；main 新增前車 −4 m/s，保留原本匹配與 −1。目標最低為 0，重複值會去除，空 tuple 保留舊行為。
- 直線測試使用 0.8×0.7 m 碰撞矩形，原三個候選都不可行，新增 5 m/s 要求後可行，首步由 11 降至 10.9 m/s，與預測一致；這只支持該固定決策案例。
- 當下 main 已是 0.6×0.4 m、0.05 s 規劃週期，21 s 前後比較都沒有碰撞，也都未選到低速候選。詳細結果見 `reports/experiments/2026-11-18_extra_speed_candidates/`，尚未 commit／push。
- 完整 346 項測試通過；舊九候選測試明確停用速度候選，避免受 main 開關影響。

## 2026-11-17

### 本次提交

- 一併整理速度控制、試探跟車、GP 訓練及實驗紀錄。
- 推送前重跑 342 項測試，全部通過。原始資料、模型與環境檔仍留在本機。

### 互動 GP 第一階段

- 新增 interaction_gp.py 與 training_data/interaction.py，收集七項狀態特徵及後車動作，預測 0.5 s 前車橫向位移，沿用正式模擬步進及既有 GP；尚未接入策略。
- 實跑 20 場各 15 s、2900 筆樣本，12／4／4 場切分，800 筆擬合；測試 GP MAE／RMSE 0.059223／0.086608 m，優於不動及橫向等速基準。兩倍標準差涵蓋率 89.14%，尚未校準。
- 完整 342 項通過，補充事件輸出後相關 5 項通過。固定防守及試探順序，未驗證泛化；詳見 reports/experiments/2026-11-17_interaction_gp/。資料與模型僅在本機，未 commit／push。

### 試探目標保持至到達後 2 秒

- 修正試探階段仍可隨每次成本比較改選中心線的行為：有可行候選時，固定本階段的橫向目標，只比較其速度候選；該目標全部不可行時，清除計時並允許其他可行路徑。
- `hold_seconds` 的類別預設與 main 明確設定改為 2 s。首次實際到達才計時，保持期間車距波動與追蹤誤差不重置計時；回中也保持相同時間，下一次側向試探仍須車距合適，成本公式與碰撞條件保持原樣。
- 當下圓角矩形、1 m 初距、10／12 m/s 的 15 s 前後量測：修改前有 9 段短促左側目標，修改後為左、右、左三段，首次到達後分別保持 2.06、2.08、2.02 s，未偵測碰撞或 no_feasible_path。
- 新增三項保持行為回歸測試，完整 337 項通過；紀錄見 `reports/experiments/2026-11-17_probe_following/hold_*` 及 `verify_hold.py`。保持的是控制目標，未保證實際偏移全程在容許誤差內；未人工驗證 GUI，尚未 commit／push。

### 試探跟車加入原巡航速度候選

- `probe_follow` 保留前車速度 −speed_step、匹配、＋speed_step 三種要求，另加入原巡航速度並去重，每條路徑最多四種速度。候選沿用原 rollout、成本與碰撞／禁止超越條件；main 的參數及預設場景保持原樣。
- 固定前次 3 m 初距、10／12 m/s 的圓角場景比較 15 s：最大車距由 8.805 降至 7.620 m，末距由 3.684 降至 1.878 m；新候選被選中 31／150 次，第 7.16 s 首次回到試探車距區間。兩組均未偵測碰撞或後軸進度超越，但新場景只出現左試探，尚未完成左右循環，起步落後仍存在。
- 新增四項測試，並讓三個依賴 2.5 m 目標的既有測試明確設定配置，保留類別預設 1.5 m。完整 334 項測試通過；目前 main 的 1 m 初距場景在 Agg 下完成 15 s，未回報碰撞，未人工驗證 GUI。
- 比較程式、JSON、逐步 CSV、圖表與測試日誌位於 `reports/experiments/2026-11-17_probe_following/cruise_*` 及 `compare_cruise.py`，尚未 commit／push。

### 追近距離的量測與原因定位

- 重現目前圓角矩形、前車 10 m/s、後車巡航上限 12 m/s、初距 3 m 的 15 s 場景。車距曾增至 8.805 m，最後為 3.684 m，150 次規劃中有 139 次選中最快候選。
- 確認目前最快候選受前車當下速度＋1 m/s 限制，穩態僅約 11 m/s。第 4、8、12 s 另計算未執行的 12 m/s 候選，均在預測視野內可行且成本較低，結果支持候選集合限制追近能力。
- 類別 target_gap 預設 1.5 m，但 main 明確指定 2.5 m，實際試探區間為 1.5～3.5 m。僅新增診斷程式及證據，正式策略未修改，修正建議仍待驗證，見 `reports/experiments/2026-11-17_probe_following/gap_diagnosis_*`。

### 補回 main.py 的試探策略註冊

- 檢查到磁碟上的 main.py 缺少 probe_follow 的匯入、配置、工廠及示範入口，導致一般 STRATEGY_ASSIGNMENTS 指派後拋出 unregistered strategy name。補回整組入口並新增一般配置的回歸測試。
- 將防守的 vehicle-1 起點調整為 s=4 m、試探的 vehicle-2 為 s=0 m。車速沿用 10／12 m/s，模擬時間 15 s、地圖 Oschersleben。
- 完整 330 項測試通過，舊兩圈回歸場景固定使用 21 s。一般 `main.py --realtime` 在 Agg 下完成 15 s、exit code 0、未回報碰撞，此趟試探車僅出現 FOLLOW。
- 本次為入口修復，詳細設定與日誌見 `reports/experiments/2026-11-17_probe_following/registration_*`。本次未 commit／push。

### 目標相對進度的試探跟車策略

- 新增獨立 `probe_follow` 策略，以目標車距誤差、終點相對進度速度、試探位置與目標切換四項正規化成本選路徑及速度，支援追近、匹配與減速候選。
- 左、回中、右、回中的試探循環依實際到達與保持時間推進，保留進行中的橫移起點。每輪對手各預測一次，候選執行既有碰撞檢查與整段車身縱向投影留距剔除。
- `main.py --probe-follow` 新增圓角矩形雙車示範，前車巡航 3 m/s、後車上限 4 m/s，從靜止與 4 m 初距起步。資料入口同步註冊策略、配置及配對，小型 CSV 記錄動作與防守反應。
- 21 s／1050 步測試完成 210 次評估，車距由 4.0000 m 至 2.2878 m，末 5 s 車距 MAE 為 0.1565 m，最小實際車身縱向留距 1.3758 m，未超越或偵測碰撞，出現左右試探與前車左右防守。
- 新增 11 項測試，完整 329 項通過。僅量測此場景，GP 互動模型的標籤與訓練尚未實作。設定、CSV、JSON、圖表與日誌見 `reports/experiments/2026-11-17_probe_following/`，本次未 commit／push。

### 雙車閉迴路速度候選驗證

- 新增單項回歸測試與 `verify_closed_loop.py`，重用原速度候選案例及 `run_episode()`，以真實策略在 1 s 內持續重規劃，橫向候選僅於驗證配置限制為 d=0。
- 共評估候選 9 次，含初始規劃及後續 8 次重規劃。首次選中 3 m/s 要求，執行首步為 −3 m/s²，實際車速由 6 降至 5.94 m/s，前車全程維持 3 m/s，51 個狀態檢查均無碰撞。
- 新增測試 1 項通過，本次僅新增驗證與紀錄，結果涵蓋此單一 1 s 場景。逐步數據與測試日誌位於 `reports/experiments/2026-11-17_speed_requests/closed_loop_*`，尚未 commit／push。

### 策略速度要求與離散速度候選

- `StrategyDecision`、相對進度候選與診斷快照新增可選 `speed_limit`，經主模擬與資料收集入口傳入控制器。None 恢復巡航目標，0 透過正常控制逐步停止，負值與非有限值會被拒絕。
- Pure Pursuit 與 bicycle rollout 共用比例速度控制，新增正值 `max_deceleration`。省略時沿用舊版對稱限制，每次速度要求僅改變當次目標。
- `relative_overtake` 新增預設關閉的 `speed_candidates_enabled` 與 `speed_candidate_delta`。有前方目標時，在目前 maneuver 最多加入兩個低速候選，使用原有碰撞剔除與相對間距成本，選中後連同路徑傳遞速度。
- 新增 17 項測試通過。修改前完整 300 項有兩項既有賽道依賴失敗，本次固定該兩項測試的圓角場景後，完整 317 項通過，並以 Agg 完成原預設 21 秒執行。
- 固定直線案例的巡航候選在 0.66 s 預測碰撞，策略選中 3 m/s 候選，實際首步為 −3 m/s²、5.94 m/s，固定決策逐步結果與 rollout 一致。結果僅驗證資料流與短視野候選比較，完整 fallback 與安全煞停尚未驗證。
- 操作見 `README.md`，設定、結果與日誌位於 `reports/experiments/2026-11-17_speed_requests/`。本次未 commit／push。

## 2026-11-16

### 事件修正後的新舊圖比較

- 以提交 6dc4a2b 的正式程式、各一次 8 秒 Agg 比較，原圓角 GP／新圖的重新決策平均為 175.26／188.96 ms，每步含繪圖為 46.71／46.45 ms，未重現新圖大幅較慢；不是 GUI FPS。
- 記錄點數與對手預測差異，以及新圖長度 260.63 m、原圖 58.85 m 的視覺尺度影響；實際 GUI 額外差距仍未定位。未修改正式程式或重跑測試。
- 證據位於 `reports/experiments/2026-11-16_postfix_comparison/`，尚未提交／推送。

### 本次進度提交與推送

- 依使用者指示，將地圖接入、背景快取、TkAgg 事件等待修正及各階段小型證據一併納入提交與推送。
- 測試沿用本次已完成紀錄：固定原圓角場景後 300 項通過；目前新圖直接跑全套有 2 項舊場景假設不符。真實 Tk 視窗驗證仍受 init.tcl 啟動錯誤限制，本次未重跑測試或實驗。
- 原始地圖、GP 模型、原始訓練資料與本機環境仍被忽略；歷史實驗中的「未提交／推送」描述保留當時狀態。

### 修正 TkAgg 事件等待造成的重繪

- 正式程式僅調整視覺化快取分支，以 canvas.start_event_loop 取代 plt.pause，避免再次 show 排程完整重畫；零等待不啟動無限事件迴圈，其他後端保留原方式。
- 視覺化 16 項測試通過；固定原有圓角測試場景後，完整 300 項通過。直接使用目前 Oschersleben 選圖跑全套時，2 項舊測試的圓角／GP 假設不符而失敗，未改使用者選圖。
- 真實 TkAgg 隱藏視窗驗證受本機 init.tcl 啟動錯誤阻擋，未取得 GUI 效能數字；證據在 `reports/experiments/2026-11-16_tk_cache_fix/`，未提交／推送。

### 快取後仍卡頓的原因追查

- 8 秒／400 步 Agg 比較未重現新圖大幅變慢；原圓角／Oschersleben 重新決策平均約 106／113 ms，平均含繪圖步驟約 28 ms。
- 本機預設 TkAgg；檢查實際套件原始碼並以模擬視窗驗證，保留的 plt.pause 會經 show 再次呼叫 draw_idle，導致完整重繪。前次快取測試略過 pause，未涵蓋此 GUI 路徑。
- 本次只定位與修正結果解讀，未改正式程式或重新跑完整單元測試；證據在 `reports/experiments/2026-11-16_cache_comparison/`，尚未提交／推送。

### 視覺化背景快取

- 正式程式僅調整 `sim/visualizer.py`，即時模式快取 PNG 與座標軸、重繪動態圖元；縮放後重建，存圖與不支援後端保留完整繪製。策略與時間週期未變。
- 原雙車 Oschersleben、2 秒／100 幀 Agg 比較：圖元更新＋渲染平均由 141.37 ms 降為 8.38 ms，完整重繪由 100 次降為 1 次，兩組最終狀態一致。各案僅一次，非 GUI FPS。
- 新增 3 項視覺化測試，完整 299 項通過；證據在 `reports/experiments/2026-11-16_background_cache/`，未 commit／push。

### 換圖卡頓的效能量測

- 以原雙車配置、各 2 秒／100 步、Agg 每步重繪比較內建圖與 Oschersleben；同圖開啟 PNG 後重繪平均由 14.65 ms 升至 140.14 ms，顯示背景重繪是本次新增卡頓主因。
- 無 PNG 的重新決策平均由內建圖 110.17 ms 增至外部圖 115.72 ms；cProfile 顯示原有候選自行車預測、臨時路徑與全賽道座標查詢仍占主要策略計算。
- 僅新增量測紀錄，未改正式程式或重新跑測試；各案僅一次短場景，非 GUI FPS。證據位於 `reports/experiments/2026-11-16_map_performance/`，尚未提交／推送。

### 縮減地圖接入修改範圍

- 依使用者要求撤回額外單車入口、附加命令列選項、正式程式內結果匯出及 TrackSource／訓練資料格式擴充；恢復 Track、訓練資料程式及原執行介面。
- 僅保留原有 Track 工廠的外部圖分支、CSV 載入、PNG 顯示及舊圖 GP 防誤用保護。原有車輛、策略、控制器、模擬迴圈與正式參數不變。
- 下方單車與較大範圍接入為先前歷史紀錄；縮減後重新驗證與證據見 `reports/experiments/2026-11-16_map_minimal/`，舊數值與雜湊不回填。

### 地圖作為原有多車架構的單一選項

- 依使用者要求移除 `main.py` 的額外 `TRACK_SOURCE` 開關，只改 `TRACK_NAME` 即可在原有多車競速與資料收集入口選用下載地圖；沿用 `VEHICLE_SPECS`、`STRATEGY_ASSIGNMENTS` 與所有正式策略參數。
- 用原有 `relative_overtake`＋`progress` 雙車配置在 Oschersleben 跑完 21 秒、各 1050 步，PNG 與候選路徑正常顯示。此場偵測到兩車碰撞，沒有宣稱無碰撞或策略效果改善；外部圖保留停用舊 GP 的保護並隱藏其圖例。
- 本次完整測試 299 項通過；小型圖表、設定、實際輸出與測試日誌位於 `reports/experiments/2026-11-16_map_multivehicle/`，未 commit／push。

### 外部地圖載入、PNG 疊圖與單車循跡

- 新增 `core/track_loader.py`，讀取 F1TENTH 的 x/y/右半寬/左半寬 CSV，以公尺弧長重採樣閉合路徑；驗證有限值、固定對稱路寬、重複點及異常接縫。工廠保留原有內建圖，資料匯出／重載新增可選來源與 SHA-256，舊資料仍可讀取。
- 新增 `sim/map_background.py`，解析下載資料的扁平 YAML，以解析度、原點及 yaw 疊加 PNG；不新增套件、不將影像當作控制器或碰撞輸入。
- `main.py` 新增 `--track`、`--single-vehicle`、圈數／速度／時間上限、背景開關及結果匯出。單車沿用 Pure Pursuit 與原自行車模型，外部圖停用原賽道的 GP；原有多車預設設定保留。
- Oschersleben 以 0.4 m 間距產生 652 點，目標速度 5 m/s、步長 0.02 s 完成 2.00024 圈（104.44 s、5222 步）；最大 |d| 0.3647 m、RMS 0.0900 m，四個實體車身角點越界樣本為 0。僅驗證此單車設定，未量測新圖多車超車效果。
- 本次完整測試 299 項通過；修正既有防守整合測試場景的 6 m 過渡／4 m 路徑不一致，僅將測試參考路徑固定為 6 m，正式策略參數未變。23 圖通過載入及中心線像素對齊檢查，Agg 即時流程更新 11 幀且重用單一背景圖元，未人工操作 GUI。
- 以專案內既有 Python 3.10.20 修復本機 `.venv` 啟動指向，未安裝套件；程式、研究紀錄與小型證據尚未 commit／push，原始地圖仍維持忽略。證據位於 `reports/experiments/2026-11-16_map_import/`。

### 流程完成摘要與推送

- 依使用者明確說明，將「GP 資料收集 → 訓練 → 模型預測已串接，並已實際跑過預測」置於最新一週摘要最前面，直接附已有整圈路線與對手預測圖。
- 本次只更新文字紀錄，將目前既有程式與實驗圖、CSV／JSON、日誌一併納入提交；未重跑測試、訓練、預測或產圖，以下測試與執行結果皆為先前紀錄。
- 完整模型與原始訓練資料維持既有忽略規則，未上傳；圖表與小型證據位於 `reports/experiments/2026-11-16_gp_visualization/` 與 `2026-11-16_gp_attack/`。

### GP 路線視覺化

- 新增 `plot_gp_route.py` 與 `sim/gp_visualization.py`，輸出整圈 GP 平均路線及 s→d 圖，另保存取樣 CSV、模型雜湊與賽道 JSON；不模擬或訓練。
- `CandidateSnapshot` 增加 GP 預測顯示資料，直接沿用碰撞篩選所用的 states/times；視覺化不重新預測，不改成本與控制。
- 即時畫面以紫色點虛線與終點圓點顯示對手 GP 軌跡，標示觀測者／對手 ID、評估時間與預測時長；支援獨立曲線／標籤開關、圖元重用與多觀測者。
- 本次相關 34 項測試通過；完整執行 281 項，仍有既有防守整合初始化錯誤。使用原模型輸出 1000 點整圈圖，並以 Agg 執行 2 秒即時更新流程共 100 幀、保存最後一幀，兩張 PNG 已目視檢查；未實際操作 GUI。
- 更新 README、GP_TRAINING 與當週研究紀錄；小型圖片與日誌保存在 `reports/experiments/2026-11-16_gp_visualization/`，未安裝套件、未重新訓練或推送。

### GP 接入進攻

- 新增 `strategy/gp_prediction.py`，載入既有同時刻 s→d GP，以定縱向速度產生時間對齊的對手位置與路線航向。
- `relative_overtake` 可依對手 ID 選用 GP，沿用原矩形碰撞篩選及相對進度成本；各進攻車獨立保存速度估計歷史。
- `main.py` 預設為 vehicle-2 啟用本機模型，新增集中設定開關與模型路徑；資料收集入口保留原預測。依使用者指示，未加入離群補償、平滑接續或不確定範圍處理。
- 本次 44 項相關測試通過；完整測試執行 275 項，唯一錯誤仍為既有防守整合初始化的 6 m 過渡與 4 m 路徑不一致。
- 載入既有模型完成預設 21 秒無視窗場景，本次 1050 個模擬步驟未偵測到實際重疊；這是單一場景接入驗證，未量測多場景成功率或受控改善。證據見 `reports/experiments/2026-11-16_gp_attack/`。
- 使用原有 .venv 的 Python 3.10.10 驗證，未安裝套件或重訓既有模型；本次未 commit 或 push。

### 先前文件整理

- 將研究紀錄主頁改為每週 3～5 個主題摘要，以 11/11 紀錄整理第一個範例；未新增研究實驗或修改演算法。
- 原長版移至 `reports/archive/2026-11-11_research_log_full.md`，修正移動後相對連結與 README 的舊段落入口；詳細證據沿用 `reports/experiments/`。
- 更新 AGENTS.md 的摘要維護規則，同週後續進展優先合併已有主題；研究目的、理由與下一步僅記使用者明確說明。

## 2026-11-11

### 上次推送後內容彙整與本次提交檢查

- 確認遠端 `master` 基準為 `d3910dd`，彙整其後的相對進度策略、可調碰撞尺寸、同時刻資料欄位／GP 支援與研究交接文件；詳細差異見 `reports/research_log.md`。
- 本次重新執行完整測試：`Ran 264 tests in 21.436s`、`FAILED (errors=1)`；唯一錯誤仍為防守整合測試的 6 m 過渡與 4 m 參考路徑初始化設定不一致，未修改演算法。
- 三組 GP、1352 筆預測與 14 個封存檔雜湊核對通過；本次日誌與環境保存在 `reports/experiments/2026-11-11_push_review/`。
- 修正首次整理的三個 JSON 說明欄位亂碼；使用者已授權將本次程式、文件與小型證據 commit 並 push，原始資料與模型沿用忽略規則。
- 新增 `.gitattributes` 保留實驗證據原始位元組，確保提交後可跨平台核對來源 SHA-256。

### 研究紀錄與交接設定（本次僅整理文件）

- 在 `AGENTS.md` 附加長期紀錄規則，區分使用者意圖、可查證事實與未確認建議。
- 新增 `reports/research_log.md` 目前研究狀態基線，以及 `reports/experiments/2026-11-11_baseline/` 小型證據；保留原始資料與模型。
- 依三組既有 GP 的逐筆預測重算誤差，確認與原報告一致；保留 1 s 預測未優於零值基準的不利結果。
- README 同步目前策略、碰撞尺寸與成本式進攻示範參數，新增研究紀錄入口。
- 本次選定測試輸出為 `Ran 62 tests`、`FAILED (errors=1)`；防守整合測試在 `setUpClass` 觸發既有路徑長度設定錯誤。未重跑完整測試或大型實驗，未重訓既有模型；欄位介面測試含微型 GP 擬合。
- 記錄既有 `.venv` 啟動失敗，以及使用專案內 Python 3.10.20 搭配現有套件的檢查環境；未修改演算法、未 commit 或 push。
- 下方原有驗證段落為既有工作紀錄，不是本次重新執行的結果；來源與限制詳見研究基線。

### 新增與調整

- 新增 `relative_overtake`，唯一成本為預測終點的對手 s 減自車 s，移除額外自車進度項及終點超車成功獎勵。
- 使用目前有號間距加兩車累積進度差，支援跨圈並保留負成本表示領先。
- 保留所有其他車的預測碰撞篩選；超車後繼續保留原對象並比較負成本，有新前車時優先切換。
- 同一對象的相對間距跨決策連續累積，領先半圈或一圈仍不跳變；只有對象消失且沒有新前車時才停用成本。
- 沒有比較對象時不加入替代進度項，優先沿用仍可行的原目標。
- 新增獨立的 `RELATIVE_OVERTAKING_CONFIG`，預設九條候選、0.8 s 預測與 0.1 s 決策週期。
- 新增共用 `COLLISION_CONFIG` 可調碰撞矩形長寬，保持原車身中心；預測與實際判斷共用，車身繪圖及路寬限制仍使用原尺寸。原始資料保存碰撞尺寸並相容舊檔。
- 主程式、候選繪圖與無視窗資料生成均支援新策略，設定保存至 metadata；原有策略及目前指派保留。
- 原成本式進攻的五候選回歸測試固定自己的候選設定，避免受到示範候選改為九條影響。

### 驗證

- 相對進度成本、跨圈、負間距、無對象、多車碰撞、超車後保留對象、半圈／整圈間距及資料生成共 20 項測試通過。
- 完成 21 秒 relative_overtake 與 progress 混合場景，超車後成本持續運作，最後選中成本約 -15.91 m。
- 此混合場景記錄到車輛碰撞；預測篩選仍保留，但定速等偏移的對手假設不能保證實際無碰撞。
- 碰撞幾何、實際／預測碰撞與訓練資料相關 45 項測試通過，主程式完成 21 秒執行驗證。
- 完整測試執行 264 項，仍有修改前已存在的防守整合測試初始化錯誤（6 m 橫移與 4 m 參考路徑不一致）。
- `git diff --check` 通過；未 commit 或 push。

## 2026-11-10

### 新增與調整

- 新增 `training_data/fields.py`，集中選擇輸入／輸出及註冊 `InputField`、`OutputField`。
- 新增 `opponent_s` 與 `opponent_current_d`，預設學習 progress 車同時刻的 s→d 行車路線。
- 答案以 `timing="current"` 或 `timing="future"` 註明時間；CLI 自動選擇 0 或 1 步。
- 同時刻資料包含最後一幀；未來資料仍須有足夠未來狀態。錯誤時間設定或混用兩種答案會明確報錯。
- 原有四輸入／未來位移欄位與 Python 註冊介面保留；既有 CSV／JSON 資料可繼續載入。
- 新增 `--list-fields`，可直接查看名稱、單位、說明與答案時間。
- 配對優先以 progress 車作為對手；main.py 中保留使用者啟用的 vehicle-2，將其策略改為 progress。
- GP 自動選用唯一答案，保存同時刻／未來模式；增加訓練平均值基準及單輸入同時刻對應圖。
- 輸出 metadata 及 GP 模型記錄答案時間，避免將同時刻 d 誤標為未來預測。

### 驗證

- 欄位註冊、同時刻／未來配對、資料生成與 GP 相關 51 項測試通過。
- 完整測試執行 239 項，仍有既有防守整合測試的 1 個路徑長度設定錯誤，未更動該策略設定。
- 使用目前場景實際收集 3 次各 2 秒，共 63 筆 opponent_s→opponent_current_d 資料，
  並以 42 筆訓練、21 筆測試完成 GP 執行驗證；此小樣本僅用於驗證流程，不作為精度結論。
- 測試使用原有 .venv，未安裝 Python 或套件；本次未 commit 或 push。

## 2026-11-09

### 新增

- 新增 `train_gp.py`，直接讀取收集程式輸出的資料，訓練指定的單一答案欄位。
- GP 依 episode 切分訓練與測試資料；輸入標準化只使用訓練集，並與模型一起保存。
- 訓練輸出包含模型、MAE／RMSE 與零值基準報告、逐筆預測及標準差、評估圖。
- 新增 GP 操作說明 `GP_TRAINING.md` 與獨立套件清單 `requirements-gp.txt`。
- 新增圓角矩形賽道，保留原有橢圓賽道，透過共同入口切換與調整尺寸。
- 新增 `progress` 策略，只比較自車五條候選軌跡的前進進度，不預測其他車或篩選預測碰撞。
- 新增 GP、賽道幾何、賽道切換及 progress 策略的 30 個測試案例。

### 調整

- 模擬與無視窗資料生成共用 `create_demo_track()`，使用同一份賽道設定。
- 資料生成支援 progress 策略、自動配對及設定紀錄。
- 預設地圖改為中心線跨度 20 × 12 m、彎道半徑 3 m 的圓角矩形。
- 示範配置改為 vehicle-1 使用 cost_overtake、vehicle-3 使用 progress，基準起始間距為 4 m。
- 防守對齊條件由指定對象在後方超過 2 m，改為超過 1 m；仍受偵測範圍限制。
- 防守邊界測試同步改為 1 m，並覆蓋後方 1～2 m 內應啟用對齊成本的情境。
- 防守橫移過渡長度由 4 m 改為 2 m，決策週期由 0.3 s 改為 0.2 s。
- 視覺化使用通用模擬器標題，README 補充賽道與純進度策略操作說明。
- Git 忽略訓練資料與模型產物，檔案保留在本機，程式與文件納入版本控制。

### 已知限制

- GP 精度仍需改善，目前預設輸入未包含對手相對賽道的航向資訊。
- 舊規則式超車的 6 m 過渡長度與主程式的 4 m 參考路徑長度不一致。
- 示範中 vehicle-1 的目標速度為 11 m/s，實際速度仍受車模 10 m/s 上限限制。

### 驗證

- GP、資料生成、progress、賽道及防守單元測試共 75 項通過。
- 完整 `python -m unittest` 執行 225 項測試，仍有 1 個已知錯誤：
  `tests.test_defending_integration.DefendingIntegrationTests.setUpClass`
  觸發 `path_length must cover the complete transition`，原因為上述 6 m／4 m 設定不一致。
- 測試使用既有 `.venv`（Python 3.10.10、scikit-learn 1.7.2），未安裝 Python 或套件。
- `git diff --check` 通過。
