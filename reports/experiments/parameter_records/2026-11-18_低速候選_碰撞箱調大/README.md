# 加入低速後選後進攻車與防守車互動模擬，碰撞箱調大

記錄日期：2026-11-18。這是當下磁碟設定的快照，本次沒有重跑模擬或測試，也未新增結果判定。
名稱沿用使用者指定；以下「候選」指策略比較的路徑／速度選項。

## 版本與執行方式

- 分支：master。
- 基底 commit：`de3581a06604976980aa6efa36c1cf61b84fa84c`，含未提交修改。
- Python 3.10.20、NumPy 2.2.6、Matplotlib 3.10.9；本機 scikit-learn 1.7.2，但本場景 GP 關閉。
- 使用內建圓角矩形，不需要下載地圖或 GP 模型；本場景沒有隨機初始化，不使用 random seed。

在 repo 根目錄啟動一般場景：

```powershell
.\.venv\Scripts\python.exe main.py --realtime
```

省略 `--realtime` 會先計算再顯示。不要加 `--probe-follow`（會換成另一組試探場景），也不要加 `--no-defense`。

## 場景與車輛

| 項目 | 設定 |
| --- | --- |
| 模擬時間 | 21 s |
| 模擬步長 | 0.02 s，共 1050 步 |
| 賽道 | rounded_rectangle，封閉、逆時針 |
| 中心線左右／上下跨度 | 20／12 m，不含路寬 |
| 彎角半徑／路寬 | 3／2 m |
| 中心線點數 | 500 |
| 進攻車 vehicle-1 | 起始 s=0 m、d=0 m，巡航目標 11 m/s，relative_overtake |
| 防守車 vehicle-2 | 起始 s=4 m、d=0 m，巡航目標 9 m/s，defend，防守 vehicle-1 |
| 兩車初始速度／車距 | 都由 0 m/s 起步；後軸沿賽道車距 4 m |
| 初始位置與 yaw | 由各自起始 s 的中心線位置及前方 0.05 m 切線建立 |
| 防守／GP 總開關 | DEFENSE_ENABLED=True；GP_PREDICTION_ENABLED=False |

VEHICLE_SPECS 的第三個數字是巡航目標，不是初始實際速度。沒有第三台車。

## 車身、碰撞箱與控制

| 項目 | 設定 |
| --- | --- |
| 實體車身長 × 寬 | 0.55 × 0.30 m |
| 後軸到車身中心 | 0.12 m |
| **碰撞判定箱長 × 寬** | **0.60 × 1.00 m，兩車共用** |
| 車輛模型／軸距 | rear-axle kinematic bicycle；0.33 m |
| 模型速度範圍 | 0～30 m/s，不倒退 |
| 最大轉向角 | 50°，即 0.8726646259971648 rad |
| Pure Pursuit lookahead | 2.0 m |
| 比例速度增益 | 1.5 |
| 最大加速度／減速度 | 5.0／5.0 m/s²，設定皆填正值 |
| 臨時路徑取樣間距／長度 | 0.2／4.0 m |

碰撞箱保持同一個車身中心。預測、終端與資料的重疊紀錄使用碰撞箱；畫面的紅色 COLLISION 另看實體車身。
路寬候選限制仍以實體車寬 0.30 m 計算，不會因放大碰撞箱自動變窄。

## 進攻策略 relative_overtake

| 參數 | 值 |
| --- | --- |
| candidate_offsets | (-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6) m |
| detection_range | 6.0 m |
| transition_length | 3.0 m |
| decision_period | 0.05 s |
| prediction_horizon | 1.0 s |
| safety_margin | 0.1 m |
| speed_candidates_enabled | True |
| speed_candidate_delta | 1.0 m/s |
| extra_speed_candidate_deltas | (4.0,) m/s |

九條橫向候選使用原巡航目標；只在目前追蹤的路徑加入前車當下速度 vF、max(vF−1,0)、max(vF−4,0)。
只保留低於巡航目標且不重複的要求，所以每輪最多 12 個候選；速度只會透過控制器逐步改變。
前車若達 9 m/s，額外速度要求為 9、8、5 m/s。
規劃只在模擬步上更新，0.05 s 的配置不代表能在 0.02 s 網格上精準每 0.05 s 更新。

自車 rollout 每步使用 Pure Pursuit 與 bicycle；對手維持定速、等 d 預測，不執行對手防守策略。
通過碰撞檢查後，以預測終點的「前車進度−自車進度」取最小值，仍鼓勵超越；沒有跟車距離成本。
全部候選不可行時，沿用上一個路徑及速度要求，未加入安全 fallback。

## 防守策略 defend

| 參數 | 值 |
| --- | --- |
| q_block／progress_weight／smooth_weight | 15.0／2.0／1.0 |
| detection_range／return_tolerance | 6.0／0.08 m |
| transition_length／decision_period | 2.0 m／0.2 s |
| candidate_offsets | (-0.6, -0.3, 0, 0.3, 0.6) m |
| prediction_horizon／prediction_step | 1.0／0.1 s |
| path_sample_spacing／safety_margin | 0.2／0.1 m |
| collision_check_enabled | False，前車不啟用獨立的候選碰撞剔除 |
| collision_horizon | 1.5 s，因碰撞檢查停用，本次不生效 |

防守成本保留原公式：進度項 −2×progress，加上橫移平順項 1×Σ(Δd)²；防守生效時再加 15×Σ(d−進攻車當下 d)²。
前車只選橫向目標，仍朝自身 9 m/s 巡航目標控制速度。

## 顯示設定

`create_display_options()` 中：

- True：show_track_boundaries、show_vehicle_footprints、show_collision_status、show_temporary_paths、show_candidate_paths、show_gp_predictions、show_gp_prediction_labels、show_trajectories、show_pursuit_targets。
- False：show_centerline、show_vehicle_markers、show_vehicle_heading、show_envelope_overlap、show_strategy_status、show_candidate_labels、show_track_positions。
- 預設 equal_axis=True、show_map_background=True、pause_seconds=0.001；內建賽道沒有背景圖，GP 關閉也不會產生 GP 預測。

目前橘色 ENVELOPE OVERLAP 的文字與變色都隱藏，紅色實體碰撞標示仍開啟。

## 日後還原

除了本表，也保存 [程式差異](source_snapshot.patch) 與 [程式雜湊](source_sha256.txt)。
差異包含 main.py、strategy/relative_overtaking.py、sim/visualizer.py，補回基底版本尚無的額外速度候選與畫面開關。

先準備基於上述 commit 的乾淨工作區，再於 repo 根目錄檢查、套用差異：

```powershell
git apply --check reports/experiments/parameter_records/2026-11-18_低速候選_碰撞箱調大/source_snapshot.patch
git apply reports/experiments/parameter_records/2026-11-18_低速候選_碰撞箱調大/source_snapshot.patch
.\.venv\Scripts\python.exe main.py --realtime
```

需先把本紀錄資料夾帶到該工作區；若當時已包含這些修改，不要重複套用。
雜湊以 UTF-8 原始碼統一 LF 換行計算。此次只保存設定，不宣稱本組 1.00 m 寬碰撞箱已通過任何測試。
