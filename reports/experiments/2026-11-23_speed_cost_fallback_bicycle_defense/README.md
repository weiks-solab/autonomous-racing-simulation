# 速度入成本、最晚碰撞退路、防守 bicycle 預測

日期：2026-11-23；分支 `master`；基於 `1d4a182`，含未提交修改。
所有量測都包含使用者在本次工作前已改、未提交的 `LOOKAHEAD_DISTANCE` 2.0→1.0（main.py 修改時間早於基準量測）。

## 使用者要求

使用者明確要求：
1. 進攻車評估時把速度一起納入成本選擇。
2. 全部候選不可行時改用最晚碰撞的路徑。
3. 防守車使用 bicycle model 預測軌跡。

選用的形式：
- 速度採用 9 橫向 × 速度的全組合，並加入安全距離項。
- 碰撞時間相同時先選低速，再比進度。
- 防守車只替換自車軌跡的預測方式。

修改理由使用者未說明。

## 修改

**進攻車 `relative_overtake`**（[relative_overtaking.py](../../../strategy/relative_overtaking.py)）
- 每條橫向候選都搭配巡航，以及前車速度、−1、−4 m/s，最多 36 條。原本速度候選只放在目前路徑上。
- 成本改為 `J = gap_T + w·Σ_o mean_k max(0, d_s − clear_k)²`，其中：
  - `clear_k = max(|Δs_k| − L̄, |Δd_k| − W̄)`，是 Frenet 座標下的矩形間隙近似。
  - L̄、W̄ 是兩車碰撞矩形長、寬的半和。
  - main 設定 `w=10 /m`、`d_s=0.5 m`；類別預設 `w=0`，等於維持舊成本。
- 全部不可行時，排序依序為：最晚的第一次碰撞時間、較低的速度要求、較多的自車進度。決策採用該候選的路徑與速度要求，不再沿用舊決策。

**預測**
- `BicyclePrediction` 新增 `offsets`，記錄每步的 d。GP 預測同步補上 offsets。

**防守車 `defend`**（[defending.py](../../../strategy/defending.py)）
- 有 `BicyclePredictionConfig` 時，自車候選改用自己的 Pure Pursuit＋自行車模型 rollout，再插值到 0.1 s 的成本時間點。
- 對齊成本、後車預測、碰撞條件不變；碰撞條件在 main 中仍是關閉的。
- 設定由 `StrategyManager(prediction_configs=...)` 傳入，main 與 `sim/data_generation.py` 都有傳。

## 實驗設定

- 場景：main 預設，`rounded_rectangle`、21 s、dt 0.02 s。
  - vehicle-1：`relative_overtake`，11 m/s，起點 s=0。
  - vehicle-2：`defend`，9 m/s，起點 s=4。
  - 碰撞矩形 0.6×1.2 m；GP 關閉。
- 每組各跑一次（模擬是確定性的），不是多初始條件的統計結果。
- 命令（Agg 後端，Python 3.10.20、NumPy 2.2.6、Matplotlib 3.10.9）：

```powershell
.\.venv\Scripts\python.exe reports/experiments/2026-11-23_speed_cost_fallback_bicycle_defense/measure.py <out.json> [--safety-weight 0] [--geometric-defense]
```

[measure.py](measure.py) 只包裝策略的 `update` 計時，並記錄每步碰撞與縱向進度，不改變行為。
- `before` 在修改前的工作區執行。
- `after` 與兩組對照組在修改後執行。

## 結果

| 組別 | 最終 s（攻／守） | 超車 | 碰撞步數（首次） | no_feasible 次數 | 攻／守 目標切換次數 | 進攻每次決策 平均／最大 | 防守每次決策 平均 | 總耗時 |
|---|---|---|---|---|---|---|---|---|
| [before](before.json) | 222.4／192.8 | 是 | 36（6.56 s） | 18 | 27／14 | 76／106 ms | 0.8 ms | 27 s |
| [after](after.json) | 185.0／186.1 | 否 | 2（4.14 s） | 1 | 167／48 | 255／278 ms | 31 ms | 93 s |
| [安全項 0](ablation_safety0.json) | 191.7／192.6 | 否 | 20（4.00 s） | 7 | 124／24 | 252／286 ms | 32 ms | — |
| [防守幾何](ablation_geometric_defense.json) | 184.9／185.9 | 否 | 11（4.14 s） | 4 | 178／46 | 245／266 ms | 0.8 ms | — |

- 進攻車每輪都有 350 次決策（實際週期 0.06 s），防守車 100 次。
- 選到速度要求的決策次數：before 30 次；修改後三組都在 213～217 次之間。
- 完整日誌：[before.log](before.log)、[after.log](after.log)、[ablation_safety0.log](ablation_safety0.log)、[ablation_geometric_defense.log](ablation_geometric_defense.log)。

**目視觀察（依日誌）**
- before 的超車發生在 6.24–7.32 s 的 no_feasible_path 期間，碰撞也從 6.56 s 開始。
- 修改後，進攻車多數時間以低速跟在防守車後方，左右目標頻繁切換。

**Claude 分析，未經使用者確認；待驗證**
- before 的「超車」可能是沿用舊的高速決策，直接穿過防守車完成的。本模擬沒有碰撞反應，所以重疊不會阻止前進。
- 修改後拒絕碰撞路徑並減速，結果在 0.6×1.2 m 碰撞矩形與跟隨式防守下，找不到無碰撞的超車路徑。
- 兩組對照都沒有超車，表示安全項和防守車的 bicycle 預測都不是不超車的主因。
- 目標切換次數大增的原因，可能是候選數增加，加上缺少切換成本。

## 驗證

- `python -m unittest`：360 項，2 項失敗（[日誌](unittest.log)）。
  - `test_passive_vehicle_is_unchanged_and_defender_actually_moves`
  - `test_relative_cost_stays_active_after_the_actual_overtake`
  - 兩項在 11/18 的紀錄中就已失敗（[11/18 日誌](../2026-11-18_action_gp/unittest.log)）。前者失敗的斷言出現在停用防守的被動場景（規則式 overtake 加中心線車），沒有用到本次修改的程式；把 lookahead 改回 2.0 仍然失敗。
- 新增與更新的測試涵蓋：offsets、全組合候選、安全項讓低速勝出、最晚碰撞與同時間選低速、防守 bicycle rollout 與插值成本、管理器傳遞設定。
- 依賴舊行為（只在目前路徑加速度候選、全部不可行時沿用舊決策）的 4 項測試，已改成新行為的斷言。

## 限制

- 只量測單一場景與單一初始條件，不能代表超車成功率或碰撞率。
- 安全項權重沒有調參。Frenet 間隙在彎道與 SAT 判定不一致。
- 對手預測仍是等速、等 d、不互動。
- 計算量約為原本的 3.3 倍，每次決策 255 ms，超過 0.06 s 的決策週期，無法即時運算。
- `sim/data_generation.py` 也套用了新行為，今後產生的訓練資料與 11/18 的資料不同。
