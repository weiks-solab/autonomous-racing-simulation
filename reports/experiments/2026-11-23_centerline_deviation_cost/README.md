# 偏離中心線成本（進攻與防守）

日期：2026-11-23；分支 `master`；基於 `1d4a182`，含未提交修改，包括 [上一輪修改](../2026-11-23_speed_cost_fallback_bicycle_defense/README.md)。

## 使用者要求

- 使用者明確要求：加入偏離中心線的成本，進攻車與防守車都加。理由使用者未說明。
- 使用者已提及的下一步：以最小曲率計算全域賽車線，並把所有車的偏移基準由中心線改為賽車線。尚未實作。

## 修改

**進攻車**（[relative_overtaking.py](../../../strategy/relative_overtaking.py)）
- 成本加上 `deviation_weight × mean_k(d_k²)`，k≥1，d 取自自車 bicycle rollout。單位 1/m×m²=m，可與 gap_T 直接相加。

**防守車**（[defending.py](../../../strategy/defending.py)）
- `J_race` 加上 `deviation_weight × Σ_k d_k²`，與對齊成本使用相同的 0.1 s 取樣點。總成本仍是 J_race + blocking。

**共通設定**
- 類別預設 `deviation_weight=0`，等於維持原成本。main 兩者都設為 1.0，尚未調參。
- 被碰撞剔除的候選不計成本；最晚碰撞退路的排序不變。

**顯示修正**
- 上一輪在 `cost_terms` 放了 `total` 項，而候選標籤會把所有項加總成 J，造成 J 重複計算。這次移除 `total`，標籤的 J 等於策略比較用的總成本。

## 實驗設定

- 場景：main 預設，`rounded_rectangle`、21 s、dt 0.02 s。
  - vehicle-1：`relative_overtake`，11 m/s，起點 s=0。
  - vehicle-2：`defend`，9 m/s，起點 s=4。
- 本輪量測包含使用者在上一輪之後、未提交的參數修改：
  - 碰撞矩形 0.6×0.4 m
  - 進攻、防守、progress 的候選偏移改為 ±0.6／±0.4／±0.2／0（進攻 7 橫向 × 4 速度 = 28 條）
  - 防守觸發條件改為 `gap < 0`
  - `LOOKAHEAD_DISTANCE=1.0`
- 修改前後各跑一次（確定性模擬），使用同一支 [measure.py](measure.py)。這支是上一輪腳本的複本，另外加了平均 |d| 與目標切換次數。

```powershell
.\.venv\Scripts\python.exe reports/experiments/2026-11-23_centerline_deviation_cost/measure.py <out.json>
```

## 結果

| 組別 | 最終 s（攻／守） | 超車 | 平均 \|d\|（攻／守） | 碰撞步數（首次） | no_feasible | 目標切換（攻／守） | 每次決策平均（攻／守） |
|---|---|---|---|---|---|---|---|
| [修改前](before_deviation.json) | 229.3／194.3 | 是 | 0.52／0.52 m | 30（3.94 s） | 11 | 8／6 | 85／46 ms |
| [修改後](after_deviation.json) | 228.3／186.3 | 是 | 0.41／0.13 m | 20（3.76 s） | 8 | 45／29 | 84／45 ms |

- 日誌：[修改前](before_deviation.log)、[修改後](after_deviation.log)。
- 防守車的模式分佈：修改前 defend／return 為 20／80 次；修改後 defend／return／cruise 為 19／53／28 次。
- 計算量沒有可量測的增加。

**目視觀察（依日誌）**
- 兩組的超車都發生在 no_feasible_path 期間之後：修改前 3.90–4.56 s，修改後 3.72–4.20 s，期間都有碰撞。

**Claude 分析，未經使用者確認；待驗證**
- 防守車最終 s 少了約 8 m，可能是改走中心附近，彎道不再走內線，造成以中心線 s 計算的進度較少。
- 目標切換次數大增，可能是偏離項讓相近候選的成本差距縮小，又沒有切換成本。

## 驗證

- `python -m unittest`：365 項，3 項失敗（[日誌](unittest.log)）。
  - `test_no_defense_for_ahead_alongside_far_or_unspecified_vehicle` 有 2 個子案例失敗：使用者把防守觸發條件改成 `gap < 0`，並排車（間距 −0.05、−1.0 m）現在會觸發防守。測試使用預設 `deviation_weight=0`，與本次修改無關，未更動該測試。
  - `test_relative_cost_stays_active_after_the_actual_overtake`：11/18 就已存在的失敗。
  - 上一輪失敗的被動防守場景，在使用者把碰撞矩形改為 0.4 m 寬後通過。
- 新增測試涵蓋：
  - 進攻、防守兩邊的偏離值公式
  - 權重使目標靠向中心線
  - 防守的幾何預測與 bicycle 預測兩種路徑
  - 非法權重被拒絕
  - 標籤的 J 等於總成本
- `test_new_strategy_coexists...` 原本檢查「九條橫向候選」，但使用者改了 main 的候選數。這次在測試中明確指定 9 個偏移與 `deviation_weight=0`，維持原測試的意圖。

## 限制

- 單一場景、單次執行，權重未調參。
- 偏離量以中心線為基準；改為賽車線時需替換參考的 d。
