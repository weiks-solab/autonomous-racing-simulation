# 2026-11-24 決策週期容差（B2）與橫移接續（B3）修正

日期：2026-11-24；分支 `master`；基於 `bbd981f`，含未提交修改（含同日的架構整理、`--replay` 與賽道快取）。

## 目的

使用者決定修正先前列為已知問題的兩項，並量測前後差異。這兩項修正會改變部分模擬結果。

## 修改

- **B2 決策週期容差：** `strategy/common.decision_due` 的預設容差由 0 改為 1e-9，所有策略一致（probe_follow 原本就有）。
  - 修正前：模擬時間由步長累加，例如 0.30000000000000004 − 0.2 略小於 0.1，`decision_period=0.1` 的決策會在 5 步與 6 步之間跳動。
  - 0.05 s 週期本來就是每 3 步決策一次，不受影響。
- **B3 橫移接續：**
  - defend 的 `_evaluate_candidate` 改用 `start_or_continue_maneuver`。
  - 規則式 overtake 每次 `update` 對持有的 maneuver 呼叫 `continue_lateral_maneuver`。
  - 完成橫移後改為從目前位置起算的等偏移線。修正前若同一目標保持超過一圈，`wrapped_forward_distance` 繞回，參考線會重播整段過渡。

## 方法

- 新增 [4 項測試](../../../tests/test_decision_timing_and_maneuver.py)，修正前全部失敗，修正後通過：
  - 0.1 s 週期在 dt=0.02 下 6 s 內決策 60 次（修正前 55 次）。
  - progress 策略同樣決策 60 次（修正前 55 次）。
  - defend 保持 0.6 m 目標超過一圈後，臨時路徑起點的 d 為 0.6 m（修正前 0.073 m）。
  - 規則式 overtake 鎖定 0.6 m 一圈後同樣為 0.6 m（修正前 0.001 m）。
- [measure_fixes.py](measure_fixes.py)：4 個場景，記錄各車決策次數與間隔、碰撞步數、累積縱向進度與最終 d。
- `2026-11-23_overlap_responsibility/measure.py`：預設 21 s 場景。
- [verify_equivalence.py](../2026-11-24_refactor_equivalence/verify_equivalence.py)：11 個場景，與修正前比對。
- 分離來源的檢查：以 patch 暫時關閉其中一項修正後重跑，判斷差異來自 B2 或 B3。這些檢查只在終端執行，沒有另存檔案。

## 結果

**場景比較**（[修正前](scenes_before.json)、[修正後](scenes_after.json)）

| 場景 | 決策次數（攻／守） | 碰撞步數 | 進攻車進度 m | 差異來源 |
| --- | --- | --- | --- | --- |
| progress＋centerline 6 s | 55→60（兩台 progress，各 60，間隔都是 5 步） | 0→0 | 55.6427→55.6428 | B2 |
| cost_overtake＋defend 6 s | 55→60／100→100 | **13→0** | 55.4948→55.1081 | 全部來自 B2 |
| 預設 relative＋defend 6 s | 100→100／100→100 | 12→12 | 相同 | 無 |
| 規則式 overtake＋defend 12 s | —／200→200 | 59→59 | 相同 | 無 |

**預設 21 s 場景**（[前](default_21s_before.json)、[後](default_21s_after.json)）
- 所有指標都相同：碰撞 12 步、首次碰撞 4.08 s、首次領先 4.8 s、各車最終 s、平均 |d|、決策 350 次與模式分布。
- 修正前的結果也與同日架構整理時的量測相同。

**等價場景**（[修正後](equivalence_after_fix.json)）
- 11 個場景中 8 個相同。
- `cost_overtake_defend_3s`、`progress_centerline_3s` 因 B2 改變。
- `data_generation_2s` 因 defend 的 B3 改變：兩車狀態 2 s 內逐位元相同，不同的是觀測與決策中記錄的 maneuver 起點欄位（完成橫移後改為接續點）。

**測試**
- 全套 413 項（Agg），含新增 4 項；3 項失敗，與先前同名（[unittest.log](unittest.log)）。

## 限制

- cost_overtake＋defend 的碰撞 13→0 步只是一個 6 s 場景、一次執行。原因是決策時間點改變，不代表策略本身變得更安全。
- 使用 0.05 s 週期的預設場景（relative＋defend）結果不變，可繼續與本週先前的量測比較。
  使用 0.1 s 週期策略（progress、cost_overtake）的結果，不可直接與修正前比較。
- 資料生成的 `raw_states.csv` 中，`maneuver_start_s`／`maneuver_start_d` 在完成橫移後會更新為接續點，與先前資料的欄位意義略有不同。
- B3 在本次量測的主場景沒有造成可見差異，主要作用在保持同一目標超過一圈的情況（由新測試覆蓋）。
