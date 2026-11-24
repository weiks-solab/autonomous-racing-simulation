# 2026-11-24 架構整理的等價驗證

日期：2026-11-24；分支 `master`；基於 `bbd981f`，含未提交修改。

## 目的

使用者要求檢查架構，這次只做「策略去重複」與「main 參數區整理」，並要求模擬結果不變。
本資料夾記錄重構前後的逐位元比對；修改內容見 [CHANGELOG](../../../CHANGELOG.md)。

## 方法

- `verify_equivalence.py` 只用公開介面：patch `main` 的常數與 `SimulationVisualizer`，或呼叫 `run_episode`；不修改設定檔。
- 每個場景記錄三項 SHA-256：
  - 逐步 (x, y, yaw, v)。
  - 最後一次交給畫面的全部資料，包括候選快照、臨時路徑、標籤與碰撞配對。
  - 終端輸出。

  浮點以 hex 表示，不受 repr 或 numpy 型別影響。資料生成場景另記每一幀的觀測，以及每一步的決策與控制命令。
- 修改程式前先產生 `baseline.json`，修改後產生 `after.json`，再以 `compare` 比對；比對時忽略執行秒數。

| 場景 | 內容 |
| --- | --- |
| default_6s | 目前 main 預設：relative_overtake 對 defend，責任約束 overlap_rule |
| default_no_defense_3s | `defense_enabled=False` |
| cost_overtake_defend_3s | cost_overtake 對 defend |
| progress_centerline_3s | 3 車：兩台 progress、一台 centerline |
| probe_follow_demo_3s | `--probe-follow` 場景 |
| overtake_defend_path6_8s | 規則式 overtake 對 defend，參考路徑 patch 為 6 m |
| overtake_defend_auto_path_8s | 同上，但不 patch 路徑長度 |
| rectangle_collision_checks_3s | 關閉責任約束：進攻以矩形剔除，防守啟用碰撞條件 |
| gp_relative_3s | 開啟 GP 對手預測（本機模型，未納入 Git） |
| imported_track_2s | Oschersleben 下載圖與 PNG 背景（本機地圖，未納入 Git） |
| data_generation_2s | `generate_training_data` 設定的一場隨機場景 |

重現（專案根目錄）：

```bash
python reports/experiments/2026-11-24_refactor_equivalence/verify_equivalence.py run <輸出.json>
python reports/experiments/2026-11-24_refactor_equivalence/verify_equivalence.py compare reports/experiments/2026-11-24_refactor_equivalence/baseline.json <輸出.json>
```

## 結果

- **等價比對：** 10 個場景的所有雜湊相同（[baseline.json](baseline.json)、[after.json](after.json)）。
- **overtake_defend_auto_path_8s：**
  - 修改前報錯 `path_length must cover the complete transition`。
  - 修改後可以執行，雜湊與基線中 patch 為 6 m 的場景完全相同，也就是自動延長的結果等同手動設為 6 m。
- **比對本身的確定性：** 修改前另外重跑 2 個場景，雜湊與基線相同。
- **量測腳本仍有效：**
  - 以 `2026-11-23_overlap_responsibility/measure.py` 在修改前後各跑一次預設 21 s 場景（[前](measure_before.json)、[後](measure_after.json)）。
  - 除耗時欄位外，兩份 JSON 完全相同：碰撞 12 步，首次領先 4.8 s。
  - 表示該腳本攔截 `main.find_vehicle_collisions` 與策略 `update` 的方式仍然有效。
- **全套測試（Agg）：** 401 項，3 項失敗，與修改前同名（[unittest.log](unittest.log)）。
  - 2 項是 test_defending_strategy 的並排判定子案例。
  - 1 項是 test_relative_overtaking_integration。
- **CLI：** `python main.py --probe-follow` 與 `python main.py --no-defense`（Agg）可以跑完。

## 限制

- 只比對上述 11 個場景與時長，不保證所有參數組合都逐位元相同；realtime 畫面模式未比對。
- GP 與下載圖場景依賴本機的模型與地圖，其他電腦需自備相同檔案才能重現。
- 各 JSON 的 `seconds` 與 ms 欄位，是多個程序同時執行時量到的。只供參考，不是效能量測。
