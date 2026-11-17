# 速度要求與離散速度候選驗證

日期為 2026-11-17，分支為 `master`。
基於 `6dc4a2b9cb76c5f83810efadeeb3a0eca0d75968`，含未提交修改。
使用者明確要求完成策略速度上限資料流、共用縱向控制，以及主要進攻策略的少量速度候選。

## 實作與比較方式

`StrategyDecision`、`RelativeCandidate`、`CandidatePath` 保存可選的 `speed_limit`。
自車 rollout 與實際模擬都呼叫 `PurePursuitController.compute_command()`，再由
`KinematicBicycleModel.step()` 更新狀態。比例控制集中在 `compute_speed_acceleration()`。

有效目標為 `v_ref = min(巡航目標, speed_limit)`，省略上限則使用原巡航目標。
加速度為 `clip(speed_gain * (v_ref - v), -max_deceleration, max_acceleration)`。
加減速限制的單位為 m/s²，兩個參數均為正值，舊設定省略減速度上限時使用原加速度上限。
比例控制更新速度，原有巡航設定保留供後續取消限制使用。

速度候選開關預設關閉。啟用時，原有橫向候選使用巡航目標，另在目前追蹤的路徑加入
前車速度與 `max(前車速度 - speed_candidate_delta, 0)`，僅保留低於巡航目標的互異值。
進行中的 maneuver 保留原橫移起點，每輪對手預測仍由所有候選共用。

每個候選皆執行既有同時間車身矩形碰撞檢查，碰撞候選會被剔除。
其餘候選以原成本 `J = 目前間距 + 對手累積進度 - 自車累積進度` 取最小值，單位為 m。
全部被剔除時仍記錄 `no_feasible_path`，持有上一個路徑與速度要求。
`constant_speed` 對手繼續使用零加速度，GP 對手模型沿用原有假設。

## 固定直線案例的量化結果

封閉方形賽道的第一段直線長 100 m，路寬 2 m。
自車從 `(x, y, yaw, v) = (20, 0, 0, 6)` 起步，前車為 `(22.5, 0, 0, 3)`，
巡航目標 6 m/s、比例增益 1.5 s⁻¹、加減速上限各 3 m/s²。
車身尺寸 0.55 × 0.30 m、後軸至中心 0.12 m，碰撞矩形沿用車身尺寸。
預測視野 1 s、dt 0.02 s、橫向候選僅 `d=0`，低速差值 1 m/s。
前車採固定 3 m/s、等 d 的 bicycle 預測。

| 速度要求 | 同視野碰撞檢查 | 終點相對間距成本 | 終點車速 |
| --- | --- | --- | --- |
| `None`，巡航 6 m/s | 0.66 s 預測到重疊 | 已剔除 | 6.0000 m/s |
| 3 m/s | 視野內可行 | 0.8063 m | 3.7247 m/s |
| 2 m/s | 視野內可行 | 0.9492 m | 3.2039 m/s |

策略選擇 3 m/s 候選，實際第一步輸出 −3 m/s²，車速由 6 降至 5.94 m/s。
將此決策固定，透過正式 `main.py`、`StrategyManager`、`SimulatedVehicle.step()` 執行 50 步，
包含初始值共 51 個狀態的 x、y、yaw、velocity 與候選預測相同。
關閉速度候選後，此中心路徑的巡航候選仍被剔除，模式仍為 `no_feasible_path`。

另以直線及橢圓彎道，分別測試固定 `None`、2 m/s、0 m/s 要求，共六組、每組 51 個狀態。
逐步比較四個狀態量的絕對誤差，最大為 `8.89e-16` 以下，測試容許誤差為 `1e-10`。
從 6 m/s 使用零速度要求 12 s 後，速度為 `1.76197e-7 m/s`，低於 `1e-6 m/s` 停止容許誤差。
再取消限制 6 s 後恢復至 `5.99836 m/s`，巡航設定仍為 6 m/s。
各參數與原始量化摘要保存於 [summary.json](summary.json)。

## 實際驗證

使用專案原有 `.venv`，Python 3.10.20、NumPy 2.2.6、Matplotlib 3.10.9，渲染後端為 Agg。
驗證使用固定測試場景與一次預設模擬執行。

```powershell
$env:MPLBACKEND='Agg'
$env:MPLCONFIGDIR=Join-Path (Get-Location) '.cache/matplotlib'
$env:TEMP=Join-Path (Get-Location) '.cache'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.test_speed_requests -v
.\.venv\Scripts\python.exe -m unittest
```

- 新增 17 項測試通過，涵蓋向後相容、逐步加減速、停止與恢復、輸入驗證、
  同路徑速度候選選取、去重、前車條件、保留橫移起點、全部剔除與資料收集入口。
- 修改前直接跑完整 300 項測試，有兩項既有失敗，分別是防守兩圈測試讀到 Oschersleben，
  以及 GP 載入測試在外部圖預期啟用舊 GP。本次將兩項測試固定在其原本的圓角場景。
- 修改後完整 317 項通過，正式選圖仍為 Oschersleben。完整輸出見 [unittest.log](unittest.log)。
- 實際執行原有 `main.py`，本機預設雙車 21 s 場景以 Agg 完成，exit code 為 0。
  此次速度候選開關為 False，目標速度為使用者原工作區的 20／18 m/s，模型速度上限為 30 m/s。
  輸出回報未偵測到實際碰撞，完整輸出與 Agg 的非互動顯示提示見 [main_agg.log](main_agg.log)。

重現摘要、完整測試與預設模擬並更新此目錄的輸出，可執行以下指令。

```powershell
.\.venv\Scripts\python.exe reports/experiments/2026-11-17_speed_requests/verify.py
```

## 驗證範圍與交接

上述資料支持速度要求確實傳到加減速控制，以及固定決策下的逐步預測／執行一致性。
低速候選的可行性只涵蓋本次 1 s 視野，沒有驗證安全煞停或自動避碰保證。
緊急煞車與完整 fallback 尚未實作，互動 GUI 及啟用速度候選的長時間重規劃效果尚未驗證。

固定案例與新增測試可使用程式內賽道重現。
預設 `main.py` 執行仍需要 Git 忽略的本機外部地圖，完整 GP 模型、原始訓練資料與環境也仍在本機。
本次程式、測試及此目錄的小型證據尚未 commit／push，GitHub 讀取者暫時看不到這批修改。

## 11/17 追加的雙車閉迴路驗證

本次依使用者要求新增 1 s 雙車閉迴路驗證，版本同為基於
`6dc4a2b9cb76c5f83810efadeeb3a0eca0d75968` 的 `master` 未提交工作區。
重用 `SpeedCandidateIntegrationTests` 的直線賽道與配置，自車初速 6 m/s、
前車初速及巡航目標均為 3 m/s，初始車距 2.5 m。
驗證配置的橫向候選僅為 `d=0`，速度候選開啟，預測視野 1 s，速度差 1 m/s。

實際執行 `run_episode()` 的雙車迴圈，每步將更新後的觀測送入真實策略。
記錄包裝呼叫原始 `update()` 與 `SimulatedVehicle.step()` 並原樣回傳結果，速度要求由策略選擇產生。
前車使用初速等於巡航目標的正常控制，量測的 51 個前車速度皆為 3 m/s。

本次實際命令如下，執行一次新增回歸測試並保存 JSON 與測試日誌。

```powershell
.\.venv\Scripts\python.exe reports/experiments/2026-11-17_speed_requests/verify_closed_loop.py
```

| 項目 | 實測結果 |
| --- | --- |
| 模擬時間與步數 | 1.0 s，50 步，dt 0.02 s |
| 初始候選檢查 | 巡航候選預測碰撞，3 與 2 m/s 候選可行 |
| 候選評估次數 | 9 次，含 t=0 初始規劃與後續 8 次重規劃 |
| 設定的決策週期 | 原案例的 0.1 s |
| 首次選中的速度要求 | t=0，第 0 步，3.0 m/s |
| 該步實際加速度 | −3.0 m/s² |
| 該步實際車速 | 6.0 → 5.94 m/s |
| 首次後續重規劃 | t=0.1 s，第 5 步，仍選中 3.0 m/s |
| 該次後續決策的控制結果 | −3.0 m/s²，5.70 → 5.64 m/s |
| 結束時自車速度 | 3.51204 m/s |
| 車身碰撞結果 | 51 個狀態檢查含初始狀態，碰撞為 0 |
| 新增回歸測試 | 1 項通過，exit code 0 |

實際評估時刻為 0、0.10、0.20、0.32、0.44、0.54、0.66、0.78、0.90 s。
後續重規劃曾選擇 2 m/s 及 3 m/s 候選，逐步決策、加速度、減速前後車速及碰撞紀錄
保存於 [closed_loop_summary.json](closed_loop_summary.json)，測試輸出見
[closed_loop_unittest.log](closed_loop_unittest.log)。

三項驗收均通過，結果涵蓋此單一 1 s 雙車場景。
本次只執行新增的閉迴路測試，前文 317 項完整測試與固定決策結果為前次紀錄。
本次新增驗證與小型證據尚未 commit／push。
