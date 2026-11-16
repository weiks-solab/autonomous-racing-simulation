# GP 路線接入進攻：執行驗證

日期：2026-11-16；分支：`master`。
版本：基於 `bf1566e9edc403a4385e930976e723480a6702d4`，含未提交修改；本機 GP 歷史訓練程式版本未記錄。

## 使用者明確說明

將已學到的 s→d GP 應用至進攻車的候選路徑碰撞檢查，保留相同成本函數。
使用者要求簡潔、架構清楚並附簡單邏輯註解，先忽略前述離群等問題。

## 可查證的實作事實

- [gp_prediction.py](../../../strategy/gp_prediction.py)：載入 Pipeline 並檢查模型為 `opponent_s → opponent_current_d`、同時刻答案、horizon=0；查詢不重新標準化或訓練。相同模型檔只載入一次。
- [relative_overtaking.py](../../../strategy/relative_overtaking.py)：每個進攻車各自保存相鄰決策時的對手 s 與時間。用跨圈最短有號 s 變化除以時間差估計縱向速度，下限為 0；第一次以觀測車身速度近似。不讀取對手的策略決策。
- 預測期間假設縱向速度不變：`delta_s(t)=v_s*t`、`s(t)=(s0+delta_s(t)) mod L`、`d(t)=GP(s(t))`；s、d、L 單位為 m，v_s 為 m/s、t 為秒。GP 沒有學習時間或加速度。
- x/y 由賽道座標轉換，航向由 GP 曲線前方 0.05 m 的差分方向取得。t=0 保留觀測狀態，預測縱向速度為零時停留原位；其餘未來點直接使用 GP 平均值。
- 自車候選沿用 Pure Pursuit 與自行車模型；每個對手每輪只預測一次。沿用 [collision_check.py](../../../strategy/collision_check.py) 的共同時間點矩形重疊檢查，碰撞是剔除條件，不是成本懲罰。
- 成本維持 `J=gap_now+delta_s_opponent(T)-delta_s_self(T)`，單位 m，不另加權或正規化，對可行候選取最小。負值表示預測自車領先；同輪對手預測共用，所以剩餘候選與最大自車進度同排序。
- [main.py](../../../main.py) 預設為 vehicle-2 啟用 GP；未指定 GP 的車仍用原本定速、等 d 的自行車預測。資料收集入口未接入 GP，保留原行為。

## 實際設定與環境

完整設定、模型及三個程式檔的 SHA-256 見 [smoke_summary.json](smoke_summary.json)。
本次原有 `.venv` Python 3.10.10、NumPy 2.2.6、scikit-learn 1.7.2；未安裝套件。

- 賽道：20×12 m 圓角矩形中心線跨度，半徑 3 m、路寬 2 m、500 點。
- 兩車均從靜止、d=0 起步；vehicle-1：s=0、relative_overtake、目標 11 m/s（模型上限 10）；vehicle-2：s=4、progress、目標 9 m/s。
- 模擬 21 s、步長 0.02 s。進攻九候選 -0.6～0.6 m，間距 0.15 m；決策週期 0.1 s、預測時長 0.8 s、過渡長度 3 m。
- 對手 progress 五候選，決策週期 0.1 s、預測時長 1.5 s。碰撞矩形 0.6×0.4 m；車身 0.55×0.30 m。
- GP：`models/gp/20261111T064250Z_dbc5bc95/model.pkl`；此模型的歷史訓練設定見 [既有報告](../2026-11-11_baseline/gp_current_sd/report.json)，本次未重新訓練。場景本身沒有隨機初始化。

## 本次量化結果與驗證範圍

- [smoke.log](smoke.log)：完整場景執行成功，1050 步中實際碰撞步數 0，各車軌跡 1051 點。牆鐘時間約 21.11 s，僅為本次執行紀錄，非效能比較。
- [focused_tests.log](focused_tests.log)：44 項通過，包含本次新增 11 項；覆蓋 GP 查詢、曲線航向、跨圈、不同時間經過同一路線、當下重疊、候選碰撞篩選、原成本、獨立歷史與開關。PowerShell 收集 stderr 時附帶 NativeCommandError 包裝，但測試程序 exit code=0，結尾為 OK。
- [full_tests.log](full_tests.log)／[test_status.json](test_status.json)：`Ran 275 tests in 16.816s`，`FAILED (errors=1)`。唯一錯誤為既有 `DefendingIntegrationTests.setUpClass` 的 6 m 橫移過渡超過 4 m 參考路徑，該類別的測試方法未執行；與 11/11 既有日誌為同一初始化問題。
- 視覺化以 Mock 取代以免開視窗；未進行本次即時圖形目視檢查。未執行受控 GP 開／關效果比較、多 seed 成功率或碰撞率實驗，未重新量測 GP 測試誤差。

## 重現與缺少的本機資料

在專案根目錄，以已有環境執行。實際驗證時將環境變數指向專案內：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
$env:TEMP = Join-Path (Get-Location) '.cache/tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m unittest -v tests.test_gp_prediction tests.test_relative_overtaking tests.test_relative_overtaking_integration tests.test_bicycle_prediction tests.test_predicted_collision
.\.venv\Scripts\python.exe reports/experiments/2026-11-16_gp_attack/verify_gp_attack.py
.\.venv\Scripts\python.exe -m unittest -v
```

完整測試實際由同一 Python 的 subprocess 執行上述 unittest 命令，將 stdout/stderr 保存為 UTF-8。
場景腳本會重新寫入本資料夾的 smoke.log 與 smoke_summary.json，執行前請保留要比較的舊結果。
原模型與 `data/training/20261111T063545Z_f6eb6a97/` 原始資料仍在本機，未複製或解除忽略。
Git 內摘要、既有圖表及逐筆測試預測不足以重訓相同模型；完整重現需提供 SHA-256 相同的模型。
沒有模型時可關閉 GP 執行基本模擬，但那不算本次 GP 場景的重現。

## 限制

本次結果只支持模型已接入、程式可執行及此單一場景未偵測到重疊；不能支持超車成功率或安全性改善。
沒有加入離群補償、平滑接續、標準差碰撞膨脹或預測異常回退，第一個未來點可能不連續。
縱向估速假設兩次觀測間前進不足半圈；定速與 GP 路線可能偏離真實行駛。
碰撞檢查只在取樣時間點執行；全部候選剔除時仍沿用原路徑，不能保證無碰撞。
本次未新增使用者未決定的研究下一步。
