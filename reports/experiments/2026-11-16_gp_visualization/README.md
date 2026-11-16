# GP 整圈路線與即時預測顯示

日期：2026-11-16；分支：`master`；基於 `bf1566e9edc403a4385e930976e723480a6702d4`，含未提交修改。
使用者明確要求離線整圈 GP 路線圖及即時對手 GP 預測視覺化；本次沒有修改策略成本或車輛控制。

## 已實作與實際輸出

- [整圈圖](gp_route.png)：沿目前賽道的 s 等距取樣 1000 點，直接查 GP 平均 d，再轉成 x/y；左圖為整圈路線、右圖為 s→d。紫色圓點表示 s=0 的 GP 位置，不是實際車輛初始位置。
- [取樣 CSV](gp_route.csv)：1000 筆 s/d/x/y，單位皆 m；不含重複的一圈終點。[設定 JSON](gp_route.json) 保存原模型路徑與 SHA-256、500 點賽道中心線、路寬及封閉屬性。
- [即時預覽](gp_live_preview.png)：進攻車實際用於碰撞檢查的對手 GP 預測 states/times，透過共用快照單向送到視覺化，不重新查 GP。紫色點虛線為預測，圓點為終點；標籤包含觀測者、對手、評估時間與時長。
- 即時圖元以 `(觀測者 ID, 對手 ID)` 區分並重用；未提供快照或關閉顯示時隱藏舊曲線。`show_gp_predictions`、`show_gp_prediction_labels` 與候選路徑的開關獨立。

## 驗證與範圍

- [相關測試](focused_tests.log)：34 項通過，含新增 6 項，涵蓋 GP 預測與碰撞用資料一致、時間／ID 標籤、圖元重用、隱藏、不同觀測者、整圈取樣與 PNG／CSV／JSON 輸出。此紀錄來自標籤位置調整前；調整後另執行下列完整測試。
- [完整測試](full_tests.log)、[程序狀態](test_status.json)：`Ran 281 tests in 17.436s`、`FAILED (errors=1)`。唯一錯誤仍為既有防守整合類別初始化的 6 m 過渡長度超過 4 m 參考路徑；該類別方法未執行。
- 使用原有模型完成離線輸出，並以 Agg 後端走 `main.run_simulation(realtime=True)` 的逐步更新流程；測試專用設定只將時長縮為 2 秒、關閉 GUI 暫停，未改預測或選路。共更新 100 幀，最後一份 GP 快照於 1.88 s 評估，向前 0.8 s、41 點；見 [摘要與程式雜湊](live_summary.json)、[場景日誌](live.log)。
- 兩張 PNG 已目視檢查，圖例、路線與標籤可讀。未人工操作互動 GUI、未量測即時顯示效能或策略改善；沒有重新訓練 GP。

## 實際設定與重現

Python 3.10.10，沿用原有 `.venv` 及套件；原場景設定見 [GP 接入設定](../2026-11-16_gp_attack/smoke_summary.json)。
賽道為 20×12 m 圓角矩形、半徑 3 m、寬 2 m；離線指令使用目前 main.py 賽道，沒有自動從訓練 metadata 換圖。
本次使用與接入驗證相同的 `models/gp/20261111T064250Z_dbc5bc95/model.pkl`，完整模型仍只在本機且被 Git 忽略。

在專案根目錄使用已有環境；驗證時 `MPLBACKEND=Agg`、`MPLCONFIGDIR=.cache/matplotlib`、`TEMP/TMP=.cache/tmp`、`PYTHONUTF8=1`：

```powershell
.\.venv\Scripts\python.exe -m unittest -v tests.test_gp_visualization tests.test_gp_prediction tests.test_visualizer tests.test_candidate_snapshots
.\.venv\Scripts\python.exe reports/experiments/2026-11-16_gp_visualization/verify_visualization.py
.\.venv\Scripts\python.exe -m unittest -v
```

完整測試實際由相同 Python 的 subprocess 執行上述命令，保存 UTF-8 日誌。
驗證腳本呼叫 `plot_gp_route.main` 指定此證據目錄的 PNG，並重新輸出兩張圖與摘要；再次執行會覆寫本目錄同名圖片與資料。
平常只輸出整圈圖可用 `python plot_gp_route.py`，實際互動觀看可用 `python main.py --realtime`。

## 結果解讀與交接限制

整圈圖是 GP 平均路線，並非時間預測或實際車輛軌跡；地圖接回起點只為繪圖，未修改 GP 的跨圈連續性。
即時圖是最近決策時刻的快照，兩次決策間保持不變；它包括 t=0 的觀測位置，其後才使用 GP 預測。
目前只畫平均值，沒有新增標準差區域、離群補償或平滑接續，也不代表安全界線。
PNG、CSV、JSON 與日誌可納入 Git；完整重現仍須取得相同模型，或另外提供相容模型並清楚區分新結果。
本目錄圖表、CSV／JSON 與日誌隨程式一併納入本次提交；完整模型仍維持只在本機。
推送整理僅更新文字與收錄已有輸出，沒有重新執行本頁列出的命令或產圖；上方結果皆為先前紀錄。
