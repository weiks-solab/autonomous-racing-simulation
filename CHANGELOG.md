# 變更紀錄

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
