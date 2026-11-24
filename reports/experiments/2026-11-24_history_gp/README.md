# 互動歷史 GP：試探反應預測與三組消融

日期：2026-11-24；分支 `master`；基於 `901ae7f`，含未提交修改。

使用者要求：訓練 GP，讓 probe_follow 後車試探 defend 前車並學習其反應，輸入含互動歷史（過去 0.4 s 切兩段的雙車 Δd）。
經討論後，使用者選定：三組消融都訓練、收集時隨機化試探、保留前車曲率；答案改學 Δd（加回 front_d 即位置）列在使用者核准的計畫中。

## 方法

- 特徵與命令見 [GP_TRAINING.md](../../../GP_TRAINING.md#互動歷史-gp試探反應預測)；實作在 [interaction.py](../../../training_data/interaction.py)、[interaction_gp.py](../../../interaction_gp.py)。
- 答案：`d_front(t+0.5) − d_front(t)`，m。歷史四項：雙車的 `d(t−0.2)−d(t−0.4)` 與 `d(t)−d(t−0.2)`，d 左正。
- heading_error = 車頭角 − 所在中心線線段切線角（包裝到 ±π）；speed = 車身速度。
- 消融用同一份資料、同一 episode 切分（seed=42，12／4／4 場）：
  - (a) 基本 8 維：gap、雙車 d／heading_error／speed、front_curvature
  - (b) (a)＋歷史 4 維，共 12 維
  - (c) (b)＋d_target（後車當下實際採用的目標 d），共 13 維
- GP 沿用 StandardScaler＋Constant×ARD RBF＋WhiteKernel，`normalize_y=True`，optimizer_restarts=0；全部 1704 筆訓練樣本參與擬合，驗證／測試只用於評估、不用來調參。

## 收集設定（實際值見 [data_metadata.json](data_metadata.json)）

- 圓角矩形（20×12 m、彎道半徑 3 m、寬 2 m、500 點），步長 0.02 s；20 場各 15 s，seed=42。
- 前車 defend：q_block 15、決策週期 0.05 s、transition 2.0 m、deviation_weight 3.0，避撞責任 overlap_rule。
- 後車 probe_follow：target_gap 2.5 m、決策週期 0.1 s、transition 3.0 m；每場隨機試探序列（先左 6 場、先右 14 場，左右交替，幅度 0.3／0.45／0.6 m），hold 1.0～2.26 s。
- 初距 2～5 m、前車巡航 3～7 m/s、後車 +2 m/s，交替靜止／移動中起步。碰撞矩形 0.6×0.4 m。
- 有效樣本 2840 筆（10 Hz）；排除歷史不足 80、未來不足 100，沒有碰撞排除。
- 後車目標 d 樣本數：0 為 1462，左側（+）591，右側（−）787；左右不均衡。

## 量化結果

| 組 | 輸入數 | 驗證 MAE（m） | 驗證 RMSE（m） | 測試 MAE（m） | 測試 RMSE（m） | 測試 ±2σ 涵蓋 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) 基本 | 8 | 0.0545 | 0.0829 | 0.1016 | 0.1385 | 85.2% |
| (b) ＋歷史 | 12 | 0.0195 | 0.0325 | 0.0375 | 0.0568 | 88.7% |
| (c) ＋歷史＋d_target | 13 | 0.0141 | 0.0231 | 0.0298 | 0.0458 | 83.3% |
| 基準：固定 Δd=0 | - | 0.0792 | 0.1251 | 0.1170 | 0.1733 | - |
| 基準：最近 0.2 s 橫向速度外推 | - | 0.0947 | 0.1395 | 0.1490 | 0.2011 | - |

- 4 場測試的各場 MAE，三組排序都是 c < b < a：(a) 0.067～0.132、(b) 0.029～0.045、(c) 0.023～0.036 m。
- 測試子集 MAE（[summary.json](summary.json)）：後車橫移中（|rear_dd_late|>0.02 m，297 筆）a／b／c = 0.131／0.035／0.028 m；保持中（271 筆）0.070／0.040／0.031 m。
- 誤差 >0.1 m 的比例：a 39.3%、b 9.0%、c 7.2%。|實際 Δd|>0.1 m 的 231 筆中，預測方向相反：a 28、b 8、c 3 筆。

不利結果：

- (a) 測試 MAE 只比固定 Δd=0 好 13%，驗證與測試差近兩倍。
- 三組 ±2σ 涵蓋率都低於 95%，尚未校準。(c) 最大誤差 0.219 m（interaction-007、t=8.5 s：實際 +0.248、預測 +0.028、σ=0.032），約 6.8σ，模型過度自信。
- (b) 有 gap、rear_speed，(c) 有 front_curvature、rear_dd_early 的 length_scale 碰到上界 20；未改界限、未重訓。
- 擬合後的 ARD length_scale 見各組 report.json；輸入彼此相關（例如 heading_error 與 dd_late），不能直接當成特徵重要性。

## 證據

- 各組 [a_basic8/](a_basic8/)、[b_history12/](b_history12/)、[c_action13/](c_action13/)：report.json、驗證／測試 predictions.csv、evaluation.png。
- 彙整：[summarize.py](summarize.py) → [summary.json](summary.json)；日誌 [collect.log](collect.log)、train_*.log；雜湊 [source_hashes.json](source_hashes.json)。
- 原始資料 `data/training/history_gp_20261124/` 與三個 model.pkl（各約 23 MB）沿用 .gitignore，只在本機。

## 驗證

- [相關測試](focused_tests.log) 39 項通過，含新增 9 項：用使用者範例驗證歷史 = [+0.12, 0, 0, +0.10]、不讀未來幀、歷史窗碰撞排除、`--feature-subset`、切線角包裝、自訂試探序列。
- [完整測試](unittest.log) 422 項，3 項失敗，與 11/24 先前紀錄同名（防守被動場景 2 項、相對策略 1 項）。
- 不加 `--randomize-probes` 時 probe_follow 行為不變（`probe_sequence=None` 時序列與原本相同，有測試）。

## 限制

- 單一地圖、單一防守配置、單一 seed 切分；只有 4 場測試。
- 同一個 seed 下先右 14 場、先左 6 場，左右不均衡。
- d_target 是當下採用的目標，後續 0.5 s 內仍可能改變；不能當成「固定某動作」的反事實答案。
- 只量測離線預測誤差，尚未接入規劃器，不能說明試探或超車效果。

Codex 分析，未經使用者確認；待驗證：(a) 已含 heading_error（可推得當下橫向速度），加歷史後仍大幅改善，可能是歷史反映了雙車目前處在橫移的哪個階段；這點尚未用其他方式驗證。
