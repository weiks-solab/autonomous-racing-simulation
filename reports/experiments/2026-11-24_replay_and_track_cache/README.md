# 2026-11-24 平順播放（--replay）與賽道幾何快取

日期：2026-11-24；分支 `master`；基於 `bbd981f`，含未提交修改（含同日的架構整理）。

## 目的

使用者要求找出讓模擬畫面不卡的方法，並繼續先前規劃但未實行的項目。
本次修改都不改變模擬結果；修改內容見 [CHANGELOG](../../../CHANGELOG.md)。

## 卡頓原因（修改前實測）

- 預設場景（relative_overtake 對 defend），以 `--realtime` 路徑在 Agg 下模擬 6 s：
  - 每 3 步重新決策一次，該步策略計算平均 213.8 ms（p95 251.5 ms）；其餘 200 步約 0.03 ms。
  - 每幀繪圖約 9.7 ms。
  - 整體比實際時間慢 4.1 倍。
- 結論：卡頓來自同一執行緒邊算邊畫，不是繪圖；11/16 已做過背景快取與 TkAgg 修正。
- Windows＋Python 3.10 的計時器刻度約 15.6 ms。以隱藏的 Tk 視窗量測：

  | Tk 要求 | 預設計時器 | `timeBeginPeriod(1)` |
  | --- | --- | --- |
  | `after(5)` | 15.33 ms | 5.23 ms |
  | `after(20)` | 31.02 ms | 20.16 ms |

  這次量測沒有存成檔案。

## 結果

**賽道幾何快取**
- 以 [measure_decision_time.py](measure_decision_time.py) 量預設場景 6 s，各 2 次：

  | 項目 | 快取前 | 快取後 |
  | --- | --- | --- |
  | 進攻每次決策平均 | 162.6 ms | 106.9 ms |
  | 防守每次決策平均 | 47.0 ms | 31.0 ms |
  | 總耗時 | 21.1 s | 13.9 s |

  （[前](decision_time_before.json)、[後](decision_time_after_cache.json)）
- 快取前的 162.6 ms 低於上方 213.8 ms，是因為量法不同：此處只計策略的 `update`，上方是 `StrategyManager.update` 加總且含即時繪圖。

**播放節奏**
- 以 [measure_frame_pacing.py](measure_frame_pacing.py) 量模擬 4 s，Agg，GUI 等待以精準忙等模擬（[frame_pacing.json](frame_pacing.json)）：

  | 模式 | 播放耗時 | 幀間隔中位數 | p95 | 超過 50 ms |
  | --- | --- | --- | --- | --- |
  | `--realtime` | 12.67 s | 12.9 ms | 170.9 ms | 33.7% |
  | `--replay` | 3.98 s（另先計算約 10.4 s） | 20.0 ms | 20.0 ms | 0.5%（開頭 1 次完整繪製） |

**結果不變的驗證**
- 以 [verify_equivalence.py](../2026-11-24_refactor_equivalence/verify_equivalence.py) 比對 11 個場景，加入快取後與加入播放後，都與同日架構整理後的結果逐位元相同。
  （[快取後](equivalence_after_cache.json)、[播放後](equivalence_after_replay.json)）
- 全套測試（Agg）409 項，含新增 8 項；3 項失敗，與先前同名（[unittest.log](unittest.log)）。
- 新增測試涵蓋：
  - 假時鐘下的跳幀、等待與倍率。
  - `--replay` 最後一幀的畫面資料與一般執行相同，終端摘要相同（只多計算進度）。
  - 隱藏候選時不保存快照。
  - Windows 計時器的設定與恢復。
  - `wait()` 的分支。
- 命令列：`--speed 0` 會報錯；`--replay --track oval --seconds 0.4 --speed 2` 可跑完；未知地圖名稱會列出可用地圖。

## 限制

- 播放節奏是在 Agg 下以忙等模擬 GUI 等待量得的，不含 TkAgg 的畫面傳輸與 Tk 本身的計時誤差。實際視窗的觀感尚未目視確認。
- `--replay` 要等整場計算完才開始播放；預設 21 s 場景依上表約需 50 s。
- 播放期間的 `timeBeginPeriod(1)` 會暫時提高 Windows 系統計時器精度，結束時恢復。
- 時間量測都是單機、單一場景、各 1～2 次的結果，會受機器負載影響。
