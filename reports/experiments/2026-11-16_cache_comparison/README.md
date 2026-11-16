# 背景快取後仍卡頓的原因追查

> 後續已修正事件等待分支；此頁保留定位當時的事實與數值。目前修正與驗證限制見 [修正紀錄](../2026-11-16_tk_cache_fix/README.md)。

日期：2026-11-16；分支：`master`；基於 `1955cacea110dc57ef6dfeea293a521e549f1b8e`，含未提交修改。
使用者回報：加入快取後，複雜賽道仍比原圓角賽道卡頓。
本次只定位原因，未修改正式程式，未重新執行完整單元測試。

## 快取後的無 GUI 比較

```powershell
$env:MPLBACKEND="Agg"
$env:MPLCONFIGDIR="$PWD\.cache\matplotlib"
.\.venv\Scripts\python.exe -B reports/experiments/2026-11-16_cache_comparison/compare_tracks.py
.\.venv\Scripts\python.exe -B reports/experiments/2026-11-16_cache_comparison/trace_tk_pause.py
```

各案一次、8 秒／400 步，原有雙車設定。Agg 略過 plt.pause，非真實 GUI FPS。

| 案例 | 重新決策平均 ms | 第 2～400 幀繪圖平均 ms | 每步含策略與繪圖平均 ms | 完整重繪次數 |
| --- | --- | --- | --- | --- |
| 圓角矩形，GP 開啟 | 105.62 | 9.13 | 28.65 | 1 |
| 圓角矩形，GP 關閉 | 107.52 | 7.96 | 27.79 | 1 |
| Oschersleben，GP 關閉 | 113.19 | 7.05 | 28.20 | 1 |

完整量測及來源雜湊見 [summary.json](summary.json)。此無 GUI 比較沒有重現新圖大幅變慢；
單純點數 500→652 與 GP 改用幾何預測，仍不足以支持大幅卡頓的結論。
原有決策在同一執行緒計算約 106～113 ms，仍會阻塞顯示，這是額外的既有尖峰。

## 找到的 GUI 路徑缺口

本機 `.venv` 未指定 MPLBACKEND 時，`matplotlib.get_backend()` 回傳 `tkagg`。
讀取已安裝 Matplotlib 3.10.9 的原始碼後確認：

1. `sim/visualizer.py` 快取分支完成 blit 後仍呼叫 `plt.pause()`。
2. `matplotlib/pyplot.py` 的 pause 會呼叫 `show(block=False)`，即使 figure.stale=False。
3. `matplotlib/backends/_backend_tk.py` 的 FigureManagerTk.show 對已開啟視窗呼叫 canvas.draw_idle。
4. 因此 TkAgg 仍會排程完整重畫，PNG 也再次參與渲染；快取不是沒有建立，而是又發生完整重繪。

[trace_tk_pause.py](trace_tk_pause.py) 使用實際 pause 與 TkAgg.show 方法，僅將視窗、canvas 及
事件等待替換為 mock；在 figure.stale=False、視窗已開啟時，確認每次 pause 仍產生一次 draw_idle 請求。
結果見 [tk_pause_trace.json](tk_pause_trace.json)；沒有開啟或人工操作真實 GUI。

前次快取測試將 plt.pause 替換為空操作，因此 8.38 ms 與 100 幀只完整重畫一次，
只適用該無 GUI 量測條件，不能用來保證實際 TkAgg 視窗的改善。
這是前次驗證缺口；目前正式程式尚未修正這條 GUI 路徑。

## Codex 分析／建議，未經使用者確認；待驗證

最小修正可限於視覺化快取分支的事件處理：避免每幀透過 pause 再次 show，
保留必要的 GUI 事件迴圈與等待；不支援 blitting 的後端仍保留原方式。
應再量測真實 TkAgg 的 draw_event 次數與每幀耗時，再判斷是否需要快取幾何或調整顯示頻率。
這些修正尚未執行，不是已確認的後續工作。

本次更新 CHANGELOG 與研究摘要，保留前次數值；程式與量測證據未 commit／push，
原始地圖與本機環境仍為忽略檔。
