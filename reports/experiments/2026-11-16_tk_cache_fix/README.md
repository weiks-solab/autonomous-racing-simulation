# 修正 TkAgg 快取後再次完整重繪

日期：2026-11-16；分支：`master`；基於 `1955cacea110dc57ef6dfeea293a521e549f1b8e`，含未提交修改。
使用者要求：修正前次定位的 plt.pause／TkAgg 重複 show 問題，延續最小修改原則。

本次正式程式只改 `sim/visualizer.py` 的事件等待：支援 blitting 的快取分支直接呼叫
canvas.start_event_loop，避免 plt.pause 再次呼叫 show；不支援 blitting 時保留原本 pause。
pause_seconds=0 時略過事件等待，避免部分後端將零視為無限等待。
既有 flush_events、快取、縮放、存圖、策略與時間週期保持原樣，沒有更動 main.py。

## 驗證

- 新增一項測試，分別以 0.001 s 與零等待驗證：更新兩幀僅完整重畫一次，沒有呼叫 pause 或 show，正數等待透過 canvas 處理。
- `python -B -m unittest tests.test_visualizer`：16 項通過。
- 直接 `python -B -m unittest`：執行 300 項，2 項失敗，見 [unittest_selected_map.log](unittest_selected_map.log)。目前使用者已選 Oschersleben，舊防守測試假設 21 秒足夠兩圈、舊 GP 測試假設使用內建圖，與目前選圖不符。
- 固定測試場景為原有 rounded_rectangle 後，完整 **300 項通過**，見 [unittest.log](unittest.log)。只在測試程序內 patch，沒有改寫使用者選圖或放寬測試條件。
- 原始退出碼、受控測試狀態及來源雜湊見 [test_status.json](test_status.json)。本次未量測 GUI FPS 或宣稱實際視窗加速倍率。

受控回歸重現命令（專案根目錄、Agg）：

```powershell
$env:MPLBACKEND="Agg"
$env:MPLCONFIGDIR="$PWD\.cache\matplotlib"
@'
import unittest
from unittest.mock import patch
import main
with patch('main.TRACK_NAME', 'rounded_rectangle'):
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests'))
raise SystemExit(not result.wasSuccessful())
'@ | .\.venv\Scripts\python.exe -B -
```

## 真實 TkAgg 驗證限制

`verify_tk.py` 預計用真正 TkAgg canvas 與隱藏視窗重現舊 pause 行為並比較修正，
沒有模擬 GUI 計時。實際執行時，本機 `.venv` 在建立 Tk 視窗便出現
`_tkinter.TclError: Can't find a usable init.tcl`，未能產生 GUI 量測結果。
改用專案內既有 Python 並指定專案內 TCL_LIBRARY／TK_LIBRARY 仍出現相同錯誤。
本次沒有安裝或修改 Tcl／Tk 環境，也沒有人工操作視窗；因此此檢查尚未完成。
目前通過的是程式分支、事件呼叫與既有回歸測試，不能視為真實 GUI 效能驗證。

README、CHANGELOG 與研究摘要已更新。本次小型日誌及程式可納入 Git，尚未 commit／push；
原始地圖與環境仍為本機忽略檔。
