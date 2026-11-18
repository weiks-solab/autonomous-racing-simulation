# 額外低速候選

日期：2026-11-18；分支 master，基於 `de3581a06604976980aa6efa36c1cf61b84fa84c`，含未提交修改。
使用者希望增加前車 −4 m/s 的選項，觀察是否能增加可行解。

## 修改

`RelativeOvertakingConfig.extra_speed_candidate_deltas` 預設為空 tuple，保留舊設定行為。
main 明確加入 `(4.0,)`；速度要求為前車速度、前車 −1、前車 −4，最低截為 0，去除重複值，並排除不低於巡航目標的項目。
只在目前路徑增加低速候選，不展開所有路徑與速度的組合。
原本成本、碰撞檢查、對手預測、加減速度限制及 no_feasible_path 行為未改。

## 實際驗證

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_speed_requests
.\.venv\Scripts\python.exe -m unittest
.\.venv\Scripts\python.exe reports/experiments/2026-11-18_extra_speed_candidates/verify.py
```

測試以 Agg 執行，MPLCONFIGDIR、TEMP、TMP 放 repo 的 .cache。
新增四項測試，涵蓋去重與上下限、非法數值、額外候選確實可行且執行一致、已重疊時仍不可行。
既有路徑起點測試也加入額外候選，確認更換速度不重啟橫移。

直線案例：後車 s=20 m、11 m/s，前車 s=21.3 m、9 m/s；碰撞矩形 0.8×0.7 m，
控制 gain=1.5、最大加減速度各 5 m/s²，dt=0.02 s、視野 1 s、僅中心線候選。
巡航、9、8 m/s 候選均被碰撞檢查剔除，加入 −4 後選中 5 m/s。
實際首步加速度 −5 m/s²，車速 11→10.9 m/s，沒有瞬間降至 5 m/s；
固定決策由 main→管理器→控制器逐步執行，x、y、yaw、velocity 與 rollout 在 1e-10 絕對誤差內一致。

第一輪完整測試有一項舊案例假定僅九條橫向候選，與使用者已啟用速度候選的 main 配置不符。
該案例改為明確停用速度候選，保留九候選 baseline 檢查，未修改正式設定，原失敗日誌見 [before_fix_unittest.log](before_fix_unittest.log)。
修正後完整 346 項通過，38.837 s；速度要求相關 22 項也通過。

完整場景比較使用當下 main：圓角矩形、初距 4 m、巡航 11／9 m/s、視野 1 s、配置週期 0.05 s、dt=0.02 s。
碰撞矩形當時已改回 0.6×0.4 m；這與先前 0.8×0.7 m、0.1 s 週期的診斷不同，不能直接比較歷史結果。
21 s 兩組各規劃 350 次，均未選低速、未出現 no_feasible_path、未偵測碰撞。
因此這趟完整場景未顯示額外候選改善效果，新增解的證據僅來自上述直線測試。
設定與結果見 [summary.json](summary.json)，完整測試日誌見 [unittest.log](unittest.log)。

較低目標速度不保證能及時減速；目前已重疊、對手橫移預測誤差及所有候選不可行後沿用舊決策的限制仍存在。
沒有新增跟車狀態或安全 fallback；本次程式與證據尚未 commit／push。

## 後續畫面標示檢查

11/18 僅改 visualizer：實體碰撞標紅色 COLLISION，判定矩形重疊標橘色 ENVELOPE OVERLAP，規劃規則不變。
18 項視覺化測試通過，另以 Agg 檢視橘色標示；互動 GUI 未人工驗證。
當下 main 碰撞尺寸已為 0.6×0.8 m，全套 348 項有兩個舊場景測試失敗（[日誌](collision_display_unittest.log)）；
使用 HEAD 的原視覺化在記憶體中重跑，也得到相同兩項失敗（[baseline](collision_display_baseline.log)）。
未修改使用者場景或演算法，本次標示修改尚未 commit／push。
