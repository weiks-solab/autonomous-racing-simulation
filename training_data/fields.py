"""修改輸入、輸出時，通常只需要編輯這個檔案。"""

from core.track_geometry import signed_longitudinal_gap, track_length
from training_data.extractors import FeatureContext, InputField, OutputField


# ===== 1. 選擇這次要收集的欄位；名稱必須在下方註冊 =====
SELECTED_INPUTS = ["opponent_s"]
SELECTED_OUTPUTS = ["opponent_current_d"]

# 舊的未來位移問題可改用這組，時間預設自動選 1 步：
# SELECTED_INPUTS = ["relative_s", "relative_d", "relative_speed", "opponent_d"]
# SELECTED_OUTPUTS = ["opponent_delta_d"]


def relative_s(context: FeatureContext) -> float:
    """我方 s 減對手 s；封閉賽道取最短有號距離。"""
    ego = context.ego.track_position
    opponent = context.opponent.track_position
    if context.track.closed:
        return signed_longitudinal_gap(opponent, ego, track_length(context.track))
    return ego.s - opponent.s


# ===== 2. 註冊可選輸入：名稱、單位、說明、計算方式 =====
# c.ego / c.opponent：當下的我方 / 對手，輸入不提供未來資料。
# 常用位置：.track_position.s、.track_position.d
# 常用狀態：.state.x、.state.y、.state.yaw、.state.velocity
INPUT_FIELDS = (
    InputField("opponent_s", "m", "對手當下的賽道 s，封閉賽道每圈繞回。",
               lambda c: c.opponent.track_position.s),
    InputField("opponent_d", "m", "對手當下的賽道 d，左正右負。",
               lambda c: c.opponent.track_position.d),
    InputField("opponent_speed", "m/s", "對手當下的車身速度。",
               lambda c: c.opponent.state.velocity),
    InputField("ego_speed", "m/s", "我方當下的車身速度。",
               lambda c: c.ego.state.velocity),
    InputField("relative_s", "m", "我方 s 減對手 s，封閉賽道取最短有號距離。", relative_s),
    InputField("relative_d", "m", "我方 d 減對手 d，正值表示我方在對手左側。",
               lambda c: c.ego.track_position.d - c.opponent.track_position.d),
    InputField("relative_speed", "m/s", "我方車身速度減對手車身速度。",
               lambda c: c.ego.state.velocity - c.opponent.state.velocity),
    # 新增輸入範例：取消下方註解，再將 "opponent_yaw" 加到 SELECTED_INPUTS。
    # InputField("opponent_yaw", "rad", "對手當下的世界座標航向。",
    #            lambda c: c.opponent.state.yaw),
)


# ===== 3. 註冊可選答案，並指定時間 =====
# timing="current"：與輸入同時刻，程式自動使用 horizon=0。
# timing="future"：使用未來資料，預設 horizon=1，可由 --horizon-steps 調整。
# c.ego / c.opponent 是當下；c.future_ego / c.future_opponent 是未來。
# 同一次收集的答案必須使用相同 timing；新增後記得修改 SELECTED_OUTPUTS。
OUTPUT_FIELDS = (
    OutputField("opponent_current_d", "m", "與輸入同一時刻的對手 d。",
                lambda c: c.opponent.track_position.d, timing="current"),
    OutputField("opponent_delta_d", "m", "對手未來 d 減當下 d。",
                lambda c: c.future_opponent.track_position.d - c.opponent.track_position.d,
                timing="future"),
    OutputField("opponent_future_d", "m", "對手在指定未來時間的絕對 d。",
                lambda c: c.future_opponent.track_position.d, timing="future"),
    OutputField("opponent_delta_speed", "m/s", "對手未來速度減當下速度。",
                lambda c: c.future_opponent.state.velocity - c.opponent.state.velocity,
                timing="future"),
    # 新增當下答案範例：取消下方註解，再修改 SELECTED_OUTPUTS。
    # OutputField("opponent_current_speed", "m/s", "與輸入同一時刻的對手速度。",
    #             lambda c: c.opponent.state.velocity, timing="current"),
)
