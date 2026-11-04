import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)

# --------------------------------------------------
# 1. 準備示範資料
# --------------------------------------------------
rng = np.random.default_rng(42)

# 輸入：我方相對對手的橫向位置，單位 m
# 正值代表我方位於對手左側，負值代表右側
X_train = rng.uniform(-0.8, 0.8, size=(40, 1))

# 答案：對手在接下來 0.1 秒內的橫向位移，單位 m
# 人工設定：我方偏左，對手也傾向往左阻擋
def demo_response(relative_d):
    return 0.05 * np.tanh(3.0 * relative_d)

y_train = (
    demo_response(X_train[:, 0])
    + rng.normal(0.0, 0.005, size=40)
)

print("輸入資料形狀：", X_train.shape)  # (40, 1)
print("答案資料形狀：", y_train.shape)  # (40,)

# --------------------------------------------------
# 2. 建立並訓練 GP
# --------------------------------------------------
kernel = (
    ConstantKernel(1.0, (0.01, 100.0))
    * RBF(length_scale=0.3, length_scale_bounds=(0.05, 3.0))
    + WhiteKernel(
        noise_level=0.02,
        noise_level_bounds=(1e-5, 1.0),
    )
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=2,
    random_state=42,
)

gp.fit(X_train, y_train)
print("訓練完成！")
print("學到的 kernel：", gp.kernel_)

# --------------------------------------------------
# 3. 用新輸入進行預測
# --------------------------------------------------
X_test = np.linspace(-1.2, 1.2, 200).reshape(-1, 1)
mean, std = gp.predict(X_test, return_std=True)

# 問模型：我方位於對手左側 0.3 m 時，對手會怎麼移動？
query = np.array([[0.3]])
query_mean, query_std = gp.predict(query, return_std=True)

print(
    f"我方偏左 0.3 m 時，"
    f"預測對手下一步橫向位移 = {query_mean[0]:.4f} m，"
    f"標準差 = {query_std[0]:.4f} m"
)

# --------------------------------------------------
# 4. 畫出資料、預測與不確定性
# --------------------------------------------------
x = X_test[:, 0]

plt.scatter(
    X_train[:, 0], y_train,
    color="black", s=25, label="Training observations",
)
plt.plot(
    x, demo_response(x),
    "--", color="gray", label="Demo ground truth",
)
plt.plot(x, mean, color="tab:blue", label="GP prediction")
plt.fill_between(
    x,
    mean - 2 * std,
    mean + 2 * std,
    color="tab:blue",
    alpha=0.2,
    label="Mean +/- 2 std",
)

plt.xlabel("Relative lateral position (m)")
plt.ylabel("Opponent lateral change over 0.1 s (m)")
plt.title("GP regression demo")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()