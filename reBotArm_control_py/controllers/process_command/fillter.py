import numpy as np


class EMA():
    """ 关节速度指令的一阶指数平滑滤波器 """

    def __init__(self, alpha):
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0.0, 1.0]")

        self.alpha = float(alpha)
        self.value = None

    def update(self, velocity):
        """ 输入原始速度指令, 返回平滑后的速度指令 """
        velocity = np.asarray(velocity, dtype=np.float64)
        if not np.all(np.isfinite(velocity)):
            raise ValueError("velocity must contain only finite values")

        if self.value is None:
            self.value = velocity.copy()
        else:
            if velocity.shape != self.value.shape:
                raise ValueError(
                    f"velocity shape changed: "
                    f"{self.value.shape} != {velocity.shape}"
                )
            self.value += self.alpha * (velocity - self.value)
        return self.value

    def reset(self):
        """ 清空滤波状态 """
        self.value = None
