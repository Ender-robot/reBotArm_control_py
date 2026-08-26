import numpy as np


def compute_scale_factor(values, limits):
    """ 计算保持方向不变的整体限幅系数 """
    values = np.asarray(values, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)

    if values.shape != limits.shape:
        raise ValueError(
            f"values and limits must have the same shape: "
            f"{values.shape} != {limits.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite values")
    if not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
        raise ValueError("limits must contain only finite non-negative values")

    nonzero = np.abs(values) > 0.0
    if not np.any(nonzero):
        return 1.0

    scale = float(np.min(limits[nonzero] / np.abs(values[nonzero])))
    return min(1.0, scale)
