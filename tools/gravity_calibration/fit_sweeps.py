"""将 DM 机械臂 PD 扫描数据拟合到当前 URDF 重力模型。

拟合模型：

    tau_est ~= k * g_urdf(q + offset) + friction * direction + bias
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.kinematics import load_robot_model


def load_model(urdf=None):
    return load_robot_model(urdf)


def gravity(model, data, positions):
    configuration = np.zeros(model.nq)
    configuration[:len(positions)] = positions
    return pin.computeGeneralizedGravity(
        model,
        data,
        configuration,
    )[:len(positions)]


def expand_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    return files


def fit_sweep(document, model, qmin=None, offset_max=0.3):
    joint = document["joint"]
    joint_index = int(joint.removeprefix("joint")) - 1
    samples = [
        sample
        for sample in document["samples"]
        if "q6" in sample
        and (qmin is None or sample["q"] > qmin)
    ]
    if len(samples) < 50:
        raise ValueError("need at least 50 q6-logged samples")

    torques = np.array([
        sample["tau_est"]
        for sample in samples
    ])
    directions = np.array([
        sample.get("direction", sample.get("dir"))
        for sample in samples
    ], dtype=float)
    positions = np.array([
        sample["q6"]
        for sample in samples
    ])

    data = model.createData()
    best = None
    offsets = np.arange(-offset_max, offset_max + 0.01, 0.02)
    for offset in offsets:
        adjusted = positions[::2].copy()
        adjusted[:, joint_index] += offset
        predicted = np.array([
            gravity(model, data, row)[joint_index]
            for row in adjusted
        ])
        matrix = np.column_stack((
            predicted,
            directions[::2],
            np.ones_like(predicted),
        ))
        coefficients, _, _, _ = np.linalg.lstsq(
            matrix,
            torques[::2],
            rcond=None,
        )
        residual = matrix @ coefficients - torques[::2]
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if best is None or rms < best["rms"]:
            best = {
                "joint": joint,
                "samples": len(samples),
                "offset": float(offset),
                "k": float(coefficients[0]),
                "friction": float(coefficients[1]),
                "bias": float(coefficients[2]),
                "rms": rms,
            }
    return best


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fit DM PD sweep JSON files against the configured URDF.",
    )
    parser.add_argument("files", nargs="+", help="JSON paths or glob patterns")
    parser.add_argument(
        "--qmin",
        type=float,
        default=None,
        help="discard samples below this active-joint position",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="optional URDF override; defaults to the current hardware config",
    )
    parser.add_argument(
        "--off-max",
        type=float,
        default=0.3,
        help="zero-offset grid half-width in radians",
    )
    return parser


def main():
    args = build_parser().parse_args()
    model = load_model(args.urdf)
    files = expand_files(args.files)
    if not files:
        raise FileNotFoundError("no sweep files matched")

    print(
        f"{'file':44s} {'n':>5s} {'off':>7s} {'k':>7s} "
        f"{'bias':>8s} {'fric':>8s} {'rms':>8s}"
    )
    for filename in files:
        with open(filename) as source:
            document = json.load(source)
        try:
            result = fit_sweep(
                document,
                model,
                qmin=args.qmin,
                offset_max=args.off_max,
            )
        except ValueError as error:
            print(f"{Path(filename).name:44s} skipped ({error})")
            continue
        print(
            f"{Path(filename).name:44s} "
            f"{result['samples']:5d} "
            f"{result['offset']:+7.3f} "
            f"{result['k']:7.3f} "
            f"{result['bias']:+8.3f} "
            f"{result['friction']:+8.3f} "
            f"{result['rms']:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
