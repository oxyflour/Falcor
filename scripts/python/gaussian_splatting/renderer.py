"""Minimal 3D Gaussian splatting renderer with PLY import support."""

from __future__ import annotations

from dataclasses import dataclass
from math import tan
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class GaussianPoint:
    position: np.ndarray
    color: np.ndarray
    scale: float


def _parse_ply_header(lines: Iterable[str]) -> tuple[int, List[str]]:
    vertex_count = None
    properties: List[str] = []
    for line in lines:
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            _, _, name = line.split(maxsplit=2)
            properties.append(name)
        elif line == "end_header":
            break
    if vertex_count is None:
        raise ValueError("PLY header is missing vertex count.")
    return vertex_count, properties


def load_ply(path: str | Path) -> List[GaussianPoint]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        header_lines = []
        format_line = handle.readline().strip()
        if format_line != "ply":
            raise ValueError("Not a PLY file.")
        header_lines.append(handle.readline())
        if "format ascii" not in header_lines[0]:
            raise ValueError("Only ASCII PLY is supported.")
        for line in handle:
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        vertex_count, properties = _parse_ply_header(header_lines)
        data = [line.strip().split() for line in handle if line.strip()]

    if len(data) < vertex_count:
        raise ValueError("PLY file has fewer vertices than expected.")

    prop_index = {name: idx for idx, name in enumerate(properties)}
    required = ["x", "y", "z"]
    for key in required:
        if key not in prop_index:
            raise ValueError(f"PLY missing required property '{key}'.")

    points: List[GaussianPoint] = []
    for row in data[:vertex_count]:
        pos = np.array([float(row[prop_index["x"]]),
                        float(row[prop_index["y"]]),
                        float(row[prop_index["z"]])], dtype=np.float32)
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if "red" in prop_index and "green" in prop_index and "blue" in prop_index:
            color = np.array(
                [
                    float(row[prop_index["red"]]) / 255.0,
                    float(row[prop_index["green"]]) / 255.0,
                    float(row[prop_index["blue"]]) / 255.0,
                ],
                dtype=np.float32,
            )
        elif "r" in prop_index and "g" in prop_index and "b" in prop_index:
            color = np.array(
                [
                    float(row[prop_index["r"]]) / 255.0,
                    float(row[prop_index["g"]]) / 255.0,
                    float(row[prop_index["b"]]) / 255.0,
                ],
                dtype=np.float32,
            )

        scale = 0.05
        if "scale" in prop_index:
            scale = float(row[prop_index["scale"]])
        elif {"scale_0", "scale_1", "scale_2"}.issubset(prop_index):
            scale = (
                float(row[prop_index["scale_0"]])
                + float(row[prop_index["scale_1"]])
                + float(row[prop_index["scale_2"]])
            ) / 3.0
        points.append(GaussianPoint(position=pos, color=color, scale=scale))

    return points


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_vec = np.cross(right, forward)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up_vec
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def render_points(
    points: Sequence[GaussianPoint],
    width: int,
    height: int,
    fov_degrees: float = 45.0,
    camera_pos: Sequence[float] = (0.0, 0.0, 3.0),
    camera_target: Sequence[float] = (0.0, 0.0, 0.0),
    background: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    eye = np.array(camera_pos, dtype=np.float32)
    target = np.array(camera_target, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view = _look_at(eye, target, up)

    focal = 0.5 * width / tan(np.deg2rad(fov_degrees) * 0.5)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    accum_color = np.zeros((height, width, 3), dtype=np.float32)
    accum_weight = np.zeros((height, width), dtype=np.float32)

    for point in points:
        pos_h = np.array([*point.position, 1.0], dtype=np.float32)
        cam = view @ pos_h
        if cam[2] >= -1e-4:
            continue
        depth = -cam[2]
        x = (cam[0] / depth) * focal + cx
        y = (-cam[1] / depth) * focal + cy

        sigma = max(point.scale * focal / depth, 0.75)
        radius = int(3.0 * sigma)
        min_x = max(int(x - radius), 0)
        max_x = min(int(x + radius), width - 1)
        min_y = max(int(y - radius), 0)
        max_y = min(int(y + radius), height - 1)

        if min_x >= max_x or min_y >= max_y:
            continue

        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        dx = grid_x - x
        dy = grid_y - y
        dist2 = dx * dx + dy * dy
        weight = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)

        accum_color[min_y:max_y + 1, min_x:max_x + 1] += weight[..., None] * point.color
        accum_weight[min_y:max_y + 1, min_x:max_x + 1] += weight

    mask = accum_weight > 1e-6
    output = np.zeros_like(accum_color)
    output[mask] = accum_color[mask] / accum_weight[mask, None]
    output[~mask] = np.array(background, dtype=np.float32)
    return np.clip(output, 0.0, 1.0)


def write_ppm(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    height, width, _ = image.shape
    pixels = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(pixels.tobytes())
