#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO 数据集离线数据增强 / 数据集扩充工具
================================================

支持三种 YOLO 任务，自动保持标注与图像严格对齐：

    detection : 标签 `cls xc yc w h`              (归一化 xywh，检测框)
    segment   : 标签 `cls x1 y1 x2 y2 ...`        (归一化多边形，实例分割)
    obb       : 标签 `cls x1 y1 ... x4 y4`        (归一化 8 点，旋转框)

只增强 train 划分，val 原样复制（绝对不要增强验证集，否则指标虚高）。

快速示例
--------
    # 检测数据集：在 datasets/U2 的基础上，每张图再生成 2 张增强图
    python augment_dataset.py --src datasets/U2 --task detection --times 2

    # 分割数据集
    python augment_dataset.py --src datasets/U1_fenge_jin --task segment --times 2

    # OBB 数据集
    python augment_dataset.py --src datasets/U1_obb_jin --task obb --times 2

    # 先小规模试跑 + 生成可视化预览图，确认标注没有跑偏
    python augment_dataset.py --src datasets/U2 --task detection --times 1 \
        --limit 20 --out datasets/_preview --preview 12 --overwrite

生成结果
--------
    <out>/images/train  <out>/labels/train     # 原图(可选) + 增强图
    <out>/images/val    <out>/labels/val       # 原样复制
    <out>/data.yaml                            # 指向新目录，直接拿去 train.py

注意
----
* 几何增强（旋转/缩放/平移/剪切/透视/翻转/Mosaic）会重算标注；
  颜色增强（HSV/亮度对比度/模糊/噪声）不改变标注。
* 擦除(Cutout)只会擦掉不覆盖任何目标的区域，不会让标注失效。
* 复制粘贴(copy-paste)只对 segment 数据集开放（需要多边形做掩码）。
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Windows 控制台默认 GBK，中文日志会乱码，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 本脚本可以被放在任意位置（例如单独放进 augment_dataset/ 子目录）。
# 相对路径按"当前目录 -> 脚本目录 -> 脚本父目录 -> 祖父目录"依次尝试，
# 这样 `--src datasets/U2` 在 ultralytics 根目录下运行也依然有效。
_HERE = Path(__file__).resolve().parent
SCRIPT_ANCHORS: List[Path] = [_HERE, _HERE.parent, _HERE.parent.parent]


# --------------------------------------------------------------------------------------
# 基础 IO（兼容中文路径：cv2.imread 读不了中文路径，必须 imdecode）
# --------------------------------------------------------------------------------------
def imread_unicode(path: Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取图片: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 95) -> None:
    ext = path.suffix.lower() if path.suffix else ".jpg"
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------------------
# 标注模型：统一用"像素坐标多边形"表示，按 kind 决定如何回写
# --------------------------------------------------------------------------------------
@dataclass
class Instance:
    cls: int
    kind: str          # "det" | "seg" | "obb" | "pose"
    pts: np.ndarray    # (N, 2) float32，像素坐标（对象轮廓；pose 为外接框 4 点）
    # --- 仅 task=pose 使用 ---
    kpts: Optional[np.ndarray] = None   # (K, 2) float32，像素坐标关键点
    kvis: Optional[np.ndarray] = None   # (K,) float32，可见性 0/1/2（标签里可能不写，则 None）


def instance_bbox(it: "Instance") -> Tuple[float, float, float, float]:
    """取实例的包围盒（像素）。pose 的外接框由关键点决定时以框为准。"""
    return bbox_of(it.pts)


def instance_outside(it: "Instance", w: float, h: float) -> bool:
    """实例的中心点是否已完全跑出画面（用于丢弃)。"""
    x1, y1, x2, y2 = bbox_of(it.pts)
    return not (x2 > 0 and y2 > 0 and x1 < w and y1 < h)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def bbox_of(poly: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
    return x1, y1, x2, y2


def ioa(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    """交并比里的 IoA：交集面积 / 前者面积（判断两个框是否重叠）。"""
    if boxes.size == 0:
        return np.zeros(0, np.float32)
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return np.zeros(len(boxes), np.float32)
    ix1 = np.maximum(boxes[:, 0], x1)
    iy1 = np.maximum(boxes[:, 1], y1)
    ix2 = np.minimum(boxes[:, 2], x2)
    iy2 = np.minimum(boxes[:, 3], y2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / area


def clip_polygon(poly: np.ndarray, w: float, h: float) -> np.ndarray:
    """Sutherland-Hodgman：把多边形裁剪到 [0,w]x[0,h] 画面内。"""
    if len(poly) < 3:
        return np.empty((0, 2), np.float32)
    pts = poly.astype(np.float64)
    # (轴, 边界值, 保留哪一侧)
    for axis, val, keep_ge in ((0, 0.0, True), (0, float(w), False), (1, 0.0, True), (1, float(h), False)):
        if len(pts) < 3:
            return np.empty((0, 2), np.float32)
        out: List[np.ndarray] = []
        n = len(pts)
        for i in range(n):
            cur, nxt = pts[i], pts[(i + 1) % n]
            cin = cur[axis] >= val if keep_ge else cur[axis] <= val
            nin = nxt[axis] >= val if keep_ge else nxt[axis] <= val
            if cin:
                out.append(cur)
            if cin != nin:
                t = (val - cur[axis]) / (nxt[axis] - cur[axis] + 1e-12)
                out.append(cur + t * (nxt - cur))
        pts = np.array(out) if out else np.empty((0, 2), np.float64)
        if len(pts) < 3:
            return np.empty((0, 2), np.float32)
    return pts.astype(np.float32)


def transform_poly(poly: np.ndarray, M: np.ndarray) -> np.ndarray:
    """用 3x3 矩阵变换多边形点（支持透视）。"""
    h = np.hstack([poly, np.ones((len(poly), 1))])
    q = h @ M.T
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return (q[:, :2] / w).astype(np.float32)


# --------------------------------------------------------------------------------------
# 标签读写
# --------------------------------------------------------------------------------------
def read_labels(txt: Path, task: str, w: int, h: int,
               kpt_shape: Tuple[int, int] = (0, 0)) -> List[Instance]:
    """读取 YOLO 标签为 Instance 列表。

    关键点(pose)标签格式（Ultralytics 标准）：
        cls xc yc w h  [kx1 ky1 (kvis1)]  [kx2 ky2 (kvis2)] ...
    其中前 5 个是目标框，后面按 kpt_shape=[K, 2或3] 依次排列 K 个关键点。
    可见性 kvis 取 0/1/2，2 表示可见（训练时可省略整列，此时 dim=2）。
    """
    insts: List[Instance] = []
    if not txt.exists():
        return insts  # 允许纯背景图（空标签）
    nkpt, ndim = kpt_shape
    for ln, line in enumerate(txt.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            raise ValueError(f"{txt}:{ln} 解析失败: {line!r}")

        if task == "pose":
            if nkpt <= 0:
                raise ValueError(f"{txt}:{ln} 关键点任务需要先确定 kpt_shape（-1 表示数据集中不统一）")
            expect = 4 + nkpt * ndim
            if len(vals) != expect:
                raise ValueError(
                    f"{txt}:{ln} 关键点标签应为 {expect + 1} 个数"
                    f"（4 个框值 + {nkpt} 个关键点 × {ndim} 维），实际 {len(vals) + 1}")
            xc, yc, bw, bh = vals[:4]
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            rest = np.array(vals[4:], np.float64).reshape(nkpt, ndim)
            kxy = rest[:, :2] * np.array([w, h], np.float64)       # 归一化 -> 像素
            kvis = rest[:, 2].astype(np.float32) if ndim == 3 else None
            insts.append(Instance(cls, "pose", box, kxy.astype(np.float32), kvis))
        elif task == "detection":
            if len(vals) != 4:
                raise ValueError(f"{txt}:{ln} 检测标签应为 5 个数，实际 {len(vals) + 1}")
            xc, yc, bw, bh = vals
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            insts.append(Instance(cls, "det", pts))
        elif task == "segment":
            if len(vals) < 6 or len(vals) % 2 != 0:
                raise ValueError(f"{txt}:{ln} 分割标签应为 cls + 至少 3 组 xy，实际 {len(vals) + 1} 个数")
            pts = np.array(vals, np.float64).reshape(-1, 2) * np.array([w, h], np.float64)
            insts.append(Instance(cls, "seg", pts.astype(np.float32)))
        elif task == "obb":
            if len(vals) != 8:
                raise ValueError(f"{txt}:{ln} OBB 标签应为 9 个数，实际 {len(vals) + 1}")
            pts = np.array(vals, np.float64).reshape(-1, 2) * np.array([w, h], np.float64)
            insts.append(Instance(cls, "obb", pts.astype(np.float32)))
        else:
            raise ValueError(f"未知 task: {task}")
    return insts


def encode_labels(insts: List[Instance], task: str, w: int, h: int,
                  min_side: float = 2.0, min_area: float = 8.0) -> List[str]:
    """把像素坐标转回归一化标签；丢弃过小/完全出界的目标。"""
    lines: List[str] = []
    for it in insts:
        if it.kind == "pose":
            line = _encode_pose(it, w, h, min_side)
            if line:
                lines.append(line)
            continue

        poly = clip_polygon(it.pts, w, h)
        if len(poly) < 3:
            continue
        if it.kind == "det":
            x1, y1, x2, y2 = bbox_of(poly)
            bw, bh = x2 - x1, y2 - y1
            if bw < min_side or bh < min_side:
                continue
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"{it.cls} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}")
        elif it.kind == "seg":
            if polygon_area(poly) < min_area:
                continue
            pairs = " ".join(f"{p[0] / w:.6f} {p[1] / h:.6f}" for p in poly)
            lines.append(f"{it.cls} {pairs}")
        elif it.kind == "obb":
            rect = cv2.minAreaRect(poly.astype(np.float32))
            (rw, rh) = rect[1]
            if min(rw, rh) < min_side:
                continue
            box = cv2.boxPoints(rect)
            pairs = " ".join(f"{p[0] / w:.6f} {p[1] / h:.6f}" for p in box)
            lines.append(f"{it.cls} {pairs}")
    return lines


def _encode_pose(it: Instance, w: int, h: int, min_side: float) -> Optional[str]:
    """把 pose 实例写回标签。关键点写到画面外时按约定置 0（视为不可见）。

    Ultralytics 读取时把 kx<0 或 ky<0 的关键点标记为不可见(kvis=0)，训练不会计入损失。

    重要：pose 的框必须覆盖关键点本身。因为旋转/透视后原本的框四边形不再是轴对齐矩形，
    直接取它的外接矩形会与关键点错位。这里统一由"可见关键点"的外接框来确定，
    保证框和关键点永远自洽（这正是 COCO-pose 标注的定义）。
    """
    if it.kpts is None or not len(it.kpts):
        return None
    kxy = it.kpts.copy()
    kvis = (np.asarray(it.kvis).astype(float).copy()
            if it.kvis is not None else np.full(len(kxy), 2.0))
    # 关键点跑出画面：坐标置 0 且可见性置 0。
    # 只置坐标不置可见性会造成"可见性=2 但坐标=(0,0)"的自相矛盾标签：
    # 训练时该点会被当成画面左上角的真实关键点来算损失，直接污染模型。
    out = (kxy[:, 0] < 0) | (kxy[:, 1] < 0) | (kxy[:, 0] > w) | (kxy[:, 1] > h)
    kxy[out] = 0.0
    kvis[out] = 0.0

    # 用可见关键点确定框
    vis_mask = ~out
    if it.kvis is not None:
        vis_mask &= (np.asarray(it.kvis) > 0)
    pts_ref = kxy[vis_mask] if vis_mask.any() else kxy[out == False] if (out == False).any() else None
    if pts_ref is None or not len(pts_ref):
        return None   # 所有关键点都跑出画面，整个目标丢弃

    x1, y1 = float(pts_ref[:, 0].min()), float(pts_ref[:, 1].min())
    x2, y2 = float(pts_ref[:, 0].max()), float(pts_ref[:, 1].max())
    bw, bh = x2 - x1, y2 - y1
    if bw < min_side or bh < min_side:
        return None
    if not (x2 > 0 and y2 > 0 and x1 < w and y1 < h):
        return None   # 整个目标跑出画面

    xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    parts = [f"{it.cls}", f"{xc:.6f}", f"{yc:.6f}", f"{bw / w:.6f}", f"{bh / h:.6f}"]
    for i in range(len(kxy)):
        parts.append(f"{kxy[i, 0] / w:.6f}")
        parts.append(f"{kxy[i, 1] / h:.6f}")
        if it.kvis is not None:
            parts.append(f"{kvis[i]:.0f}")
    return " ".join(parts)


# --------------------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------------------
@dataclass
class AugCfg:
    # 值域
    degrees: float = 10.0          # 旋转角 ±deg
    scale: float = 0.30            # 缩放 ±比例
    translate: float = 0.10        # 平移 ±比例
    shear: float = 0.0             # 剪切角 ±deg
    perspective: float = 0.0005    # 透视强度
    hsv_h: float = 0.015
    hsv_s: float = 0.50
    hsv_v: float = 0.30
    brightness: float = 0.20       # 亮度 ±
    contrast: float = 0.20         # 对比度 ±
    noise_sigma: float = 8.0
    sharpen_amount: float = 0.6    # 锐化强度 0~1（1 = 强锐化核）
    saltpepper_ratio: float = 0.02  # 椒盐噪点占"目标区域"像素的比例
    salt_ratio: float = 0.5        # 噪点里白点(salt)的占比，其余为黑点(pepper)
    occlusion_max: float = 0.35    # 遮挡块面积最多占目标框的比例
    # 触发概率
    p_hflip: float = 0.5
    p_vflip: float = 0.1
    p_rot90: float = 0.1
    p_rotate: float = 1.0
    p_scale: float = 0.6
    p_translate: float = 0.6
    p_shear: float = 0.0
    p_perspective: float = 0.2
    p_hsv: float = 0.6
    p_bright: float = 0.3
    p_blur: float = 0.15
    p_noise: float = 0.15
    p_sharpen: float = 0.15        # 锐化概率
    p_saltpepper: float = 0.10     # 椒盐概率
    p_cutout: float = 0.2
    p_occlusion: float = 0.0       # 遮挡增强概率（默认关，见 README 说明）
    p_mosaic: float = 0.4
    p_copypaste: float = 0.0       # 仅 segment
    # 其他
    min_side: float = 2.0          # 小于该像素边长的框/目标丢弃
    min_area: float = 8.0          # 分割/擦除最小面积(px^2)
    copy_paste_max: float = 0.3     # copy-paste 重叠上限
    # --- 仅关键点(pose)任务 ---
    kpt_shape: Tuple[int, int] = (0, 0)          # (关键点数量, 维度 2 或 3)
    flip_idx: Optional[Sequence[int]] = None     # 左右翻转时的关键点索引映射


# --------------------------------------------------------------------------------------
# 几何增强
# --------------------------------------------------------------------------------------
def _about_center(M: np.ndarray, cx: float, cy: float) -> np.ndarray:
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
    Ti = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    return T @ M @ Ti


def build_affine_matrix(w: int, h: int, cfg: AugCfg, rng: random.Random) -> Tuple[np.ndarray, bool]:
    """按概率叠加 透视/剪切/旋转/缩放/平移，返回 3x3 矩阵与是否含透视。"""
    cx, cy = w / 2.0, h / 2.0
    M = np.eye(3, dtype=np.float64)
    persp = False

    if cfg.perspective > 0 and rng.random() < cfg.p_perspective:
        v = (rng.uniform(-cfg.perspective, cfg.perspective),
             rng.uniform(-cfg.perspective, cfg.perspective))
        P = np.eye(3, dtype=np.float64)
        P[2, 0], P[2, 1] = v[0], v[1]
        M = P @ M
        persp = True

    if cfg.shear > 0 and rng.random() < cfg.p_shear:
        sh = math.tan(math.radians(rng.uniform(-cfg.shear, cfg.shear)))
        S = np.array([[1, sh, 0], [0, 1, 0], [0, 0, 1]], np.float64)
        M = _about_center(S, cx, cy) @ M

    if cfg.degrees > 0 and rng.random() < cfg.p_rotate:
        a = math.radians(rng.uniform(-cfg.degrees, cfg.degrees))
        cos, sin = math.cos(a), math.sin(a)
        R = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], np.float64)
        M = _about_center(R, cx, cy) @ M

    if cfg.scale > 0 and rng.random() < cfg.p_scale:
        s = rng.uniform(1.0 - cfg.scale, 1.0 + cfg.scale)
        s = max(s, 0.05)
        Z = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)
        M = _about_center(Z, cx, cy) @ M

    if cfg.translate > 0 and rng.random() < cfg.p_translate:
        tx = rng.uniform(-cfg.translate, cfg.translate) * w
        ty = rng.uniform(-cfg.translate, cfg.translate) * h
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)
        M = T @ M

    return M, persp


def warp_image(img: np.ndarray, M: np.ndarray, persp: bool) -> np.ndarray:
    h, w = img.shape[:2]
    if np.allclose(M, np.eye(3)):
        return img
    border = (114, 114, 114)
    if persp:
        return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return cv2.warpAffine(img, M[:2], (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def rot90_image(img: np.ndarray, insts: List[Instance], k: int) -> Tuple[np.ndarray, List[Instance]]:
    """逆时针 90° 的整数倍旋转（画布尺寸随之交换）。"""
    k %= 4
    if k == 0:
        return img, insts
    for _ in range(k):
        cur_h = img.shape[0]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for it in insts:
            x, y = it.pts[:, 0], it.pts[:, 1]
            # 顺时针 90°： (x, y) -> (H - y, x)
            it.pts = np.stack([cur_h - y, x], axis=1).astype(np.float32)
            if it.kpts is not None and len(it.kpts):
                kx, ky = it.kpts[:, 0], it.kpts[:, 1]
                it.kpts = np.stack([cur_h - ky, kx], axis=1).astype(np.float32)
    return img, insts


def transform_kpts(it: Instance, M: np.ndarray) -> None:
    """把实例的关键点用 3x3 矩阵变换（与框用同一矩阵，保证严格对齐）。"""
    if it.kpts is not None and len(it.kpts):
        it.kpts = transform_poly(it.kpts, M)


def flip_kpts(it: Instance, w: float, h: float, horizontal: bool,
              flip_idx: Optional[Sequence[int]] = None) -> None:
    """翻转关键点坐标，并按 flip_idx 重排索引。

    flip_idx 是"左右对称关键点的编号映射"。例如人脸 5 点标注
    flip_idx=[1,0,3,2,4] 表示翻转后 0 号和 1 号关键点互换身份。
    若数据集的 yaml 没给 flip_idx，就不做索引重排（此时翻转仅镜像坐标，
    对"左右对称"的关键点语义是错的 —— 所以 augment 会在启动时给出警告）。
    """
    if it.kpts is None or not len(it.kpts):
        return
    k = it.kpts.copy()
    if horizontal:
        k[:, 0] = w - k[:, 0]
    else:
        k[:, 1] = h - k[:, 1]
    if flip_idx is not None and len(flip_idx) == len(k):
        k = k[np.asarray(flip_idx, dtype=int)]
    it.kpts = k


def flip_image(img: np.ndarray, insts: List[Instance], horizontal: bool,
               flip_idx: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, List[Instance]]:
    h, w = img.shape[:2]
    img = cv2.flip(img, 1 if horizontal else 0)
    for it in insts:
        it.pts = it.pts.copy()
        if horizontal:
            it.pts[:, 0] = w - it.pts[:, 0]
        else:
            it.pts[:, 1] = h - it.pts[:, 1]
        flip_kpts(it, w, h, horizontal, flip_idx)
    return img, insts


# --------------------------------------------------------------------------------------
# 颜色增强（不改标注）
# --------------------------------------------------------------------------------------
def color_aug(img: np.ndarray, cfg: AugCfg, rng: random.Random) -> np.ndarray:
    if rng.random() < cfg.p_hsv:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hsv_h, cfg.hsv_h) * 180.0) % 180.0
        hsv[..., 1] *= 1.0 + rng.uniform(-cfg.hsv_s, cfg.hsv_s)
        hsv[..., 2] *= 1.0 + rng.uniform(-cfg.hsv_v, cfg.hsv_v)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if rng.random() < cfg.p_bright:
        alpha = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        beta = rng.uniform(-cfg.brightness, cfg.brightness) * 255.0
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if rng.random() < cfg.p_blur:
        k = rng.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if rng.random() < cfg.p_sharpen:
        img = sharpen_image(img, cfg.sharpen_amount)

    if rng.random() < cfg.p_noise:
        noise = np.random.default_rng(rng.getrandbits(32)).normal(0, cfg.noise_sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def sharpen_image(img: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """锐化。amount=1 用经典 3x3 强锐化核；较小时退化为温和的 unsharp mask。

    金属反光表面用强核会把高光噪点放大，所以按 amount 线性混合：
        out = img + amount * (hpf)
    其中 hpf = img - blur(img) 是高频细节（unsharp mask 形式，可控且不会过冲）。
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # unsharp mask：原图 + 强度 * (原图 - 模糊)，等价于增强边缘但不过度放大噪点
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def saltpepper_aug(img: np.ndarray, insts: List[Instance], cfg: AugCfg,
                   rng: random.Random) -> np.ndarray:
    """椒盐噪声：模拟传感器坏点 / 传输误码。

    只作用在目标（标注框）区域 —— 背景噪点对检测几乎无信息增益，反而稀释信号。
    参考实现里 numPixels=0 时切片为空、赋值静默失效，这里显式跳过。
    """
    if rng.random() >= cfg.p_saltpepper or not insts:
        return img
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for it in insts:
        # 用 floor/ceil 让区域向内收缩半像素，保证噪点严格落在目标内部，
        # 而不是压在目标与背景的边界线上（那样会污染背景像素）。
        x1f, y1f, x2f, y2f = bbox_of(it.pts)
        x1, y1 = int(np.floor(x1f)) + 1, int(np.floor(y1f)) + 1
        x2, y2 = int(np.ceil(x2f)) - 1, int(np.ceil(y2f)) - 1
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 >= x1 and y2 >= y1:
            mask[y1:y2 + 1, x1:x2 + 1] = 255
    ys, xs = np.nonzero(mask)
    n_total = len(xs)
    n = int(n_total * cfg.saltpepper_ratio)
    if n <= 0:
        return img   # 目标太小，比例算出来 0 个点 -> 直接跳过（旧实现会静默失效）
    n_salt = int(n * float(np.clip(cfg.salt_ratio, 0.0, 1.0)))
    idx = np.random.default_rng(rng.getrandbits(32)).choice(n_total, size=n, replace=False)
    out = img.copy()
    if n_salt > 0:
        out[ys[idx[:n_salt]], xs[idx[:n_salt]]] = (255, 255, 255)
    if n > n_salt:
        out[ys[idx[n_salt:]], xs[idx[n_salt:]]] = (0, 0, 0)
    return out


def occlusion_aug(img: np.ndarray, insts: List[Instance], cfg: AugCfg,
                  rng: random.Random) -> Tuple[np.ndarray, List[Instance]]:
    """遮挡增强：用背景纹理块盖住目标的一部分，模拟产线上的相互遮挡/遮挡物。

    与 cutout 的区别：
        cutout   只擦"没有目标"的区域（保证标注一定有效）
        occlusion 故意盖住目标（提升遮挡鲁棒性），因此必须处理标注

    标注处理规则（重要）：
        · 遮挡后目标残余可见面积 < min_area  -> 丢弃该目标（几乎全被盖住，留着是错误标注）
        · 否则保留原框（YOLO 检测任务中，被遮挡目标仍应被检测，这是标准做法）

    仅对 det / pose 生效：分割任务的 polygon 被遮挡后形状不再准确，
    OBB 的旋转角也会失效，所以这两类任务下本函数不做事。
    """
    if rng.random() >= cfg.p_occlusion or not insts:
        return img, insts
    # occlusion_max=0 表示"不遮挡"：直接跳过，不再被下限夹成 0.05
    if cfg.occlusion_max <= 0:
        return img, insts
    h, w = img.shape[:2]
    usable = [it for it in insts if it.kind in ("det", "pose")]
    if not usable:
        return img, insts

    out = img.copy()
    survivors: List[Instance] = []
    omax = float(np.clip(cfg.occlusion_max, 0.0, 1.0))
    blocks: List[Tuple[int, int, int, int]] = []
    # 让"被遮目标"尽量互不重复：每个目标生成 1~2 块，目标数 ≥2 时至少有两个目标被覆盖。
    # 这样无论图里有几个目标，遮挡都确实作用到了标注上，而不是随机空转。
    order = list(range(len(usable)))
    rng.shuffle(order)
    n_targets = min(len(order), max(1, rng.randint(1, min(3, len(order)))))
    for ti in order[:n_targets]:
        target = usable[ti]
        x1, y1, x2, y2 = bbox_of(target.pts)
        bw, bh = x2 - x1, y2 - y1
        if bw < 6 or bh < 6:
            continue
        # occlusion_max 表示"该目标最多被遮掉的面积比例"，换算成边长系数是 sqrt(比例)。
        side = float(np.sqrt(omax)) * rng.uniform(0.65, 1.0)
        cw = max(3, int(round(bw * side)))
        ch = max(3, int(round(bh * side)))
        # 块中心落在目标框中心附近（±1/6 框宽高内随机）。
        # 若在整框内均匀取中心，块会偏到角上、只盖住 1/4 面积，遮挡强度远低于参数设定。
        cx = (x1 + x2) / 2 + rng.uniform(-bw / 6, bw / 6)
        cy = (y1 + y2) / 2 + rng.uniform(-bh / 6, bh / 6)
        bx1 = int(np.clip(cx - cw / 2, 0, max(0, w - cw)))
        by1 = int(np.clip(cy - ch / 2, 0, max(0, h - ch)))
        bx2 = min(w, bx1 + cw)
        by2 = min(h, by1 + ch)
        if bx2 <= bx1 or by2 <= by1:
            continue
        blocks.append((bx1, by1, bx2, by2))
        # 填充：一半用随机噪声纹理，一半用均值色块（贴近"纸箱/托盘/气泡膜"等真实遮挡物）
        if rng.random() < 0.5:
            patch = np.random.default_rng(rng.getrandbits(32)).integers(
                60, 200, (by2 - by1, bx2 - bx1, 3), dtype=np.uint8)
        else:
            patch = np.full((by2 - by1, bx2 - bx1, 3),
                            rng.randint(60, 180), np.uint8)
        out[by1:by2, bx1:bx2] = patch

    if not blocks:
        return img, insts

    # 用可见面积判断哪些目标需要丢弃
    obs_mask = np.zeros((h, w), np.uint8)
    for bx1, by1, bx2, by2 in blocks:
        obs_mask[by1:by2, bx1:bx2] = 255

    for it in insts:
        if it.kind not in ("det", "pose"):
            survivors.append(it)      # 分割/OBB 不参与遮挡统计，原样保留
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_of(it.pts)]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        covered = int((obs_mask[y1:y2, x1:x2] > 0).sum())
        visible = area - covered
        if visible < cfg.min_area:
            continue                 # 几乎全被遮挡 -> 丢弃
        survivors.append(it)
    return out, survivors


# --------------------------------------------------------------------------------------
# 擦除（只擦没有目标的区域）
# --------------------------------------------------------------------------------------
def cutout_aug(img: np.ndarray, insts: List[Instance], cfg: AugCfg, rng: random.Random) -> np.ndarray:
    if rng.random() >= cfg.p_cutout:
        return img
    h, w = img.shape[:2]
    boxes = np.array([bbox_of(it.pts) for it in insts], np.float32) if insts else np.zeros((0, 4), np.float32)
    for _ in range(rng.randint(1, 3)):
        bw = int(w * rng.uniform(0.04, 0.14))
        bh = int(h * rng.uniform(0.04, 0.14))
        if bw < 4 or bh < 4:
            continue
        x = rng.randint(0, max(0, w - bw - 1))
        y = rng.randint(0, max(0, h - bh - 1))
        if boxes.size and ioa((x, y, x + bw, y + bh), boxes).max() > 0.05:
            continue  # 会盖到目标，跳过
        if rng.random() < 0.5:
            img[y:y + bh, x:x + bw] = (114, 114, 114)
        else:
            patch = np.random.default_rng(rng.getrandbits(32)).integers(0, 256, (bh, bw, 3), dtype=np.uint8)
            img[y:y + bh, x:x + bw] = patch
    return img


# --------------------------------------------------------------------------------------
# Mosaic：4 宫格拼接（保持宽高比 cover + 居中裁剪，避免拉伸变形）
# --------------------------------------------------------------------------------------
def place_in_cell(img: np.ndarray, insts: List[Instance], cw: int, ch: int,
                  rng: random.Random) -> Tuple[np.ndarray, List[Instance]]:
    sh, sw = img.shape[:2]
    s = max(cw / sw, ch / sh) * rng.uniform(1.0, 1.25)
    nw, nh = max(cw, int(round(sw * s))), max(ch, int(round(sh * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ox = rng.randint(0, nw - cw)
    oy = rng.randint(0, nh - ch)
    crop = resized[oy:oy + ch, ox:ox + cw]
    out: List[Instance] = []
    for it in insts:
        p = (it.pts * s - np.array([ox, oy], np.float32)).astype(np.float32)
        poly = clip_polygon(p, cw, ch)
        if len(poly) < 3:
            continue
        # 关键点用同样的缩放+偏移，且不裁剪（裁剪会把点掰到边上，反而失真）
        kpts = None
        if it.kpts is not None and len(it.kpts):
            kpts = (it.kpts * s - np.array([ox, oy], np.float32)).astype(np.float32)
        out.append(Instance(it.cls, it.kind, poly, kpts,
                            None if it.kvis is None else it.kvis.copy()))
    return crop, out


def mosaic_aug(sources: List[Tuple[np.ndarray, List[Instance]]], w: int, h: int,
               rng: random.Random) -> Tuple[np.ndarray, List[Instance]]:
    cw, ch = w // 2, h // 2
    canvas = np.full((h, w, 3), 114, np.uint8)
    insts: List[Instance] = []
    offsets = [(0, 0), (cw, 0), (0, ch), (cw, ch)]
    for (img, ins), (cx, cy) in zip(sources, offsets):
        patch, pins = place_in_cell(img, ins, cw, ch, rng)
        canvas[cy:cy + ch, cx:cx + cw] = patch
        for it in pins:
            it.pts = it.pts + np.array([cx, cy], np.float32)
            if it.kpts is not None and len(it.kpts):
                it.kpts = it.kpts + np.array([cx, cy], np.float32)
            insts.append(it)
    return canvas, insts

# --------------------------------------------------------------------------------------
# Copy-Paste（仅分割；从其他图里抠一个实例贴过来）
# --------------------------------------------------------------------------------------
def copy_paste_aug(img: np.ndarray, insts: List[Instance],
                   donor_img: Optional[np.ndarray], donor_insts: List[Instance],
                   cfg: AugCfg, rng: random.Random) -> Tuple[np.ndarray, List[Instance]]:
    if donor_img is None or not donor_insts or rng.random() >= cfg.p_copypaste:
        return img, insts
    h, w = img.shape[:2]
    dh, dw = donor_img.shape[:2]
    donor_img = cv2.resize(donor_img, (w, h), interpolation=cv2.INTER_LINEAR)
    src = donor_insts[rng.randrange(len(donor_insts))]
    poly = src.pts.copy()
    poly *= np.array([w / dw, h / dh], np.float32)

    if rng.random() < 0.5:  # 水平翻转
        poly[:, 0] = w - poly[:, 0]
    poly += np.array([rng.uniform(-0.12, 0.12) * w, rng.uniform(-0.12, 0.12) * h], np.float32)

    poly_c = clip_polygon(poly, w, h)
    if len(poly_c) < 3 or polygon_area(poly_c) < cfg.min_area * 4:
        return img, insts
    box = bbox_of(poly_c)
    boxes = np.array([bbox_of(it.pts) for it in insts], np.float32) if insts else np.zeros((0, 4), np.float32)
    if boxes.size and ioa(box, boxes).max() >= cfg.copy_paste_max:
        return img, insts  # 贴上去会盖住已有目标，跳过

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly_c.round().astype(np.int32)], 255)
    if mask.sum() == 0:
        return img, insts
    cv2.copyTo(donor_img, mask, img)
    out = insts + [Instance(src.cls, "seg", poly_c)]
    return img, out


# --------------------------------------------------------------------------------------
# 数据集扫描
# --------------------------------------------------------------------------------------
@dataclass
class Item:
    img_path: Path
    lbl_path: Path


def list_items(images_dir: Path, labels_dir: Path) -> List[Item]:
    items: List[Item] = []
    for p in sorted(images_dir.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            items.append(Item(p, labels_dir / (p.stem + ".txt")))
    return items


def load_pair(item: Item, task: str, cfg: Optional[AugCfg] = None) -> Tuple[np.ndarray, List[Instance]]:
    img = imread_unicode(item.img_path)
    h, w = img.shape[:2]
    kpt_shape = cfg.kpt_shape if (cfg is not None and task == "pose") else (0, 0)
    insts = read_labels(item.lbl_path, task, w, h, kpt_shape)
    return img, insts


def _read_yaml(path: Path) -> dict:
    """读 data.yaml，失败返回空字典。"""
    if not path.exists():
        return {}
    import yaml
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:  # noqa: BLE001
        return {}


def find_cfg_files(image_path: Path, label_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """在图片目录 / 标签目录及其上级目录里找 data.yaml 与 classes.txt。

    图片和标签可以放在完全不同的位置，所以两类文件依次在这些地方找：
        图片目录本身 -> 标签目录本身 -> 图片目录上级 -> 标签目录上级
    返回 (data_yaml 路径或 None, classes_txt 路径或 None)。
    """
    yaml_names = ("data.yaml", "dataset.yaml")
    dirs: List[Path] = []
    for d in (image_path, label_path):
        # 往上找 4 层：足以覆盖 .../root/images/train、.../root/images 这类布局
        for cand in (d, d.parent, d.parent.parent, d.parent.parent.parent):
            if cand not in dirs:
                dirs.append(cand)

    yml: Optional[Path] = None
    for d in dirs:
        for nm in yaml_names:
            p = d / nm
            if p.is_file():
                yml = p
                break
        if yml:
            break
    cls: Optional[Path] = None
    for d in dirs:
        p = d / "classes.txt"
        if p.is_file():
            cls = p
            break
    return yml, cls


@dataclass
class Layout:
    """输入目录布局：图片与标签各自独立的路径。"""
    image_path: Path
    label_path: Path
    has_splits: bool
    yaml_file: Optional[Path]
    classes_file: Optional[Path]

    @property
    def root(self) -> Path:
        """配置文件所在的"数据集根目录"（找不到时退化为图片目录的上级）。"""
        if self.yaml_file is not None:
            return self.yaml_file.parent
        if self.classes_file is not None:
            return self.classes_file.parent
        return self.image_path.parent

    def train_dirs(self) -> Tuple[Path, Path]:
        return ((self.image_path / "train", self.label_path / "train")
                if self.has_splits else (self.image_path, self.label_path))

    def val_dirs(self) -> Optional[Tuple[Path, Path]]:
        if not self.has_splits:
            return None
        v = (self.image_path / "val", self.label_path / "val")
        return v if v[0].is_dir() else None


def build_layout(image_path, label_path, resolve: bool = True) -> Layout:
    """把用户输入的两条路径解析成一个 Layout。resolve=False 时只做字符串转换。"""
    ip = _resolve_user_path(str(image_path)) if resolve else Path(str(image_path)).expanduser()
    lp = _resolve_user_path(str(label_path)) if resolve else Path(str(label_path)).expanduser()
    if not ip.is_dir():
        raise ValueError(f"图片路径不存在或不是目录：{ip}")
    if not lp.is_dir():
        raise ValueError(
            f"标签路径不存在或不是目录：{lp}\n"
            f"（如果还没标注，请先用 X-AnyLabeling 标注并生成 .txt 标签）")
    has_splits = (ip / "train").is_dir()
    yml, cls = find_cfg_files(ip, lp)
    return Layout(ip, lp, has_splits, yml, cls)


def derive_from_src(src: str) -> Tuple[str, str]:
    """兼容旧参数 --src：从数据集根目录推出 image_path / label_path。"""
    s = _resolve_user_path(str(src))
    img = s / "images"
    lbl = s / "labels"
    if img.is_dir() and lbl.is_dir():
        return str(img), str(lbl)
    if img.is_dir():
        return str(img), str(s)
    raise ValueError(
        f"--src 已改为 --image-path / --label-path 两个参数。\n"
        f"在 {s} 下没有找到 images/ 与 labels/ 子目录，请直接指定这两个路径。")


def detect_kpt_shape(root: Path, items: List[Item]) -> Tuple[int, int]:
    """从 data.yaml 的 kpt_shape 读取关键点数量与维度；读不到则按标签列数反推。"""
    d = _read_yaml(root / "data.yaml") or _read_yaml(root / "dataset.yaml")
    ks = d.get("kpt_shape")
    if isinstance(ks, (list, tuple)) and len(ks) == 2 and int(ks[0]) > 0:
        return int(ks[0]), int(ks[1])
    # 反推：列数 = 1(cls) + 4(box) + K*ndim
    lens = set()
    for it in items[:100]:
        if not it.lbl_path.exists():
            continue
        for line in it.lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if p:
                lens.add(len(p))
    if len(lens) == 1:
        n = lens.pop()
        extra = n - 5
        if extra > 0:
            for ndim in (3, 2):   # 优先匹配 3 维（带可见性，最常见）
                if extra % ndim == 0:
                    return extra // ndim, ndim
    raise ValueError("无法确定 kpt_shape：请在 data.yaml 中写明，例如 kpt_shape: [5, 3]")


def load_flip_idx(root: Path) -> Optional[List[int]]:
    """读取 data.yaml 里的 flip_idx（关键点左右翻转映射）。"""
    d = _read_yaml(root / "data.yaml") or _read_yaml(root / "dataset.yaml")
    fi = d.get("flip_idx")
    if isinstance(fi, (list, tuple)) and len(fi):
        return [int(v) for v in fi]
    return None


def load_names(root: Path, classes_file: Optional[Path] = None) -> List[str]:
    """类别名：优先 data.yaml 的 names，其次 classes.txt。"""
    d = _read_yaml(root / "data.yaml") or _read_yaml(root / "dataset.yaml")
    names = d.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(int(i) for i in names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    txt = classes_file or (root / "classes.txt")
    if txt and txt.exists():
        got = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines()
               if ln.strip()]
        if got:
            return got
    raise SystemExit(f"找不到类别名：{root / 'data.yaml'} 里没有 names，"
                     f"也没有 classes.txt。请用 --names 指定")


def autodetect_task(root: Path, items: List[Item]) -> str:
    """按标签列数（及 yaml 里的 kpt_shape）推断任务。

    判断顺序：
      1. data.yaml 明确写了 kpt_shape -> pose
      2. 全是 5 列 -> detection
      3. 全是 9 列 -> 有歧义（OBB 与 4 点多边形），要求显式指定
      4. 其余可解析为多边形的偶数列 -> segment
      5. 列数 = 5 + K*(2 或 3) -> pose（关键点）
    """
    # 优先相信 yaml 的 kpt_shape
    d = _read_yaml(root / "data.yaml") or _read_yaml(root / "dataset.yaml")
    ks = d.get("kpt_shape")
    if isinstance(ks, (list, tuple)) and len(ks) == 2 and int(ks[0]) > 0:
        return "pose"

    lens = set()
    for it in items[:200]:
        if not it.lbl_path.exists():
            continue
        for line in it.lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if parts:
                lens.add(len(parts))
    if not lens:
        return "detection"
    if lens == {5}:
        return "detection"
    if lens == {9}:
        raise SystemExit("标签都是 9 列：可能是 OBB(4 点)、4 点多边形分割，"
                         "也可能是 1 个关键点(3 维)的 pose。\n"
                         "请显式指定 --task obb / --task segment / --task pose")
    if 5 in lens and any(n > 5 for n in lens):
        raise SystemExit(f"标签列数混杂 {sorted(lens)}，无法自动推断，请显式 --task")
    # 注意判定顺序：分割的列数是"任意奇数的多边形点数"，几乎能覆盖所有列数，
    # 所以必须先把 segment 认出来，否则会把分割误判成 pose（85 列会被当成 5+80）。
    # pose 的列数 = 5 + K*ndim，且**只有**在 yaml 有 kpt_shape 时才能可靠确认；
    # 没有 kpt_shape 时，只有列数不能被解析为合法多边形（即偶数且 >=6 才可能是分割）才判 pose。
    if all((n - 1) % 2 == 0 and n >= 7 for n in lens):
        return "segment"
    pose_ok = True
    for n in lens:
        extra = n - 5
        if extra <= 0 or not (extra % 3 == 0 or extra % 2 == 0):
            pose_ok = False
            break
    if pose_ok and lens:
        return "pose"
    if 9 in lens:
        return "obb"
    raise SystemExit(f"无法自动推断任务（标签列数 {sorted(lens)}），请显式 --task")


# --------------------------------------------------------------------------------------
# 单张样本生成
# --------------------------------------------------------------------------------------
def build_augmented(idx: int, pool: List[Item], cfg: AugCfg, task: str,
                    rng: random.Random) -> Tuple[np.ndarray, List[Instance]]:
    img, insts = load_pair(pool[idx], task, cfg)

    # ---- Mosaic：4 张图拼一张，等价于 1 张图造出 4 个目标位置 ----
    if cfg.p_mosaic > 0 and rng.random() < cfg.p_mosaic and len(pool) >= 4:
        h, w = img.shape[:2]
        picks = [idx] + [rng.randrange(len(pool)) for _ in range(3)]
        sources = []
        for j in picks:
            simg, sins = load_pair(pool[j], task, cfg)
            if rng.random() < cfg.p_hflip:
                simg, sins = flip_image(simg, sins, True, cfg.flip_idx)
            if rng.random() < cfg.p_vflip:
                simg, sins = flip_image(simg, sins, False, cfg.flip_idx)
            sources.append((simg, sins))
        canvas, insts = mosaic_aug(sources, w, h, rng)
        img = canvas
        insts = [it for it in insts]
        # mosaic 之后仍有颜色的随机抖动、椒盐与擦除
        img = color_aug(img, cfg, rng)
        img = saltpepper_aug(img, insts, cfg, rng)
        img = cutout_aug(img, insts, cfg, rng)
        img, insts = occlusion_aug(img, insts, cfg, rng)
        return img, insts

    # ---- 单图增强：90° 旋转 -> 翻转 -> 仿射/透视 -> 颜色 -> 擦除 -> copy-paste ----
    if rng.random() < cfg.p_rot90:
        img, insts = rot90_image(img, insts, rng.choice([1, 2, 3]))
    if rng.random() < cfg.p_hflip:
        img, insts = flip_image(img, insts, True, cfg.flip_idx)
    if rng.random() < cfg.p_vflip:
        img, insts = flip_image(img, insts, False, cfg.flip_idx)

    h, w = img.shape[:2]
    M, persp = build_affine_matrix(w, h, cfg, rng)
    if not np.allclose(M, np.eye(3)):
        img = warp_image(img, M, persp)
        for it in insts:
            it.pts = transform_poly(it.pts, M)
            transform_kpts(it, M)  # 关键点必须与框用同一个矩阵，否则会错位

    img = color_aug(img, cfg, rng)
    # 椒盐在几何变换之后：此时框已是最终位置，只污染目标区域更合理
    img = saltpepper_aug(img, insts, cfg, rng)
    img = cutout_aug(img, insts, cfg, rng)
    # 遮挡放最后：它会改变目标可见性并可能剔除目标，必须在所有几何/颜色操作之后
    img, insts = occlusion_aug(img, insts, cfg, rng)

    if task == "segment" and cfg.p_copypaste > 0:
        j = rng.randrange(len(pool))
        donor_img, donor_insts = load_pair(pool[j], task, cfg) if j != idx else (None, [])
        img, insts = copy_paste_aug(img, insts, donor_img, donor_insts, cfg, rng)
    return img, insts


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def augment_split(items: List[Item], pool: List[Item], out_dir: Path, splits: str,
                  task: str, cfg: AugCfg, times: int, include_original: bool,
                  rng: random.Random, limit: Optional[int] = None) -> Tuple[int, int]:
    (out_dir / "images" / splits).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / splits).mkdir(parents=True, exist_ok=True)
    imgs_out = out_dir / "images" / splits
    lbls_out = out_dir / "labels" / splits

    count = 0
    stats: Dict[int, int] = {}
    if limit:
        items = items[:limit]

    if include_original:
        for it in items:
            shutil.copy2(it.img_path, imgs_out / it.img_path.name)
            if it.lbl_path.exists():
                shutil.copy2(it.lbl_path, lbls_out / (it.img_path.stem + ".txt"))
            else:
                (lbls_out / (it.img_path.stem + ".txt")).write_text("", encoding="utf-8")

    for n, it in enumerate(items, 1):
        for k in range(1, times + 1):
            img, insts = build_augmented(n - 1, pool, cfg, task, rng)
            stem = f"{it.img_path.stem}_aug{k}"
            imwrite_unicode(imgs_out / f"{stem}{it.img_path.suffix}", img, quality=95)
            h, w = img.shape[:2]
            lines = encode_labels(insts, task, w, h, cfg.min_side, cfg.min_area)
            (lbls_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                  encoding="utf-8")
            count += 1
            for ln in lines:
                c = int(ln.split()[0])
                stats[c] = stats.get(c, 0) + 1
        if n % 25 == 0 or n == len(items):
            print(f"  [{splits}] {n}/{len(items)} 张原图完成，已生成 {count} 张增强图")
    print(f"  [{splits}] 类别实例数(增强部分): {stats}")
    return count, sum(stats.values())


def copy_items(items: List[Item], out_dir: Path, splits: str) -> int:
    """把图像与标签原样复制到目标 split（验证集绝不做增强）。"""
    (out_dir / "images" / splits).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / splits).mkdir(parents=True, exist_ok=True)
    n = 0
    for it in items:
        shutil.copy2(it.img_path, out_dir / "images" / splits / it.img_path.name)
        if it.lbl_path.exists():
            shutil.copy2(it.lbl_path, out_dir / "labels" / splits / (it.img_path.stem + ".txt"))
        else:
            (out_dir / "labels" / splits / (it.img_path.stem + ".txt")).write_text("", encoding="utf-8")
        n += 1
    return n


def list_output_items(out_dir: Path, task: str = "auto") -> Tuple[List[Item], str, List[str]]:
    """列出一个"已生成"的输出数据集里的图片，并推断任务 / 类别。

    输出目录结构由 augment_split 决定：<out>/images/train、<out>/labels/train。
    也兼容用户手工整理成平铺 images/ + labels/ 的情况。
    返回 (items, task, names)。失败抛 ValueError。
    """
    out_dir = _resolve_user_path(str(out_dir))
    if not out_dir.is_dir():
        raise ValueError(f"输出目录不存在：{out_dir}")

    img_dir = lbl_dir = None
    for sd in ("train", "val", ""):
        cand_i = (out_dir / "images" / sd) if sd else (out_dir / "images")
        cand_l = (out_dir / "labels" / sd) if sd else (out_dir / "labels")
        if cand_i.is_dir() and any(p.is_file() and p.suffix.lower() in IMG_EXT
                                   for p in cand_i.iterdir()):
            img_dir, lbl_dir = cand_i, cand_l
            break
    if img_dir is None:
        raise ValueError(
            f"在 {out_dir} 下没有找到图片。\n"
            f"期望结构：{out_dir}/images/train/*.jpg（或 {out_dir}/images/*.jpg）")

    items = list_items(img_dir, lbl_dir)
    if not items:
        raise ValueError(f"在 {img_dir} 里没有找到图片")

    # 输出目录里通常有 data.yaml，直接用它读类别与任务
    t = task
    yml = out_dir / "data.yaml"
    d = _read_yaml(yml) or _read_yaml(out_dir / "dataset.yaml")
    if t == "auto":
        ks = d.get("kpt_shape")
        if isinstance(ks, (list, tuple)) and len(ks) == 2 and int(ks[0]) > 0:
            t = "pose"
        else:
            try:
                t = autodetect_task(out_dir, items)
            except SystemExit as e:
                raise ValueError(str(e).strip())
    try:
        names = load_names(out_dir, out_dir / "classes.txt")
    except SystemExit as e:
        raise ValueError(str(e).strip())
    return items, t, names


def preview_output_image(out_dir, index: int = 0) -> np.ndarray:
    """读取输出数据集里的第 index 张图，画上它的标注，返回 BGR 图。

    与 preview_image 的区别：那个是对**源图片实时增强**，
    这个只是**如实显示已经生成好的文件**（用来核对增强产物和标注是否正确）。
    """
    items, task, names = list_output_items(out_dir)
    if not 0 <= index < len(items):
        raise ValueError(f"索引 {index} 超出范围（共 {len(items)} 张）")
    item = items[index]
    img = imread_unicode(item.img_path)
    h, w = img.shape[:2]
    kpt_shape = (0, 0)
    if task == "pose":
        try:
            kpt_shape = detect_kpt_shape(_resolve_user_path(str(out_dir)), items)
        except (ValueError, OSError):
            kpt_shape = (0, 0)
    insts = read_labels(item.lbl_path, task, w, h, kpt_shape)
    vis = img.copy()
    for it in insts:
        draw_instance(vis, it, w, h, names)
    cv2.putText(vis, f"{item.img_path.name}  [{task}]  (augmented)", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_instance(vis: np.ndarray, it: Instance, w: int, h: int,
                  names: Optional[List[str]] = None) -> None:
    """在图上画出一个实例：检测框为绿色矩形，分割/OBB 为黄色多边形，
    关键点为红点 + 连线（绿色表示该点可见，灰色表示被标为不可见）。"""
    poly = clip_polygon(it.pts, w, h)
    if it.kind == "pose":
        if len(poly) >= 3:
            x1, y1, x2, y2 = [int(v) for v in bbox_of(poly)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 关键点：先连线，再画点
        if it.kpts is not None and len(it.kpts):
            k = it.kpts
            # 画点
            for i, (kx, ky) in enumerate(k):
                if not (0 <= kx < w and 0 <= ky < h):
                    continue  # 跑出画面不画
                visible = True
                if it.kvis is not None and i < len(it.kvis):
                    visible = it.kvis[i] > 0
                color = (0, 255, 0) if visible else (150, 150, 150)
                cv2.circle(vis, (int(kx), int(ky)), 4, color, -1)
                cv2.circle(vis, (int(kx), int(ky)), 4, (0, 0, 0), 1)
            # 关键点彼此连成骨架（按标注顺序）
            pts = [[int(x), int(y)] for x, y in k if 0 <= x < w and 0 <= y < h]
            if len(pts) > 1:
                cv2.polylines(vis, [np.array(pts, np.int32)], False, (0, 165, 255), 1)
    elif it.kind == "det":
        if len(poly) >= 3:
            x1, y1, x2, y2 = [int(v) for v in bbox_of(poly)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        if len(poly) >= 3:
            cv2.polylines(vis, [poly.round().astype(np.int32)], True, (0, 255, 255), 2)

    nm = names[it.cls] if (names and 0 <= it.cls < len(names)) else str(it.cls)
    # 标签文字放在第一个关键点或框左上角
    if it.kind == "pose" and it.kpts is not None and len(it.kpts):
        tx, ty = int(it.kpts[0][0]), int(it.kpts[0][1]) - 6
    elif len(poly) >= 3:
        x1, y1, _, _ = bbox_of(poly)
        tx, ty = int(x1), int(y1) - 4
    else:
        return
    cv2.putText(vis, nm, (max(0, tx), max(10, ty)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)


def write_preview(item: Item, out: Path, task: str, cfg: AugCfg, pool: List[Item],
                  rng: random.Random, idx: int) -> np.ndarray:
    img, insts = build_augmented(idx, pool, cfg, task, rng)
    vis = img.copy()
    h, w = vis.shape[:2]
    for it in insts:
        draw_instance(vis, it, w, h)
    cv2.putText(vis, f"{item.img_path.name}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    imwrite_unicode(out, vis)
    return vis


# --------------------------------------------------------------------------------------
# 参数单一数据源：CLI(argparse) 与 GUI(augment_gui.py) 都从这里生成，避免两边不一致
# kind: path/int/float/ratio(0-1)/bool/choice/str
# --------------------------------------------------------------------------------------
PARAM_SPEC: List[dict] = [
    # ---- 基础 ----
    dict(name="image_path", flag="--image-path", kind="path", type=str, default=None,
         group="", label="图片路径",
         help="图片所在目录（如 .../images 或 .../images/train）"),
    dict(name="label_path", flag="--label-path", kind="path", type=str, default=None,
         group="", label="标签路径",
         help="标签(txt)所在目录（如 .../labels 或 .../labels/train），可与图片不同盘"),
    dict(name="out", flag="--out", kind="path", type=str, default=None,
         group="", label="输出目录", help="输出目录，默认在图片目录旁生成 <图片目录名>_aug"),
    dict(name="task", flag="--task", kind="choice", type=str, default="auto",
         group="", label="任务类型",
         choices=["auto", "detection", "segment", "obb", "pose"],
         help="任务类型；auto 时按标签列数推断（9 列有歧义需指定）"),
    dict(name="times", flag="--times", kind="int", type=int, default=2, lo=1, hi=20,
         group="", label="扩增倍数", help="每张原图生成几张增强图（扩倍数）"),
    dict(name="val_ratio", flag="--val-ratio", kind="ratio", type=float, default=0.2,
         lo=0.05, hi=0.5, group="", label="val 划分比例",
         help="源数据集没有 val 时，从 train 划出多少比例作为 val"),
    dict(name="seed", flag="--seed", kind="int", type=int, default=0, lo=0, hi=2**31 - 1,
         group="", label="随机种子", help="随机种子，保证可复现"),
    dict(name="limit", flag="--limit", kind="int", type=int, default=None, lo=0, hi=100000,
         group="", label="只处理前 N 张", help="只处理前 N 张（调试用，留空=全部）"),
    dict(name="preview", flag="--preview", kind="int", type=int, default=0, lo=0, hi=200,
         group="", label="预览图数量", help="额外生成 N 张可视化预览到 <out>/preview"),
    dict(name="quality", flag="--quality", kind="int", type=int, default=95, lo=50, hi=100,
         group="", label="JPEG 质量", help="JPEG 保存质量"),
    dict(name="min_side", flag="--min-side", kind="float", type=float, default=2.0, lo=0.0, hi=50.0,
         group="", label="最小框边长(px)", help="小于该像素边长的框丢弃"),
    dict(name="names", flag="--names", kind="str", type=str, default=None,
         group="", label="类别名(逗号分隔)", help="类别名，逗号分隔，覆盖 data.yaml"),
    dict(name="no_original", flag="--no-original", kind="bool", action="store_true", default=False,
         group="", label="不含原图", help="不把原图复制进新数据集"),
    dict(name="overwrite", flag="--overwrite", kind="bool", action="store_true", default=False,
         group="", label="覆盖输出目录", help="输出目录已存在时覆盖"),
    dict(name="dry_run", flag="--dry-run", kind="bool", action="store_true", default=False,
         group="", label="只统计不写", help="只统计不写文件"),
    dict(name="allow_missing_labels", flag="--allow-missing-labels", kind="bool",
         action="store_true", default=False, group="", label="允许缺标签",
         help="允许部分图片没有 .txt 标签（默认会报错，防止标签路径填错）"),

    # ---- 几何增强 ----
    dict(name="degrees", flag="--degrees", kind="float", type=float, default=10.0, lo=0.0, hi=180.0,
         group="增强强度", label="旋转 ±度", help="旋转 ±度"),
    dict(name="scale", flag="--scale", kind="float", type=float, default=0.30, lo=0.0, hi=1.0,
         group="增强强度", label="缩放 ±比例", help="缩放 ±比例"),
    dict(name="translate", flag="--translate", kind="float", type=float, default=0.10, lo=0.0, hi=1.0,
         group="增强强度", label="平移 ±比例", help="平移 ±比例"),
    dict(name="shear", flag="--shear", kind="float", type=float, default=0.0, lo=0.0, hi=45.0,
         group="增强强度", label="剪切 ±度", help="剪切 ±度"),
    dict(name="perspective", flag="--perspective", kind="float", type=float, default=0.0005,
         lo=0.0, hi=0.001, group="增强强度", label="透视强度", help="透视强度（0~0.001）"),

    # ---- 各增强触发概率 ----
    dict(name="p_mosaic", flag="--p-mosaic", kind="ratio", type=float, default=0.4, lo=0.0, hi=1.0,
         group="触发概率", label="Mosaic 概率", help="Mosaic 概率(0 关闭)"),
    dict(name="p_copypaste", flag="--p-copypaste", kind="ratio", type=float, default=0.0, lo=0.0, hi=1.0,
         group="触发概率", label="Copy-Paste 概率", help="Copy-Paste 概率(仅 segment)"),
    dict(name="p_hflip", flag="--p-hflip", kind="ratio", type=float, default=0.5, lo=0.0, hi=1.0,
         group="触发概率", label="水平翻转概率", help="水平翻转概率"),
    dict(name="p_vflip", flag="--p-vflip", kind="ratio", type=float, default=0.1, lo=0.0, hi=1.0,
         group="触发概率", label="垂直翻转概率", help="垂直翻转概率"),
    dict(name="p_rot90", flag="--p-rot90", kind="ratio", type=float, default=0.1, lo=0.0, hi=1.0,
         group="触发概率", label="90°旋转概率", help="90°旋转概率"),
    dict(name="p_cutout", flag="--p-cutout", kind="ratio", type=float, default=0.2, lo=0.0, hi=1.0,
         group="触发概率", label="随机擦除概率", help="随机擦除概率(只擦背景)"),
    dict(name="p_sharpen", flag="--p-sharpen", kind="ratio", type=float, default=0.15,
         lo=0.0, hi=1.0, group="触发概率", label="锐化概率",
         help="锐化概率(0 关闭)，模拟对焦波动，对金属反光表面有效"),
    dict(name="p_saltpepper", flag="--p-saltpepper", kind="ratio", type=float, default=0.10,
         lo=0.0, hi=1.0, group="触发概率", label="椒盐噪声概率",
         help="椒盐噪声概率(0 关闭)，只污染目标区域，模拟传感器坏点"),
    dict(name="p_occlusion", flag="--p-occlusion", kind="ratio", type=float, default=0.0,
         lo=0.0, hi=1.0, group="触发概率", label="遮挡概率",
         help="遮挡增强概率(0 关闭)，盖住目标一部分以提升遮挡鲁棒性"),

    # ---- 新增强的强度参数 ----
    dict(name="sharpen_amount", flag="--sharpen-amount", kind="ratio", type=float, default=0.6,
         lo=0.0, hi=1.0, group="增强强度", label="锐化强度",
         help="锐化强度 0~1（越小越温和，金属反光建议 0.3~0.5）"),
    dict(name="saltpepper_ratio", flag="--saltpepper-ratio", kind="ratio", type=float, default=0.02,
         lo=0.0, hi=0.2, group="增强强度", label="椒盐噪点比例",
         help="噪点占目标区域像素的比例（0.02 = 2%）"),
    dict(name="salt_ratio", flag="--salt-ratio", kind="ratio", type=float, default=0.5,
         lo=0.0, hi=1.0, group="增强强度", label="白点占比",
         help="噪点里白点(salt)的占比，其余为黑点(pepper)"),
    dict(name="occlusion_max", flag="--occlusion-max", kind="ratio", type=float, default=0.35,
         lo=0.0, hi=1.0, group="增强强度", label="遮挡块上限",
         help="单个目标最多被遮掉的面积比例（0 = 不遮挡；越大遮得越狠）"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    """按 PARAM_SPEC 生成 argparse 解析器（CLI 与 GUI 共用同一份定义）。"""
    p = argparse.ArgumentParser(
        description="YOLO 离线数据增强 / 扩充数据集（detection / segment / obb）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    groups: Dict[str, argparse._ArgumentGroup] = {}
    for spec in PARAM_SPEC:
        gname = spec.get("group", "")
        if gname:
            target = groups.get(gname) or groups.setdefault(gname, p.add_argument_group(gname))
        else:
            target = p
        kw = dict(dest=spec["name"], help=spec["help"])
        if spec["kind"] == "bool":
            kw.update(action="store_true", default=spec["default"])
        else:
            kw.update(type=spec["type"], default=spec["default"], metavar=spec["name"].upper())
            if spec.get("choices"):
                kw["choices"] = spec["choices"]
            if spec.get("required"):
                kw["required"] = True
        target.add_argument(spec["flag"], **kw)
    # 旧参数兼容：--src 会被翻译成 --image-path / --label-path（见 parse_args）
    p.add_argument("--src", dest="src", default=None, metavar="SRC",
                   help="[已弃用] 旧的数据集根目录，等价于同时指定 images 与 labels")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    # 兼容旧的 --src 写法：从根目录推出图片/标签两个路径
    if args.src:
        if args.image_path and args.label_path:
            raise SystemExit("--src 与 --image-path/--label-path 不能同时使用")
        try:
            img, lbl = derive_from_src(args.src)
        except ValueError as e:
            raise SystemExit(str(e))
        args.image_path, args.label_path = img, lbl
        print(f"! --src 已弃用，已自动转换为：\n"
              f"    --image-path \"{img}\"\n    --label-path \"{lbl}\"", file=sys.stderr)
    missing = [f for f, v in (("--image-path", args.image_path),
                              ("--label-path", args.label_path)) if not v]
    if missing:
        raise SystemExit("缺少必填参数: " + ", ".join(missing))
    return args


def cfg_from_values(v: Dict[str, object], layout: Optional[Layout] = None,
                    task: str = "auto") -> AugCfg:
    """从参数字典构造 AugCfg（GUI 用；缺省值取 PARAM_SPEC 的默认）。

    若 task=pose，会从数据集的 data.yaml 读取 kpt_shape 与 flip_idx
    （这两个决定关键点格式和翻转语义，必须来自数据集本身而非界面参数）。
    """
    d = {s["name"]: s["default"] for s in PARAM_SPEC}
    d.update({k: val for k, val in v.items() if k in d})
    cfg = AugCfg(
        degrees=float(d["degrees"]), scale=float(d["scale"]), translate=float(d["translate"]),
        shear=float(d["shear"]), perspective=float(d["perspective"]),
        p_mosaic=float(d["p_mosaic"]), p_copypaste=float(d["p_copypaste"]),
        p_hflip=float(d["p_hflip"]), p_vflip=float(d["p_vflip"]),
        p_rot90=float(d["p_rot90"]), p_cutout=float(d["p_cutout"]),
        # 新增：锐化 / 椒盐 / 遮挡
        p_sharpen=float(d["p_sharpen"]), sharpen_amount=float(d["sharpen_amount"]),
        p_saltpepper=float(d["p_saltpepper"]), saltpepper_ratio=float(d["saltpepper_ratio"]),
        salt_ratio=float(d["salt_ratio"]),
        p_occlusion=float(d["p_occlusion"]), occlusion_max=float(d["occlusion_max"]),
        min_side=float(d["min_side"]),
    )
    if task == "pose" and layout is not None:
        root = layout.root
        try:
            ti, tl = layout.train_dirs()
            items = list_items(ti, tl)
            cfg.kpt_shape = detect_kpt_shape(root, items)
        except (ValueError, OSError):
            cfg.kpt_shape = (0, 0)
        cfg.flip_idx = load_flip_idx(root)
        if cfg.flip_idx is not None and cfg.kpt_shape[0] and \
                len(cfg.flip_idx) != cfg.kpt_shape[0]:
            raise ValueError(
                f"data.yaml 的 flip_idx 长度({len(cfg.flip_idx)}) 与"
                f" kpt_shape 关键点数量({cfg.kpt_shape[0]}) 不一致")
    return cfg


def resolve_source(image_path, label_path, task: str = "auto"
                   ) -> Tuple[List[Item], str, List[str], bool]:
    """读取数据集：返回 train 图片列表、任务类型、类别名、是否已有 train/val 划分。

    供 GUI 做统计/预览用；不写任何文件。失败时抛 ValueError。
    """
    layout = build_layout(image_path, label_path)
    ti, tl = layout.train_dirs()
    items = list_items(ti, tl)
    if not items:
        raise ValueError(f"在 {ti} 里没有找到图片（支持 .jpg/.png/.bmp 等）")
    check_pairing(items, ti, tl)
    t = task
    if t == "auto":
        try:
            t = autodetect_task(layout.root, items)
        except SystemExit as e:
            raise ValueError(str(e).strip())
    try:
        names = load_names(layout.root, layout.classes_file)
    except SystemExit as e:
        raise ValueError(str(e).strip())
    return items, t, names, layout.has_splits


def source_exists(image_path, label_path) -> bool:
    """两个输入目录是否都存在（支持相对路径回退）。"""
    if not image_path or not label_path:
        return False
    try:
        build_layout(image_path, label_path)
        return True
    except (ValueError, OSError):
        return False


def check_pairing(items: List[Item], image_dir: Path, label_dir: Path,
                  allow_missing: bool = False) -> None:
    """检查图片与标签是否成对，数量严重不符时直接报错。

    图片远多于标签是最常见的误操作（例如标签存在别的目录、或只标了一部分图）。
    这时如果不报错，会生成一堆"空标签图"，训练出来全是虚警。
    allow_missing=True 时只做提示，不拦截（用于确实存在部分未标注图的情况）。
    """
    n_img = len(items)
    n_lbl = sum(1 for it in items if it.lbl_path.exists())
    if n_img == 0:
        return
    if n_lbl == 0:
        raise ValueError(
            f"在 {label_dir} 里没有找到任何 .txt 标签（图片有 {n_img} 张）。\n"
            f"请确认标签路径是否正确：标签文件名应与图片同名（xxx.jpg -> xxx.txt）。")
    missing = n_img - n_lbl
    if missing > int(n_img * 0.05):
        sample = [it.img_path.name for it in items if not it.lbl_path.exists()][:3]
        msg = (f"图片与标签数量不匹配：图片 {n_img} 张，但只有 {n_lbl} 个标签（缺 {missing} 个）。\n"
               f"例如这些图片没有对应标签：{', '.join(sample)}\n"
               f"请检查：\n"
               f"  - 标签目录是否填对（图片 {image_dir} 对应标签 {label_dir}）\n"
               f"  - 是否只标注了部分图片\n"
               f"如果确实允许部分图片没有目标，请加 --allow-missing-labels 参数。")
        if not allow_missing:
            raise ValueError(msg)
        print(f"! {msg.splitlines()[0]}")
        print("  （已按 --allow-missing-labels 继续，缺少标签的图片会生成空标签）")


def guess_label_path(image_path) -> Optional[Path]:
    """从图片路径推测标签路径：把路径里的 `images` 段换成 `labels`。

    只在路径中确实含 `images` 目录时生效（YOLO 的标准布局），否则返回 None，
    让调用方提示用户手动指定 —— 不猜别的目录名，避免指到错误位置。
    """
    p = Path(str(image_path)).expanduser()
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            new = Path(*parts[:i]) / "labels"
            for rest in parts[i + 1:]:
                new = new / rest
            return new
    return None


def default_out_for(image_path) -> Path:
    """推导默认输出目录。

    输入可能是 `.../images`（数据集根下）也可能是 `.../images/train`（某个 split 内）。
    为了输出到用户直觉上的位置，规则是：
        - 路径本身叫 images -> 输出到它的父目录旁（.../U2_aug）
        - 路径是 images/xxx  -> 同样回到 images 的父目录旁（.../U2_aug）
        - 其它情况          -> <该目录名>_aug
    """
    s = _resolve_user_path(str(image_path))
    if s.name.lower() == "images":
        base = s.parent
    elif s.parent.name.lower() == "images":
        base = s.parent.parent
    else:
        base = s
    return base.with_name(base.name + "_aug")


def preview_image(image_path, label_path, values: Dict[str, object],
                  index: int = 0) -> np.ndarray:
    """按当前参数生成一张增强示例图（BGR，已画好标注），用于 GUI 即时预览。"""
    items, task, names, has_splits = resolve_source(
        image_path, label_path, str(values.get("task", "auto")))
    layout = build_layout(image_path, label_path)
    cfg = cfg_from_values(values, layout, task)
    rng = random.Random(int(values.get("seed", 0) or 0))
    if not 0 <= index < len(items):
        raise ValueError(f"索引 {index} 超出范围（共 {len(items)} 张）")
    img, insts = build_augmented(index, items, cfg, task, rng)
    vis = img.copy()
    h, w = vis.shape[:2]
    for it in insts:
        draw_instance(vis, it, w, h, names)
    cv2.putText(vis, f"{items[index].img_path.name}  [{task}]", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _resolve_user_path(p: str, anchors: Optional[Sequence[Path]] = None) -> Path:
    """把用户给的路径解析成绝对路径。

    先按当前工作目录解析；若不存在，再依次尝试脚本所在目录及其父目录
    （本脚本被移动到 augment_dataset/ 子目录后，`--src datasets/U2` 这类
    相对 ultralytics 根目录的写法仍然要能用）。
    """
    raw = Path(p).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cand = Path.cwd() / raw
    if cand.exists():
        return cand.resolve()
    for a in (anchors or SCRIPT_ANCHORS):
        c = a / raw
        if c.exists():
            return c.resolve()
    return (Path.cwd() / raw).resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ip_raw, lp_raw = str(args.image_path), str(args.label_path)
    try:
        layout = build_layout(ip_raw, lp_raw)
    except ValueError as e:
        raise SystemExit(str(e))
    image_path, label_path = layout.image_path, layout.label_path
    out = _resolve_user_path(str(args.out)) if args.out else default_out_for(image_path)

    if args.times < 1:
        raise SystemExit("--times 至少为 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio 必须在 (0, 1) 之间")

    # dry-run 只统计，不碰输出目录，因此不必要求 --overwrite
    if not args.dry_run and out.exists():
        if not args.overwrite:
            raise SystemExit(f"输出目录已存在: {out}\n加 --overwrite 覆盖，或换 --out")
        shutil.rmtree(out)

    rng = random.Random(args.seed)

    # 目录结构：图片与标签各自独立；都支持 train/val 子目录，也支持平铺
    tr_imgs, tr_lbls = layout.train_dirs()
    vd = layout.val_dirs()
    if vd is not None:
        va_imgs, va_lbls = vd
    else:
        va_imgs, va_lbls = None, None
        print("! 未发现 train/val 子目录，按平铺数据集处理：全部作为 train（稍后按比例切 val）")

    train_items = list_items(tr_imgs, tr_lbls)
    if not train_items:
        raise SystemExit(f"没有找到图片: {tr_imgs}")
    try:
        check_pairing(train_items, tr_imgs, tr_lbls, args.allow_missing_labels)
    except ValueError as e:
        raise SystemExit(str(e))
    val_items = list_items(va_imgs, va_lbls) if va_imgs and va_imgs.is_dir() else []

    if not val_items:
        # 源数据集没有 val：从 train 里按比例切一份出来。
        # 绝不能把 train 的所有图片都当成 val —— 那样验证指标会因数据泄漏而虚高。
        split_rng = random.Random(args.seed)
        shuffled = list(train_items)
        split_rng.shuffle(shuffled)
        k = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * args.val_ratio))))
        val_items = shuffled[:k]
        train_items = shuffled[k:]
        print(f"! 源数据集没有 val，已从 train 按 {1 - args.val_ratio:.0%}/{args.val_ratio:.0%} 重新划分："
              f"train {len(train_items)} / val {len(val_items)}（val 不参与增强）")

    task = args.task
    if task == "auto":
        task = autodetect_task(layout.root, train_items)
        print(f"* 自动识别任务: {task}")

    # cfg 必须在确定 task 之后构造：pose 需要从数据集读取 kpt_shape / flip_idx
    try:
        cfg = cfg_from_values({s["name"]: getattr(args, s["name"]) for s in PARAM_SPEC},
                              layout, task)
    except ValueError as e:
        raise SystemExit(str(e))

    names = args.names.split(",") if args.names else load_names(layout.root, layout.classes_file)
    print(f"* 图片路径: {image_path}")
    print(f"* 标签路径: {label_path}")
    if layout.yaml_file:
        print(f"* 配置文件: {layout.yaml_file}")
    print(f"* 输出目录: {out}")
    print(f"* 任务/类别: {task} / {names}")
    if task == "pose":
        if cfg.kpt_shape[0] <= 0:
            raise SystemExit(
                "关键点任务需要确定关键点数量：请在 data.yaml 里写 kpt_shape: [K, 3]，\n"
                "例如 kpt_shape: [5, 3]（K 个关键点，每个 3 维 = x,y,可见性）")
        print(f"* 关键点: kpt_shape={list(cfg.kpt_shape)}  "
              f"flip_idx={cfg.flip_idx if cfg.flip_idx else '未定义'}")
        if cfg.flip_idx is None and (cfg.p_hflip > 0 or cfg.p_vflip > 0):
            print("  ! data.yaml 未定义 flip_idx：关键点左右翻转时不会交换编号，"
                  "对称关键点可能学错。建议在 yaml 里加 flip_idx，或把翻转概率设为 0。")
    print(f"* train 图片: {len(train_items)}  val 图片: {len(val_items)}  times={args.times}")

    # 统计原数据集实例数
    before: Dict[int, int] = {}
    for it in train_items:
        if not it.lbl_path.exists():
            continue
        for line in it.lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if parts and parts[0].isdigit():
                c = int(parts[0])
                before[c] = before.get(c, 0) + 1
    print(f"* 原 train 类别实例数: {dict(sorted(before.items()))}")

    # ---- 试跑/预览：先验证标注是否对齐 ----
    if args.dry_run:
        print("dry-run：不写文件")
        return 0

    if args.preview:
        pv = out / "preview"
        pv.mkdir(parents=True, exist_ok=True)
        for i in range(min(args.preview, len(train_items))):
            write_preview(train_items[i], pv / f"preview_{i:02d}.jpg", task, cfg,
                          train_items, rng, i)
        print(f"* 预览图已写入: {pv}")

    t0 = time.time()
    inc = not args.no_original
    cnt, _ = augment_split(train_items, train_items, out, "train", task, cfg,
                           args.times, inc, rng, args.limit)
    n = copy_items(val_items, out, "val")
    print(f"* val 已原样复制 {n} 张（不增强验证集）")

    # ---- data.yaml ----
    # 必须写绝对路径：Ultralytics 8.3.x 在 path 为空时会在扫描/缓存的路径拼接上出问题
    # （实测 path:'' 会导致训练卡住不结束）。绝对路径是验证过的可靠写法。
    # 代价是输出目录改名后 data.yaml 会失效 —— 所以README里提醒：改名后重跑一次，
    # 或直接重新指定 --out。
    import yaml
    posix = str(out).replace("\\", "/")
    data = {
        "path": posix,
        "train": f"{posix}/images/train",
        "val": f"{posix}/images/val",
        "test": "",
        "names": {i: str(nm) for i, nm in enumerate(names)},
    }
    # pose 数据集必须带 kpt_shape，否则 Ultralytics 直接报错；flip_idx 决定翻转语义
    if task == "pose":
        data["kpt_shape"] = [int(cfg.kpt_shape[0]), int(cfg.kpt_shape[1])]
        if cfg.flip_idx:
            data["flip_idx"] = [int(v) for v in cfg.flip_idx]
    (out / "data.yaml").write_text(
        "# generated by augment_dataset.py: train = originals + offline-augmented, val = unaugmented\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    print(f"\n完成：{cnt} 张增强图 + {'原图 ' if inc else ''}val {n} 张，用时 {time.time() - t0:.1f}s")
    print(f"训练用: {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(130)
