#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
X-AnyLabeling / LabelMe 的 JSON 标注  ->  YOLO 标签(txt) 转换器
=====================================================================

X-AnyLabeling 保存的是 `图片名.json`（和图片放在同一个目录里），而 YOLO 需要
`labels/xxx.txt`。本脚本负责这一步，支持三种 shape_type：

    rectangle  -> 检测框     直接取 4 个角点 -> cls xc yc w h
    rotation   -> 旋转框     取 4 个角点     -> cls x1 y1 ... x4 y4   (OBB)
    polygon    -> 多边形     取全部顶点      -> cls x1 y1 ...          (分割)

用法
----
    # 预览：只看统计，不写文件
    python json2yolo.py --src "E:/yolov11/data/沿浦零件2_目标检测区分数量" --task detection --dry-run

    # 正式转换（images/*.json -> labels/*.txt）
    python json2yolo.py --src "E:/yolov11/data/沿浦零件2_目标检测区分数量" --task detection

    # 已有 labels 时按 train/val 划分
    python json2yolo.py --src <数据集目录> --task detection --split 0.8

说明
----
* 类别顺序取自 classes.txt；若没有，则按所有 json 中 label 出现顺序自动生成，
  并把生成的 classes.txt 写回，保证训练时索引一致。
* 没有被标注的图片会生成**空 txt**（纯背景图），这对降低误检是有用的，
  真正想排除就加 --skip-unlabeled。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def find_json(img: Path) -> Optional[Path]:
    """在图片目录里找同名 json。"""
    j = img.with_suffix(".json")
    if j.exists():
        return j
    cand = list(img.parent.glob(img.stem + ".json"))
    return cand[0] if cand else None


def load_classes(src: Path, jsons: List[Path], auto_write: bool) -> List[str]:
    ct = src / "classes.txt"
    if ct.exists():
        names = [ln.strip() for ln in ct.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if ln.strip()]
        if names:
            return names
    # 自动收集
    seen: List[str] = []
    for j in jsons:
        try:
            o = json.loads(j.read_text(encoding="utf-8", errors="ignore"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for s in o.get("shapes", []):
            lab = s.get("label")
            if lab and lab not in seen:
                seen.append(lab)
    if seen and auto_write:
        ct.write_text("\n".join(seen) + "\n", encoding="utf-8")
        print(f"* 已自动生成 classes.txt: {seen}")
    return seen


def shape_to_line(shape: dict, w: float, h: float, task: str,
                  class_map: Dict[str, int]) -> Optional[str]:
    st = (shape.get("shape_type") or "polygon").lower()
    label = shape.get("label")
    if label is None or label not in class_map:
        return None
    pts = shape.get("points") or []
    if len(pts) < 3:
        return None
    cls = class_map[label]

    if task == "detection":
        # 任何形状都退化为其外接矩形
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

    if task == "segment":
        if st == "rectangle" or st == "rotation":
            # 框转多边形：按外接矩形 4 点
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            poly = [(float(p[0]), float(p[1])) for p in pts]
        pairs = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in poly)
        return f"{cls} {pairs}"

    if task == "obb":
        if st == "rotation" and len(pts) >= 4:
            quad = [(float(p[0]), float(p[1])) for p in pts[:4]]
        elif st == "rectangle":
            x1, y1, x2, y2 = (float(pts[0][0]), float(pts[0][1]),
                              float(pts[2][0]), float(pts[2][1]))
            quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:  # polygon -> 最小外接旋转框
            import cv2
            import numpy as np
            arr = np.array(pts, dtype="float32")
            rect = cv2.minAreaRect(arr)
            quad = [tuple(map(float, p)) for p in cv2.boxPoints(rect)]
        pairs = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in quad)
        return f"{cls} {pairs}"

    raise ValueError(f"未知 task: {task}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="X-AnyLabeling/LabelMe JSON -> YOLO txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src", required=True, help="数据集目录（含 images/ 与 classes.txt）")
    ap.add_argument("--task", default="detection", choices=["detection", "segment", "obb"])
    ap.add_argument("--img-dir", default="images", help="图片子目录名")
    ap.add_argument("--lbl-dir", default="labels", help="输出标签子目录名")
    ap.add_argument("--split", type=float, default=0.0,
                    help=">0 时按该比例划分 train/val（labels/train、labels/val）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-unlabeled", action="store_true",
                    help="没有 json 的图片直接跳过，而不是生成空标签")
    ap.add_argument("--no-classes-write", action="store_true", help="不自动写 classes.txt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src).resolve()
    img_dir = src / args.img_dir
    if not img_dir.is_dir():
        raise SystemExit(f"找不到图片目录: {img_dir}")

    images = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT],
                    key=lambda p: p.name)
    if not images:
        raise SystemExit(f"{img_dir} 里没有图片")

    jsons = [find_json(p) for p in images]
    found = [j for j in jsons if j is not None]
    print(f"* 图片 {len(images)} 张，其中带 json 标注的 {len(found)} 张"
          f"，未标注 {len(images) - len(found)} 张")

    classes = load_classes(src, found, auto_write=not args.no_classes_write)
    if not classes:
        raise SystemExit("没有类别信息：请提供 classes.txt，或先在 X-AnyLabeling 里标几个框")
    class_map = {n: i for i, n in enumerate(classes)}
    print(f"* 类别: {['%d:%s' % (i, n) for i, n in enumerate(classes)]}")

    if len(found) == 0:
        raise SystemExit("一张标注都没有，无法转换。请先在 X-AnyLabeling 里标注（详见 --help 上方说明）")

    # ---- 生成 ----
    lbl_root = src / args.lbl_dir
    rows: List[Tuple[Path, List[str]]] = []
    inst_cnt: Counter = Counter()
    bad = 0
    for img, jp in zip(images, jsons):
        lines: List[str] = []
        if jp is not None:
            try:
                obj = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"! 解析失败 {jp.name}: {e}")
                bad += 1
                continue
            w = float(obj.get("imageWidth") or 0)
            h = float(obj.get("imageHeight") or 0)
            if w <= 0 or h <= 0:
                import cv2
                import numpy as np
                arr = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    print(f"! 读图失败 {img.name}")
                    bad += 1
                    continue
                h, w = arr.shape[:2]
            for s in obj.get("shapes", []):
                if s.get("label") not in class_map:
                    print(f"! {jp.name}: 类别 '{s.get('label')}' 不在 classes.txt 中，已跳过")
                    continue
                line = shape_to_line(s, w, h, args.task, class_map)
                if line:
                    lines.append(line)
                    inst_cnt[int(line.split()[0])] += 1
        elif not args.skip_unlabeled:
            pass  # 生成空标签（纯背景图）

        rows.append((img, lines))

    print(f"* 待写标签文件 {len(rows)} 个，目标实例数 {dict(sorted(inst_cnt.items()))}"
          + (f"，解析失败 {bad}" if bad else ""))
    if args.dry_run:
        print("dry-run：不写文件")
        return 0

    def write(imgs: List[Path], sub: str) -> None:
        (lbl_root / sub).mkdir(parents=True, exist_ok=True)
        for im in imgs:
            lines = next(l for p, l in rows if p == im)
            (lbl_root / sub / (im.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if args.split > 0:
        shuffled = list(images)
        random.Random(args.seed).shuffle(shuffled)
        k = max(1, int(round(len(shuffled) * args.split)))
        write(shuffled[:k], "train")
        write(shuffled[k:], "val")
        print(f"* 已划分: train {k} / val {len(shuffled) - k}（标签已写到 labels/train、labels/val）")
        print(f"* 注意：图片仍平铺在 images/ 里；训练前请按 split 把它们也分到 images/train、images/val")
    else:
        write(images, "")
        print(f"* 标签已写入: {lbl_root}")

    # 顺手写一份 data.yaml（平铺结构）
    if args.split == 0:
        import yaml
        out = {"path": str(src).replace("\\", "/"),
               "train": f"{str(src).replace(chr(92), '/')}/{args.img_dir}",
               "val": f"{str(src).replace(chr(92), '/')}/{args.img_dir}",
               "test": "",
               "names": {i: n for i, n in enumerate(classes)}}
        (src / "data.yaml").write_text(
            "# generated by json2yolo.py\n" + yaml.safe_dump(out, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        print(f"* 已写 data.yaml: {src / 'data.yaml'}（train/val 先指向同一目录，训练前请划分）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
