#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键打包成 Windows exe
======================

用法（在任意目录执行均可）：

    E:\\anaconda\\anaconda\\envs\\yolov11\\python.exe build_exe.py

产物：dist/YOLO数据集增强/YOLO数据集增强.exe（目录版，启动快）
      加 --onefile 可打成单文件 exe（启动略慢，约多 2~4 秒解包）

说明
----
本工具只依赖 opencv / numpy / tkinter，**不依赖 torch 与 ultralytics**，
所以体积可以控制在几十 MB，而不是几个 GB。

注意：不要用 `--onefile` + `--windowed` 之外的奇怪组合；如果杀毒软件误报，
目录版（onedir）比单文件版更不容易被拦。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_NAME = "YOLO数据集增强"

# 确定用不到的重量级依赖，显式排除以缩小体积、加快启动
EXCLUDES = [
    "torch", "torchvision", "ultralytics",
    "matplotlib", "scipy", "pandas", "PIL.ImageQt",
    "IPython", "jupyter", "notebook", "nbformat",
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "pytest", "setuptools", "pip", "wheel",
    "imageio", "skimage", "sklearn", "sympy", "onnx", "onnxruntime",
    "tensorflow", "paddle", "jax",
]


def _native_dll_args() -> list[str]:
    """收集 conda 环境下 PyInstaller 漏掉的原生 DLL。

    踩过两个坑，都是"打包成功、运行才崩溃"：
      1. tkinter：缺 tcl86t.dll / tk86t.dll
         -> ImportError: DLL load failed while importing _tkinter
      2. ctypes ：缺 ffi-*.dll（libffi）
         -> ImportError: DLL load failed while importing _ctypes
            后果是 ctypes 整个不可用 => 无法声明 DPI 感知 => exe 里字体变小，
            而且这个错误是静默的（被 try/except 吞掉），极难发现。
    这两类问题在源码运行时完全看不出来，只有 exe 才暴露。
    """
    args: list[str] = []
    base = Path(sys.base_prefix)
    lib_bin = base / "Library" / "bin"

    # tkinter 的 tcl/tk 运行库
    for name in ("tcl86t.dll", "tk86t.dll"):
        p = lib_bin / name
        if p.exists():
            # 目标写 "."，让 DLL 落在 exe 旁边，Windows 才能按默认顺序找到
            args += ["--add-binary", f"{p}{os.pathsep}."]

    # ctypes 依赖的 libffi（不同版本文件名不同：ffi-7 / ffi-8 / ffi.dll）
    seen: set[Path] = set()
    for d in (base / "DLLs", lib_bin, base):
        if not d.is_dir():
            continue
        for pat in ("ffi*.dll", "libffi*.dll"):
            for p in sorted(d.glob(pat)):
                if p in seen:
                    continue
                seen.add(p)
                args += ["--add-binary", f"{p}{os.pathsep}."]

    # tcl/tk 数据目录（不同 PyInstaller 版本行为不一致，显式带上更稳）
    for d in ("tcl8.6", "tk8.6"):
        src = base / "Library" / "lib" / d
        if src.is_dir():
            args += ["--add-data", f"{src}{os.pathsep}lib/{d}"]
    return args


def _slim_bundle(out_dir: Path) -> int:
    """打包后删除用不到的大文件，缩小体积。返回删除的字节数。

    只删明确安全的两类：
      - opencv_videoio_ffmpeg*.dll : 视频编解码，本项目只处理图片
    这些文件即使被误删也不影响 imread/imdecode/imencode，
    但如果将来要用 cv2.VideoCapture 就必须保留。
    """
    if not out_dir.is_dir():
        return 0
    freed = 0
    for pat in ("opencv_videoio_ffmpeg*.dll",):
        for p in out_dir.rglob(pat):
            try:
                freed += p.stat().st_size
                p.unlink()
            except OSError:
                pass
    return freed


def build(onefile: bool, console: bool, clean: bool) -> int:
    if not (HERE / "app_launcher.py").exists():
        print("找不到 app_launcher.py", file=sys.stderr)
        return 1

    if clean:
        for d in ("build", "dist"):
            p = HERE / d
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(HERE / "app_launcher.py"),
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        # 把项目源码目录加入搜索路径，保证 augment_gui / augment_dataset 能被找到
        "--paths", str(HERE),
    ]
    # 关键模块显式收集（避免被上面的 excludes 误伤）
    for mod in ("augment_gui", "augment_dataset"):
        cmd += ["--hidden-import", mod]

    cmd.append("--onefile" if onefile else "--onedir")
    cmd.append("--console" if console else "--windowed")
    for ex in EXCLUDES:
        cmd += ["--exclude-module", ex]
    # conda 环境必须补上 tcl/tk 的 DLL，否则 _tkinter 加载失败
    cmd += _native_dll_args()

    print("执行:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(HERE))
    if r.returncode != 0:
        print("打包失败", file=sys.stderr)
        return r.returncode

    if onefile:
        out_dir = HERE / "dist" / (APP_NAME + ".exe")
    else:
        out_dir = HERE / "dist" / APP_NAME
    print("\n打包完成:")
    print("  ", out_dir)
    if not onefile:
        freed = _slim_bundle(out_dir)
        if freed:
            print(f"   已剔除视频编解码依赖，节省约 {freed / 1024 / 1024:.1f} MB")
        print(f"   启动文件: {out_dir / (APP_NAME + '.exe')}")
        # 顺手复制一份 README，方便使用者了解用法
        rd = HERE / "README.md"
        if rd.exists():
            shutil.copy2(rd, out_dir / "README.md")
            print("   已附带 README.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="把数据增强界面打包成 Windows exe")
    ap.add_argument("--onefile", action="store_true", help="打成单个 exe（启动较慢）")
    ap.add_argument("--console", action="store_true",
                    help="保留控制台窗口（调试用，能看到报错）")
    ap.add_argument("--no-clean", action="store_true", help="不清理旧的 build/dist")
    a = ap.parse_args()
    return build(a.onefile, a.console, not a.no_clean)


if __name__ == "__main__":
    sys.exit(main())
