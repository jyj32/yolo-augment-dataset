#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双击启动入口（.pyw 无控制台）
=============================

直接双击本文件即可打开增强参数界面。之所以需要这个启动器，是因为
Windows 的 .pyw 关联通常指向 `C:\\Windows\\pyw.exe`，它默认使用系统自带的
Python（例如 C:\\Python314），而那里没有安装 opencv / numpy / ultralytics，
直接运行 augment_gui.py 会立刻闪退。

本启动器会做三件事：
    1. 若当前解释器缺少 cv2/numpy/tkinter，自动寻找一个装好的解释器
       （优先 conda 环境 yolov11、其次 PATH 上的 python、再用 py 启动器）；
    2. 用它重新启动 augment_gui.py；
    3. 全部失败时弹窗提示该装什么，而不是静默闪退。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUI = HERE / "augment_gui.py"


def _set_dpi_aware_env() -> None:
    """让子进程在启动时声明 DPI 感知。

    Tk 必须"在创建窗口之前"声明 DPI 感知才有效，而本启动器常常是用另一个
    解释器重新拉起 augment_gui.py 的，所以这里通过环境变量把意图传下去，
    augment_gui.py 的 main() 会读取它并尽早调用 SetProcessDpiAwareness。
    （augment_gui.py 自己也会尝试声明，这里只是确保子进程不会漏掉。）
    """
    os.environ.setdefault("DSH_AUGMENT_DPI_AWARE", "1")


def _enable_dpi_awareness_here() -> None:
    """本进程也声明一次，避免启动器自己的提示框被系统位图拉伸。"""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass


_set_dpi_aware_env()
_enable_dpi_awareness_here()


def _has_deps(exe: str) -> bool:
    """用指定解释器检查是否具备运行界面所需的依赖。"""
    try:
        r = subprocess.run(
            [exe, "-c", "import cv2, numpy, tkinter, PIL"],
            capture_output=True, timeout=30, text=True,
            encoding="utf-8", errors="replace",
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def candidate_interpreters() -> list[str]:
    """按优先级列出可能可用的 Python 解释器。"""
    cands: list[str] = []

    # 1) 当前解释器（若已经是正确的那个，就不用重启）
    cands.append(sys.executable)

    # 2) 本项目实际使用的 conda 环境
    envd = Path(os.environ.get("CONDA_PREFIX", "")) if os.environ.get("CONDA_PREFIX") else None
    if envd and (envd / "python.exe").exists():
        cands.append(str(envd / "python.exe"))
    for base in (r"E:\anaconda\anaconda\envs\yolov11",
                 r"C:\Users\lenovo\.conda\envs\yolov11",
                 r"E:\anaconda\anaconda"):
        exe = Path(base) / "python.exe"
        if exe.exists():
            cands.append(str(exe))

    # 3) py 启动器登记的其它版本（通过 py -0p 查询）
    try:
        r = subprocess.run(["py", "-0p"], capture_output=True, timeout=20,
                           text=True, encoding="utf-8", errors="replace")
        for line in (r.stdout or "").splitlines():
            parts = line.split()
            for p in parts:
                if p.lower().endswith(".exe"):
                    cands.append(p)
                    break
    except (OSError, subprocess.SubprocessError):
        pass

    # 4) PATH 上的 python
    for name in ("python.exe", "python3.exe"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if not d:
                continue
            exe = Path(d) / name
            if exe.exists():
                cands.append(str(exe))

    # 去重保序
    seen = set()
    out = []
    for c in cands:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def main() -> int:
    if not GUI.exists():
        _alert(f"找不到 {GUI}\n请确认 augment_gui.py 与本启动器在同一目录。")
        return 1

    for exe in candidate_interpreters():
        if _has_deps(exe):
            if Path(exe).resolve() != Path(sys.executable).resolve():
                # 用找到的正确解释器重新拉起界面（无控制台窗口）
                try:
                    subprocess.Popen([exe, str(GUI)], cwd=str(HERE), close_fds=True)
                    return 0
                except OSError:
                    continue
            else:
                # 当前解释器就合适，直接导入运行
                os.chdir(HERE)
                sys.path.insert(0, str(HERE))
                import runpy
                runpy.run_path(str(GUI), run_name="__main__")
                return 0

    _alert(
        "没有找到装好依赖的 Python 环境。\n\n"
        "需要 opencv、numpy、Pillow、tkinter、ultralytics。\n"
        "可在 Anaconda Prompt 里执行：\n\n"
        "    conda activate yolov11\n"
        "    pip install opencv-python numpy pillow\n\n"
        "然后重新双击 run_gui.pyw。\n\n"
        "或改用命令行启动：\n"
        "    conda activate yolov11\n"
        "    python augment_gui.py"
    )
    return 1


def _alert(msg: str) -> None:
    """尽力弹窗；控制台不可用时退化为打印。"""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("增强界面启动失败", msg)
        root.destroy()
    except Exception:  # noqa: BLE001
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
