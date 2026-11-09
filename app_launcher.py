#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exe 打包入口
============

这是给 PyInstaller 用的启动入口，正常开发时不要直接运行它（用 augment_gui.py）。

它做三件事：
    1. 尽早声明 DPI 感知（必须在创建 Tk 窗口之前，否则打开文件对话框后字体变小）；
    2. 把工作目录切到 exe 所在目录，让相对路径（如 datasets/U2）符合用户直觉；
    3. 调用 augment_gui.main()。

打包命令见同目录的 build_exe.py。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _base_dir() -> Path:
    """exe 运行时返回 exe 所在目录；源码运行时返回本文件目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    # 1) DPI 感知：越早越好，必须在 import tkinter 创建窗口前
    if sys.platform.startswith("win"):
        try:
            import ctypes
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):
                pass
        except Exception:  # noqa: BLE001
            pass

    base = _base_dir()
    # 2) 让相对路径相对 exe 所在目录，而不是快捷方式的"起始位置"
    try:
        os.chdir(base)
    except OSError:
        pass
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    # 3) 启动界面
    import augment_gui
    return augment_gui.main()


if __name__ == "__main__":
    sys.exit(main())
