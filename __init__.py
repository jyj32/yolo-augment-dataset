# -*- coding: utf-8 -*-
"""
YOLO 数据集离线增强工具包
=========================

本目录集中放置与数据增强相关的全部脚本，可整体拷贝到任意位置使用。

模块
----
augment_dataset.py
    离线数据增强 / 扩充数据集（detection / segment / obb），核心实现。
    所有参数定义集中在 PARAM_SPEC，命令行与图形界面共用这一份定义。

augment_gui.py
    augment_dataset.py 的 tkinter 图形界面（滑块调节参数 + 实时预览）。

json2yolo.py
    X-AnyLabeling / LabelMe 的 JSON 标注 -> YOLO txt 标签转换器。

run_gui.pyw
    双击启动的入口（不弹控制台窗口，并自动切换到有 cv2 的解释器）。

典型用法
--------
命令行：

    python augment_dataset.py --src datasets/U2 --task detection --times 3

图形界面（推荐给不熟悉参数的人）：

    python augment_gui.py
    # 或直接双击 run_gui.pyw

标注转换：

    python json2yolo.py --src "E:/yolov11/data/某个数据集" --task detection

注意
----
相对路径会按「当前目录 -> 本目录 -> 上级目录 -> 祖父目录」依次尝试，
因此 `--src datasets/U2` 在 ultralytics 根目录下运行依然有效。
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "augment_dataset",
    "augment_gui",
    "json2yolo",
]

# 对外暴露最常用的几个符号，便于 `from augment_dataset import ...`
#
# 注意：这里同时导出 main（原名）和 augment_main（别名）。
# 原因：本包名与内部模块同名（augment_dataset/ 与 augment_dataset.py）。
# 在 exe（PyInstaller）或"从父目录导入包"的场景下，`import augment_dataset`
# 拿到的是本文件而不是那个脚本模块，此时 GUI 需要 AD.main 才能工作；
# 只导出 augment_main 会导致运行时报
#   AttributeError: module 'augment_dataset' has no attribute 'main'
try:  # pragma: no cover - 导入失败不应阻塞脚本式使用
    from .augment_dataset import (  # noqa: F401
        AugCfg,
        PARAM_SPEC,
        build_arg_parser,
        cfg_from_values,
        default_out_for,
        preview_image,
        resolve_source,
        source_exists,
    )
    from .augment_dataset import main  # noqa: F401
    from .augment_dataset import main as augment_main  # noqa: F401
except ImportError:
    # 以脚本方式直接运行本目录下的文件时（没有包上下文），忽略即可
    pass
