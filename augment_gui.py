#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强参数辅助界面（tkinter）
================================

给 `augment_dataset.py` 的所有参数做一个图形界面：

    * 左侧：所有参数（滑块 + 输入框 + 复选框 + 下拉），穷举增强强度 / 触发概率
    * 右侧：实时预览 —— 点一下就立刻生成一张增强示例图并画好标注，确认框没跑偏
    * 底部：自动生成命令行、dry-run 统计、开始前先问你是否覆盖、后台线程跑增强
    * 预设：小数据集 / 工业零件 / 保守 / 关闭几何 等多套一键配置

运行
----
    python augment_gui.py
    # 或直接双击（.pyw 也行）

说明
----
* 界面只是 `augment_dataset.py` 的壳：真正干活的是它的 main()，参数定义来自
  同一个 PARAM_SPEC，所以界面和命令行永远不会不一致。
* 预览是"固定种子"生成的，方便你反复比较同一个变换结果。
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent

# 注意：本目录的包名和内部模块同名（augment_dataset/ 与 augment_dataset.py），
# 若直接用 `import augment_dataset`，在"以包方式导入"的场景下会拿到包（__init__.py）
# 而不是那个脚本模块，导致 AD.main / AD.source_exists 等符号缺失。
# 因此这里用 importlib 明确加载"本目录下的同名文件"，两种运行方式都能拿到真正的模块。
#
# 但打成 exe（PyInstaller）后，源码已被编译进可执行文件，磁盘上没有
# augment_dataset.py 这个文件，必须回退到普通 import。
_MOD_PATH = HERE / "augment_dataset.py"
if _MOD_PATH.exists():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("_augment_dataset_impl", _MOD_PATH)
    if _spec is None or _spec.loader is None:  # pragma: no cover
        raise ImportError(f"无法加载 {_MOD_PATH}")
    AD = _ilu.module_from_spec(_spec)
    sys.modules.setdefault("_augment_dataset_impl", AD)
    _spec.loader.exec_module(AD)
else:
    # exe 打包后（或作为已安装包）走这里
    try:
        import augment_dataset as AD  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover
        from . import augment_dataset as AD  # type: ignore[no-redef]

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

APP_TITLE = "YOLO 数据集离线增强 - 参数面板"
CONFIG_FILE = HERE / ".augment_gui_last.json"


def _enable_dpi_awareness() -> Optional[float]:
    """在创建任何 Tk 窗口之前声明 DPI 感知，并返回应使用的 tk scaling。

    不声明的话（Windows 默认 PROCESS_DPI_UNAWARE），系统会对整个窗口做位图拉伸；
    而打开"浏览..."的系统文件对话框、或在其中新建文件夹时，DPI 上下文会被切换，
    Tk 重算 tk scaling（实测 1.332 -> 1.998，差 50%），表现为"所有字体突然变小"。

    修复办法：
        1. 进程启动时调用 SetProcessDpiAwareness(2)（per-monitor aware）；
        2. 创建根窗口后把 tk scaling 钉死为计算出的固定值，不让它随对话框漂移。
    返回 None 表示非 Windows 或调用失败（此时不做缩放干预）。
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        # 优先用 SetProcessDPIAware()：它不返回错误、也不受"必须先于任何窗口创建"的
        # 进程清单约束影响，实测在 Python 3.10 + Tk 8.6 下能稳定让 Tk 读到真实 DPI
        # （未声明时 winfo_fpixels 显示 96，声明后为 144）。
        # shcore.SetProcessDpiAwareness 在本机返回 0 但实际不生效，故放在后面作为补充尝试。
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass
        # 取主显示器 DPI，换算成 Tk 的 scaling（Tk 以 72dpi 为 1.0）
        hdc = ctypes.windll.user32.GetDC(0)
        try:
            LOGPIXELSX = 88
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, LOGPIXELSX) or 96
        finally:
            ctypes.windll.user32.ReleaseDC(0, hdc)
        return float(dpi) / 72.0
    except Exception:  # noqa: BLE001 - 任何失败都不应影响正常启动
        return None


def _pin_tk_scaling(root: "tk.Tk", scaling: Optional[float]) -> None:
    """把 tk scaling 固定下来，避免打开系统对话框后字体跳变。

    即使 _enable_dpi_awareness 失败（返回 None），也把当前 scaling 读出来钉住，
    这样打开/关闭文件对话框不会让它漂移。
    """
    target = scaling
    if target is None:
        try:
            target = float(root.tk.call("tk", "scaling"))
        except (tk.TclError, ValueError):
            return
    if not target or target <= 0:
        return
    try:
        root.tk.call("tk", "scaling", float(target))
    except tk.TclError:
        pass


def _reassert_scaling(root: "tk.Tk", scaling: Optional[float]) -> None:
    """打开系统对话框返回后再钉一次缩放，并强制命名字体重新计算。

    Tk 控件的字号在创建时就已按当时的 scaling 换算成像素，之后若 scaling 被
    系统改小（对话框/新建文件夹会切 DPI 上下文），已存在的控件不会自动重绘，
    于是表现为"部分字体变小、字号忽大忽小"。这里显式把 scaling 改回去，
    并让 ttk 主题重算一次，保证整个界面字号统一。
    """
    _pin_tk_scaling(root, scaling)
    try:
        style = ttk.Style()
        theme = style.theme_use()
        style.theme_use(theme)   # 重新应用主题，促使控件按新 scaling 重新排版
    except tk.TclError:
        pass

# 一组贴心预设（name -> {参数名: 值}）
PRESETS: Dict[str, Dict[str, object]] = {
    "默认": {},
    "小数据集(55张以内)": dict(times=6, degrees=12.0, scale=0.35, translate=0.12,
                              p_mosaic=0.5, p_cutout=0.25, preview=6),
    "工业零件(姿态固定)": dict(times=3, degrees=6.0, scale=0.20, translate=0.08,
                               shear=0.0, perspective=0.0,
                               p_hflip=0.0, p_vflip=0.0, p_rot90=0.0,
                               p_mosaic=0.3, p_cutout=0.1,
                               p_sharpen=0.2, sharpen_amount=0.4,
                               p_saltpepper=0.1, p_occlusion=0.3, occlusion_max=0.35),
    "遮挡鲁棒性": dict(times=3, degrees=10.0, scale=0.25,
                         p_occlusion=0.6, occlusion_max=0.5,
                         p_sharpen=0.2, p_saltpepper=0.15),
    "保守(只做颜色)": dict(times=2, degrees=0.0, scale=0.0, translate=0.0, shear=0.0,
                           perspective=0.0, p_mosaic=0.0, p_hflip=0.0, p_vflip=0.0,
                           p_rot90=0.0, p_cutout=0.0,
                           p_sharpen=0.0, p_saltpepper=0.0, p_occlusion=0.0),
    "分割+复制粘贴": dict(times=3, p_copypaste=0.4, p_mosaic=0.4),
}


class ParamRow:
    """一行参数控件：标签 + 滑块 + 输入框（或下拉 / 复选框）。"""

    def __init__(self, parent: ttk.Frame, spec: dict, on_change,
                 on_after_dialog=None) -> None:
        self.spec = spec
        self.name = spec["name"]
        self.kind = spec["kind"]
        self.on_change = on_change
        self.on_after_dialog = on_after_dialog
        self.var = tk.StringVar()
        self.slider: Optional[ttk.Scale] = None
        self.entry: Optional[ttk.Entry] = None
        self.combo: Optional[ttk.Combobox] = None
        self.check: Optional[tk.BooleanVar] = None
        self._building = True
        self.row = ttk.Frame(parent)

        label = ttk.Label(self.row, text=spec.get("label", spec["help"]),
                          width=15, anchor="w")
        label.pack(side=tk.LEFT, padx=(0, 4))
        self.label_widget = label
        # 悬停显示完整说明
        self._attach_tooltip(label, spec["help"] + f"  ({spec['flag']})")

        if self.kind == "bool":
            self.check = tk.BooleanVar(value=bool(spec["default"]))
            ttk.Checkbutton(self.row, variable=self.check,
                            command=self._changed).pack(side=tk.LEFT)
        elif self.kind == "choice":
            self.combo = ttk.Combobox(self.row, values=spec["choices"], width=10, state="readonly")
            self.combo.set(str(spec["default"]))
            self.combo.pack(side=tk.LEFT)
            self.combo.bind("<<ComboboxSelected>>", lambda e: self._changed())
        elif self.kind in ("int", "float", "ratio"):
            lo = spec.get("lo", 0.0)
            hi = spec.get("hi", 1.0)
            # int 类参数默认可能是 None（如 --limit），滑块需要有具体数值兜底
            init = spec["default"]
            if init is None:
                init = lo
            init = float(init)
            self.slider = ttk.Scale(self.row, from_=lo, to=hi, orient=tk.HORIZONTAL,
                                    length=150, command=self._slider_moved)
            self.slider.set(init)
            self.slider.pack(side=tk.LEFT)
            self.entry = ttk.Entry(self.row, width=8, textvariable=self.var)
            self.entry.pack(side=tk.LEFT, padx=(4, 0))
            self.entry.bind("<Return>", lambda e: self._entry_applied())
            self.entry.bind("<FocusOut>", lambda e: self._entry_applied())
            self._refresh_entry(init)
            if spec["default"] is None:
                self.var.set("")   # 空值表示"不限制"
        elif self.kind == "path":
            self.var.set(str(spec["default"] or ""))
            self.entry = ttk.Entry(self.row, width=42, textvariable=self.var)
            self.entry.pack(side=tk.LEFT)
            ttk.Button(self.row, text="浏览...", width=6,
                       command=self._browse).pack(side=tk.LEFT, padx=(4, 0))
            self.entry.bind("<Return>", lambda e: self._changed())
            self.entry.bind("<FocusOut>", lambda e: self._changed())
        else:  # str
            self.var.set(str(spec["default"] or ""))
            self.entry = ttk.Entry(self.row, width=34, textvariable=self.var)
            self.entry.pack(side=tk.LEFT)
            self.entry.bind("<Return>", lambda e: self._changed())
            self.entry.bind("<FocusOut>", lambda e: self._changed())

        ttk.Label(self.row, text=spec["flag"], foreground="#888", width=14).pack(side=tk.LEFT, padx=4)
        self._building = False

    @staticmethod
    def _attach_tooltip(widget, text: str) -> None:
        """悬停气泡提示，长说明不至于把界面撑坏。"""
        state: Dict[str, Optional[tk.Toplevel]] = {"win": None}

        def show(_e=None):
            if state["win"] is not None:
                return
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{x}+{y}")
            tk.Label(win, text=text, background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                     justify=tk.LEFT, font=("Microsoft YaHei UI", 8), padx=4, pady=2).pack()
            state["win"] = win

        def hide(_e=None):
            if state["win"] is not None:
                state["win"].destroy()
                state["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ---------- 取值 / 设值 ----------
    def value(self):
        if self.kind == "bool":
            return bool(self.check.get())
        if self.kind == "choice":
            return str(self.combo.get())
        if self.kind in ("int", "float", "ratio"):
            txt = self.var.get().strip()
            if not txt:
                # 输入框为空 = 该参数未启用（如 --limit 留空表示处理全部）
                return None if self.spec["default"] is None else self.spec["default"]
            try:
                v = float(txt) if self.kind != "int" else int(float(txt))
            except (ValueError, TypeError):
                return self.spec["default"]
            lo, hi = self.spec.get("lo"), self.spec.get("hi")
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            if self.kind == "int":
                v = int(v)
            return v
        txt = self.var.get().strip()
        return txt or None

    def set_value(self, v) -> None:
        was = self._building
        self._building = True
        try:
            if self.kind == "bool":
                self.check.set(bool(v))
            elif self.kind == "choice":
                self.combo.set(str(v))
            elif self.kind in ("int", "float", "ratio"):
                if v is None:
                    # 留空 = 该参数不启用（如 --limit 表示处理全部）
                    if self.slider is not None:
                        self.slider.set(float(self.spec.get("lo", 0.0)))
                    self.var.set("")
                else:
                    self._refresh_entry(float(v))
                    if self.slider is not None:
                        self.slider.set(float(v))
            else:
                self.var.set("" if v is None else str(v))
        finally:
            self._building = was
        if not self._building:
            self.on_change()

    def _refresh_entry(self, v: float) -> None:
        if self.kind == "int":
            self.var.set(str(int(round(v))))
        elif self.kind == "ratio":
            self.var.set(f"{v:.3f}")
        else:
            self.var.set(f"{v:g}")

    # ---------- 事件 ----------
    def _slider_moved(self, raw) -> None:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return
        # 按该参数自身的量级取整，不能用统一的两位小数
        # （perspective 默认 0.0005，统一 round(v,2) 会直接变成 0）
        if self.kind == "int":
            v = float(round(v))
        elif self.kind == "ratio":
            v = round(v, 4)
        else:
            mag = abs(self.spec.get("hi") or 1.0)
            if mag <= 0.01:
                v = round(v, 4)
            elif mag <= 1.0:
                v = round(v, 3)
            else:
                v = round(v, 1)
        self._refresh_entry(v)
        if self._building:
            return
        self.on_change()

    def _entry_applied(self) -> None:
        v = self.value()
        if v is not None and self.slider is not None:
            was = self._building
            self._building = True
            try:
                self.slider.set(float(v))
            finally:
                self._building = was
        self.on_change()

    def _browse(self) -> None:
        p = filedialog.askdirectory(title=f"选择 {self.name}")
        # 系统文件对话框会切换 DPI 上下文，导致 Tk 重算 scaling、界面字号变小。
        # 对话框一返回立刻把缩放钉回去，用户就看不到字号跳变。
        if self.on_after_dialog is not None:
            self.on_after_dialog()
        if p:
            self.var.set(p)
            self.on_change()

    def _changed(self) -> None:
        if self._building:
            return
        self.on_change()


class AugmentApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        # 先取当前缩放值：窗口尺寸和"对话框返回后重钉"都要用到它
        try:
            self._scaling: Optional[float] = float(self.root.tk.call("tk", "scaling"))
        except (tk.TclError, ValueError):
            self._scaling = None
        # 窗口尺寸随 DPI 缩放放大。参数区各控件宽度是按字符写死的，字体放大后
        # 整体宽度会明显变大（150% 缩放下参数区约需 1050 像素），窗口不够宽时
        # 右侧的「浏览...」按钮会被裁掉，表现为"看不到/点不了"。
        # 因此这里留足余量，并允许窗口随屏幕大小自适应。
        scale = (self._scaling / (96.0 / 72.0)) if self._scaling else 1.0
        scale = max(1.0, min(scale, 2.5))
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        # 左栏（参数区）按缩放给足宽度，右栏（预览+日志）保持可用宽度
        left_w = int(620 * scale)
        right_w = int(560 * scale)
        win_w = min(left_w + right_w + 60, max(900, sw - 80))
        win_h = min(int(780 * scale), max(600, sh - 120))
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(min(int(1100 * scale), win_w), min(int(620 * scale), win_h))
        self.rows: Dict[str, ParamRow] = {}
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self._finish_queue: "queue.Queue[bool]" = queue.Queue()
        self.preview_photo = None
        self.worker: Optional[threading.Thread] = None
        self._preview_job: Optional[str] = None
        self.sample_index = 0
        self.sample_total = 0
        # 输出数据集预览的状态
        self._preview_mode = "source"    # "source" 实时增强 / "output" 读取已生成文件
        self.out_sample_index = 0
        self.out_sample_total = 0
        self.out_task = "auto"
        self.out_names: List[str] = []
        # 输出目录被改动后，输出预览需要重新加载样本列表
        self._out_dirty = True

        self._build_ui()
        self._load_config()
        self.root.after(100, self._drain_log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _after_system_dialog(self) -> None:
        """系统文件对话框关闭后调用：把 DPI 缩放钉回来并刷新字体。

        这是"新建文件夹后界面字体变小"的直接修复点 —— 对话框会切 DPI 上下文，
        Tk 因此重算 scaling；已创建的控件需重新应用主题才会按原字号重排。
        """
        try:
            _reassert_scaling(self.root, self._scaling)
            self.root.update_idletasks()
        except (tk.TclError, AttributeError):
            pass

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        # 先 pack 底部（占位），剩下的空间再分给左右两栏，
        # 否则底部会把右侧预览挤没
        bottom = ttk.Frame(outer)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        body = ttk.Frame(outer)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 用 grid 明确分配左右两栏：左栏给固定宽度（参数区需要完整显示控件），
        # 右栏吃掉剩余空间。用 pack 时左栏会被右栏挤压，导致参数行右侧控件被裁。
        body.columnconfigure(0, weight=0)   # 左：参数区，不参与拉伸
        body.columnconfigure(1, weight=1)   # 右：预览+日志，吃掉剩余宽度
        body.rowconfigure(0, weight=1)

        scale = (self._scaling / (96.0 / 72.0)) if self._scaling else 1.0
        scale = max(1.0, min(scale, 2.5))
        self._left_w = int(620 * scale)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew")
        # 不锁死宽度：左栏由内部 canvas 的真实需求决定宽度（见 _sync_scrollregion），
        # 锁死会导致 canvas 被工具栏/滚动条挤压，右侧控件被裁。

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # ---- 预设 ----
        preset_bar = ttk.Frame(left)
        preset_bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(preset_bar, text="预设:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value="默认")
        cb = ttk.Combobox(preset_bar, textvariable=self.preset_var, state="readonly",
                          values=list(PRESETS), width=18)
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        ttk.Button(preset_bar, text="恢复默认", command=self._reset_defaults).pack(side=tk.LEFT, padx=4)
        ttk.Button(preset_bar, text="自动配对标签", width=13,
                   command=self._auto_pair_labels).pack(side=tk.LEFT, padx=(10, 0))

        # ---- 参数区（可滚动）----
        # 宽度不能用"固定值 × 缩放"估算：参数行里各控件的宽度是按字符数写死的，
        # 字体放大后实际宽度会超出估算值，右侧「浏览...」按钮会被 canvas 裁掉
        # （窗口里看不到、点不到）。这里给一个初始宽度，随后由 _sync_scrollregion
        # 按 holder 的真实 reqwidth 自动撑开。
        canvas = tk.Canvas(left, width=200, highlightthickness=0)
        vbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        hbar = ttk.Scrollbar(left, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        holder = ttk.Frame(canvas)
        _win_id = canvas.create_window((0, 0), window=holder, anchor="nw")

        def _sync_scrollregion(_e=None):
            """按内容真实需求设定 canvas 宽度，保证最右侧控件不被裁掉。"""
            try:
                need = holder.winfo_reqwidth()
                if need > 0:
                    # 内容窗口固定为真实宽度（不打折、不拉伸）
                    canvas.itemconfigure(_win_id, width=need)
                    # canvas 本身也按需求撑开；窗口更宽时由 pack(fill=X) 自然填满
                    if int(canvas.cget("width")) != need:
                        canvas.configure(width=need)
                canvas.configure(scrollregion=canvas.bbox("all"))
            except tk.TclError:
                pass

        holder.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        current_group = None
        spec_box = holder
        for spec in AD.PARAM_SPEC:
            g = spec.get("group", "")
            if g != current_group:
                current_group = g
                lf = ttk.LabelFrame(holder, text=(g if g else "基础设置"), padding=6)
                lf.pack(fill=tk.X, pady=4, padx=2)
                spec_box = lf
            row = ParamRow(spec_box, spec, self._on_param_changed,
                           on_after_dialog=self._after_system_dialog)
            row.row.pack(fill=tk.X, pady=1)
            self.rows[spec["name"]] = row

        # ---- 右侧：预览 ----
        prev_head = ttk.Frame(right)
        prev_head.pack(fill=tk.X)
        ttk.Label(prev_head, text="实时预览（绿框=检测框 / 黄线=分割多边形）",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        ttk.Button(prev_head, text="下一张 ▶", width=8,
                   command=self._next_sample).pack(side=tk.RIGHT)
        ttk.Button(prev_head, text="◀ 上一张", width=8,
                   command=self._prev_sample).pack(side=tk.RIGHT, padx=4)
        ttk.Button(prev_head, text="预览", width=7, command=self.do_preview).pack(side=tk.RIGHT)

        self.preview_label = ttk.Label(right, text="（尚未预览：先选数据集，再点「② 预览增强」）",
                                       anchor=tk.CENTER, relief=tk.SUNKEN, background="#f4f4f4")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=6)

        # 日志放在右下角，避免底部越堆越高把预览挤掉
        log_box = ttk.LabelFrame(right, text="运行日志", padding=4)
        log_box.pack(fill=tk.X, side=tk.BOTTOM)
        self.log_text = tk.Text(log_box, height=8, wrap=tk.WORD, font=("Consolas", 9))
        sb = ttk.Scrollbar(log_box, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- 底部：状态 / 按钮 / 命令 ----
        stat_row = ttk.Frame(bottom)
        stat_row.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(stat_row, text="状态:", font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(stat_row, textvariable=self.status_var, foreground="#0060a0").pack(side=tk.LEFT, padx=6)

        btn_row = ttk.Frame(bottom)
        btn_row.pack(fill=tk.X, pady=4)
        self.btn_stat = ttk.Button(btn_row, text="① 统计数据集", width=14, command=self.do_stat)
        self.btn_stat.pack(side=tk.LEFT, padx=2)
        self.btn_preview = ttk.Button(btn_row, text="② 预览原图", width=14,
                                      command=self.do_preview)
        self.btn_preview.pack(side=tk.LEFT, padx=2)
        self.btn_run = ttk.Button(btn_row, text="③ 开始增强", width=14, command=self.do_run)
        self.btn_run.pack(side=tk.LEFT, padx=2)
        self.btn_open = ttk.Button(btn_row, text="打开输出目录", width=13,
                                   command=self.do_open_out)
        self.btn_open.pack(side=tk.LEFT, padx=2)
        self.btn_preview_out = ttk.Button(btn_row, text="预览输出数据集", width=15,
                                          command=self.do_preview_output)
        self.btn_preview_out.pack(side=tk.LEFT, padx=2)

        self.progress = ttk.Progressbar(bottom, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=2)

        cmd_box = ttk.LabelFrame(bottom, text="等价的命令行（可直接复制到终端）", padding=4)
        cmd_box.pack(fill=tk.X, pady=(2, 0))
        self.cmd_text = tk.Text(cmd_box, height=2, wrap=tk.WORD, font=("Consolas", 9))
        self.cmd_text.pack(fill=tk.X)

        self._on_param_changed(initial=True)

    # ------------------------------------------------------------- 参数读取
    def values(self) -> Dict[str, object]:
        return {n: r.value() for n, r in self.rows.items()}

    @staticmethod
    def _safe_value(name: str, v):
        """path/int 类参数为空时，用默认值兜底。"""
        spec = next(s for s in AD.PARAM_SPEC if s["name"] == name)
        if v in (None, "") and spec["kind"] != "str":
            return spec["default"]
        return v

    def _effective(self) -> Dict[str, object]:
        return {k: self._safe_value(k, v) for k, v in self.values().items()}

    # ------------------------------------------------------------- 变更刷新
    def _on_param_changed(self, initial: bool = False) -> None:
        if not hasattr(self, "cmd_text"):
            return  # 界面还没搭完，等最后统一刷新
        self._update_cmd()
        self._save_config()
        self._check_out_changed()
        if initial:
            return
        # 自动刷新预览（防抖 500ms），省得反复点
        if self._preview_job is not None:
            try:
                self.root.after_cancel(self._preview_job)
            except tk.TclError:
                pass
        if self._preview_mode == "output":
            # 正在看"已生成的数据集"时，改增强参数不该把画面切走（那些参数对
            # 已有文件无效）；只有改了输出路径才重新加载。
            if self._out_dirty:
                self._out_dirty = False
                self.out_sample_total = 0
                self.out_sample_index = 0
                self._preview_job = self.root.after(500, self.do_preview_output)
            return
        self._preview_job = self.root.after(500, self.do_preview)

    def _update_cmd(self) -> None:
        self.cmd_text.delete("1.0", tk.END)
        self.cmd_text.insert("1.0", build_command(self._effective()))

    # ------------------------------------------------------------- 预设/配置
    def _auto_pair_labels(self) -> None:
        """根据图片路径推测标签路径并填入。

        YOLO 数据集通常是 `<root>/images/...` 与 `<root>/labels/...` 平行存放，
        本按钮就是把 images 换成 labels。若用户填的是自定义位置（两者不平行），
        请直接用「浏览...」指定，界面不会强行覆盖已填的有效路径。
        """
        img = (self.rows["image_path"].value() or "").strip()
        if not img:
            messagebox.showinfo("提示", "请先填写或选择「图片路径」。")
            return
        cur_lbl = (self.rows["label_path"].value() or "").strip()
        if cur_lbl and AD.source_exists(img, cur_lbl):
            self._log(f"[自动配对] 标签路径已有效，保持不变：{cur_lbl}")
            return
        guess = AD.guess_label_path(img)
        if guess is None:
            messagebox.showinfo(
                "无法自动配对",
                f"没能从图片路径推出标签目录：\n{img}\n\n"
                "请用「浏览...」手动选择标签目录。\n"
                "（自动配对只在路径里含 images 目录时生效）")
            return
        self.rows["label_path"].set_value(str(guess))
        self._log(f"[自动配对] 标签路径 -> {guess}")
        if not Path(guess).is_dir():
            self._log("  ! 该目录还不存在，请检查或手动指定")

    def _apply_preset(self) -> None:
        name = self.preset_var.get()
        merged = {s["name"]: s["default"] for s in AD.PARAM_SPEC}
        merged.update(PRESETS.get(name, {}))
        # 保留用户已选的数据集路径与任务类型（预设只改增强强度，不该把任务改回 auto）
        for key in ("image_path", "label_path", "out", "task", "names"):
            cur = self.rows[key].value()
            if cur not in (None, ""):
                merged[key] = cur
        for n, val in merged.items():
            if n in self.rows:
                self.rows[n].set_value(val)
        self._log(f"已应用预设：{name}")
        self._on_param_changed()

    def _reset_defaults(self) -> None:
        img, lbl = self.rows["image_path"].value(), self.rows["label_path"].value()
        out = self.rows["out"].value()
        for spec in AD.PARAM_SPEC:
            self.rows[spec["name"]].set_value(spec["default"])
        if img:
            self.rows["image_path"].set_value(img)
        if lbl:
            self.rows["label_path"].set_value(lbl)
        if out:
            self.rows["out"].set_value(out)
        self._log("已恢复默认参数")
        self._on_param_changed()

    def _save_config(self) -> None:
        try:
            CONFIG_FILE.write_text(json.dumps(self.values(), ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except OSError:
            pass

    def _load_config(self) -> None:
        if not CONFIG_FILE.exists():
            return
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return
        changed = False
        for n, val in data.items():
            if n in self.rows and val is not None:
                try:
                    self.rows[n].set_value(val)
                    changed = True
                except (ValueError, TypeError):
                    pass
        if changed:
            self._update_cmd()

    # ------------------------------------------------------------- 日志
    def _log(self, msg: str) -> None:
        self.log_queue.put(msg)

    def _drain_log(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg.rstrip() + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        # 主线程专属：处理工作线程发来的"完成"事件
        try:
            ok = self._finish_queue.get_nowait()
        except queue.Empty:
            ok = None
        if ok is not None:
            self._finish(ok)
        self.root.after(100, self._drain_log)

    # ------------------------------------------------------------- ① 统计
    def do_stat(self) -> None:
        v = self._effective()
        img, lbl = v.get("image_path"), v.get("label_path")
        if not img or not lbl:
            messagebox.showwarning("提示", "请先填写图片路径和标签路径")
            return
        if not AD.source_exists(img, lbl):
            messagebox.showwarning(
                "提示", "图片路径或标签路径无效：\n"
                        f"图片：{img}\n标签：{lbl}\n\n请确认两个目录都存在。")
            return
        try:
            items, task, names, has_splits = AD.resolve_source(img, lbl, str(v["task"]))
        except ValueError as e:
            messagebox.showerror("无法读取数据集", str(e))
            self._log(f"[错误] {e}")
            return
        self.sample_total = len(items)
        self.sample_index = 0
        inst: Dict[int, int] = {}
        empty = 0
        for it in items:
            lines = ([l for l in it.lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                      if l.strip()] if it.lbl_path.exists() else [])
            if not lines:
                empty += 1
            for l in lines:
                c = int(l.split()[0])
                inst[c] = inst.get(c, 0) + 1
        times = int(v["times"])
        total = len(items) * times + (0 if v["no_original"] else len(items))
        self.status_var.set(
            f"任务={task}  类别={names}  源图片={len(items)}"
            f"{' (已有train/val划分)' if has_splits else ''}  实例={inst or '无'}  "
            f"空标签={empty}  增强后 train 约 {total} 张")
        self._log(f"[统计] 图片 {img}")
        self._log(f"       标签 {lbl}")
        self._log(f"  任务={task} 类别={names} 图片={len(items)} 实例={inst} 空标签={empty}")
        if not inst:
            self._log("  ⚠ 这个数据集一个标注都没有！增强前请先标注（否则只会生成一堆空标签图）")

    # ------------------------------------------------------------- ② 预览
    # 预览有两种数据来源，用 _preview_mode 区分：
    #   "source" : 对源图片按当前参数实时增强（看参数效果）
    #   "output" : 直接读输出目录里已生成的文件（核对增强产物）
    # 上/下一张按钮会自动沿用当前模式。
    def _step_sample(self, delta: int) -> None:
        """在样本间前后切换。总数未知（还没统计）时先统计一次。"""
        if self._preview_mode == "output":
            if not self.out_sample_total:
                self._load_output_samples(verbose=False)
                if not self.out_sample_total:
                    return
            self.out_sample_index = (self.out_sample_index + delta) % self.out_sample_total
            self.do_preview_output(step_only=True)
            return
        if not self.sample_total:
            # 还没点过「统计数据集」，先取一次样本总数，否则无法循环
            self.do_stat()
            if not self.sample_total:
                return
        self.sample_index = (self.sample_index + delta) % self.sample_total
        self.do_preview()

    def _next_sample(self) -> None:
        self._step_sample(1)

    def _prev_sample(self) -> None:
        self._step_sample(-1)

    def _show_image(self, bgr: np.ndarray, status: str) -> None:
        """把 BGR 图缩放到预览区并显示。"""
        import cv2
        from PIL import Image, ImageTk
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        disp_w = max(320, self.preview_label.winfo_width() - 12)
        disp_h = max(240, self.preview_label.winfo_height() - 12)
        pil.thumbnail((disp_w, disp_h), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=self.preview_photo, text="")
        self.status_var.set(status)

    def do_preview(self) -> None:
        """预览原图：对源图片按当前参数实时做一次增强。"""
        self._preview_job = None
        self._preview_mode = "source"
        v = self._effective()
        img_p, lbl_p = v.get("image_path"), v.get("label_path")
        if not img_p or not lbl_p or not AD.source_exists(img_p, lbl_p):
            self.preview_label.configure(text="请先填写有效的图片路径与标签路径")
            return
        try:
            out_img = AD.preview_image(img_p, lbl_p, v, self.sample_index)
        except ValueError as e:
            self.preview_label.configure(text=f"预览失败：{e}")
            self._log(f"[预览失败] {e}")
            return
        except Exception as e:  # noqa: BLE001 - 预览出错不该崩界面
            self.preview_label.configure(text=f"预览失败：{e}")
            self._log(f"[预览异常] {e}\n{traceback.format_exc()}")
            return

        self._show_image(out_img,
                         f"预览原图（实时增强）第 {self.sample_index + 1}/"
                         f"{self.sample_total or '?'} 张")

    # ------------------------------------------------- 预览输出数据集
    def _current_out_dir(self) -> Optional[Path]:
        """当前输出目录（未填则按默认规则推导）。"""
        v = self._effective()
        img = v.get("image_path")
        if v.get("out"):
            return Path(v["out"])
        return AD.default_out_for(img) if img else None

    def _check_out_changed(self) -> None:
        """输出目录变了就把输出预览标记为需要重新加载。"""
        cur = self._current_out_dir()
        key = str(cur) if cur else ""
        if key != getattr(self, "_out_key", None):
            self._out_key = key
            self._out_dirty = True
            self.out_sample_total = 0
            self.out_sample_index = 0

    def _load_output_samples(self, verbose: bool = True) -> bool:
        """读取输出数据集，记录样本总数。成功返回 True。"""
        out = self._current_out_dir()
        if out is None:
            self.preview_label.configure(text="请先填写图片路径（用于推导输出目录）")
            return False
        try:
            items, task, names = AD.list_output_items(out)
        except ValueError as e:
            self.preview_label.configure(text=f"无法预览输出数据集：{e}")
            if verbose:
                self._log(f"[输出预览失败] {e}")
            self.out_sample_total = 0
            return False
        self.out_sample_total = len(items)
        self.out_task = task
        self.out_names = names
        if self.out_sample_index >= self.out_sample_total:
            self.out_sample_index = 0
        if verbose:
            self._log(f"[输出数据集] {out}")
            self._log(f"  {self.out_sample_total} 张图，任务={task}，类别={names}")
        return True

    def do_preview_output(self, step_only: bool = False) -> None:
        """预览输出数据集：显示已生成图片及其标注（不重新做增强）。"""
        self._preview_job = None
        self._preview_mode = "output"
        out = self._current_out_dir()
        if out is None:
            self.preview_label.configure(text="请先填写图片路径（用于推导输出目录）")
            return
        if not out.is_dir():
            msg = (f"输出目录还不存在：\n{out}\n\n"
                   f"请先点「③ 开始增强」生成数据集。")
            self.preview_label.configure(text=msg)
            if not step_only:
                self._log(f"[输出预览] {msg}")
            return
        if not self.out_sample_total or not step_only:
            if not self._load_output_samples():
                return
        try:
            img = AD.preview_output_image(out, self.out_sample_index)
        except ValueError as e:
            self.preview_label.configure(text=f"预览失败：{e}")
            self._log(f"[输出预览失败] {e}")
            return
        except Exception as e:  # noqa: BLE001
            self.preview_label.configure(text=f"预览失败：{e}")
            self._log(f"[输出预览异常] {e}\n{traceback.format_exc()}")
            return
        self._show_image(img,
                         f"预览输出数据集 第 {self.out_sample_index + 1}/"
                         f"{self.out_sample_total} 张")

    # ------------------------------------------------------------- ③ 运行
    def do_run(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("提示", "正在运行中，请稍候")
            return
        v = self._effective()
        img, lbl = v.get("image_path"), v.get("label_path")
        if not img or not lbl:
            messagebox.showwarning("提示", "请先填写图片路径和标签路径")
            return
        if not AD.source_exists(img, lbl):
            messagebox.showwarning(
                "提示", "图片路径或标签路径无效：\n"
                        f"图片：{img}\n标签：{lbl}\n\n请确认两个目录都存在。")
            return
        out = Path(v["out"]) if v.get("out") else AD.default_out_for(img)
        if out.exists() and not v["overwrite"]:
            if not messagebox.askyesno(
                    "输出目录已存在",
                    f"{out}\n\n已存在。是否清空并覆盖？（等同勾选 --overwrite）"):
                return
            # 关键：勾上覆盖后必须同步回 v，否则后面 build_argv(v) 仍用旧字典，
            # 命令里不会带 --overwrite，脚本依然报"输出目录已存在"而失败。
            self.rows["overwrite"].set_value(True)
            v["overwrite"] = True

        # 二次确认：没有标注就白干
        try:
            items, task, names, _ = AD.resolve_source(img, lbl, str(v["task"]))
        except ValueError as e:
            messagebox.showerror("无法读取数据集", str(e))
            return
        has_any = any(it.lbl_path.exists() and it.lbl_path.read_text(
            encoding="utf-8", errors="ignore").strip() for it in items)
        if not has_any:
            if not messagebox.askyesno(
                    "确认", "这个数据集没有任何标注（labels 为空）。\n"
                            "继续也只会生成一堆空标签图，训练没有意义。\n\n仍然继续？"):
                return

        argv = build_argv(v)
        self._log("=" * 60)
        if "--overwrite" in argv:
            self._log(f"[注意] 将清空并重建输出目录: {out}")
        self._log("[开始] python augment_dataset.py " + " ".join(argv))
        self.progress.start(12)
        self._set_busy(True)
        self.worker = threading.Thread(target=self._run_worker, args=(argv,), daemon=True)
        self.worker.start()

    def _run_worker(self, argv: List[str]) -> None:
        """在后台线程里调用 augment_dataset.main()，重定向 print 到日志。"""
        import contextlib
        import io

        buf = io.StringIO()

        class _Tee(io.TextIOBase):
            def __init__(self, *streams):
                self.streams = streams

            def write(self, s):
                for st in self.streams:
                    st.write(s)
                self.flush()
                return len(s)

            def flush(self):
                for st in self.streams:
                    try:
                        st.flush()
                    except (ValueError, OSError):
                        pass

        ok = True
        try:
            # 不要 chdir 到脚本目录：那会让用户在界面里填的相对路径失效。
            # augment_dataset 内部已能按"当前目录 -> 脚本目录 -> 上级目录"回退解析。
            with contextlib.redirect_stdout(_Tee(buf, sys.__stdout__)), \
                    contextlib.redirect_stderr(_Tee(buf, sys.__stderr__)):
                AD.main(argv)
        except SystemExit as e:
            if e.code not in (0, None):
                ok = False
                self._log(f"[退出码 {e.code}]")
        except Exception as e:  # noqa: BLE001
            ok = False
            self._log(f"[异常] {e}\n{traceback.format_exc()}")
        finally:
            text = buf.getvalue()
            if text:
                self._log(text)
            # Tkinter 不是线程安全的：不能在工作线程里直接碰控件或 root.after，
            # 只能把"结束"事件塞进队列，由主线程的 _drain_log 轮询处理。
            self._finish_queue.put(ok)

    def _finish(self, ok: bool) -> None:
        self.progress.stop()
        self._set_busy(False)
        self.status_var.set("完成" if ok else "失败/中断")
        self._log("[完成]" if ok else "[失败]")
        if ok:
            messagebox.showinfo("完成", "增强完成，可用「打开输出目录」查看结果。")
        else:
            messagebox.showerror("失败", "详见右侧日志。")

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for b in (self.btn_stat, self.btn_preview, self.btn_run):
            b.configure(state=state)

    # ------------------------------------------------------------- 打开输出
    def do_open_out(self) -> None:
        v = self._effective()
        img = v.get("image_path")
        out = Path(v["out"]) if v.get("out") else (AD.default_out_for(img) if img else None)
        if not out or not out.exists():
            messagebox.showinfo("提示", "输出目录还不存在，先运行一次增强。")
            return
        try:
            subprocess.Popen(["explorer", str(out)])
        except OSError as e:
            messagebox.showerror("打不开", str(e))

    # ------------------------------------------------------------- 关闭
    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("正在运行", "增强仍在进行，确定要退出吗？"):
                return
        self._save_config()
        self.root.destroy()


def sep_skip(spec: dict, val) -> bool:
    """非必填参数若等于默认值（或为空），就不写进命令。"""
    if spec.get("required"):
        return False
    if val in (None, ""):
        return True
    dflt = spec["default"]
    # 浮点用近似比较，避免滑块 round 后与默认值的微小差异被判为"改动"
    if spec["kind"] in ("float", "ratio"):
        try:
            return dflt is not None and abs(float(val) - float(dflt)) <= 1e-9
        except (TypeError, ValueError):
            return False
    if spec["kind"] == "int":
        try:
            return dflt is not None and int(val) == int(dflt)
        except (TypeError, ValueError):
            return False
    return val == dflt


def build_argv(v: Dict[str, object]) -> List[str]:
    """把参数值转成 augment_dataset.py 的命令行参数列表。"""
    argv: List[str] = []
    for spec in AD.PARAM_SPEC:
        name = spec["name"]
        val = v.get(name, spec["default"])
        if spec["kind"] == "bool":
            if val:
                argv.append(spec["flag"])
            continue
        if name in ("image_path", "label_path"):
            # 界面刚启动时路径可能还没填，这里用占位符展示，不要抛异常（否则界面起不来）
            argv += [spec["flag"], str(val) if val not in (None, "") else "<未选择>"]
            continue
        if sep_skip(spec, val):
            continue
        argv += [spec["flag"], str(val)]
    return argv


def build_command(v: Dict[str, object]) -> str:
    argv = build_argv(v)
    return "python augment_dataset.py " + " ".join(
        f'"{a}"' if " " in a else a for a in argv)


def main() -> int:
    # 必须在创建 Tk 之前声明 DPI 感知，否则打开文件对话框后字体大小会跳变
    scaling = _enable_dpi_awareness()
    root = tk.Tk()
    _pin_tk_scaling(root, scaling)
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass
    AugmentApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
