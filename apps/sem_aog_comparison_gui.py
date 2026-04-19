"""
SEM AOG comparison figure GUI tool (multi-model version).

Key capabilities:
- Select multiple model result overlay folders
- Customize model display names
- Select multiple samples (columns)
- Choose whether to output Pred rows and/or Overlay rows for each model
- Generate publication-quality aligned figure and save at 300 dpi
- Remember last selected model folders
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageTk
import matplotlib.pyplot as plt


# =========================
# Config (easy to edit)
# =========================
DEFAULT_GT_FOLDER = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/GT/"
DEFAULT_IMAGE_FOLDER = "/Users/phoenix/Desktop/AOGs Detection/Train and Test/test/images/"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NORMALIZE_SUFFIXES = ["_overlay", "_pred", "_mask"]
APP_CONFIG_FILE = Path.home() / ".sem_aog_comparison_gui.json"

# Force figure text to use Times New Roman.
plt.rcParams["font.family"] = "Times New Roman"


@dataclass
class ModelSelection:
    display_name: str
    overlay_folder: Path
    mask_folder: Optional[Path]
    overlay_index: Dict[str, Path]
    pred_index: Dict[str, Path]


def load_app_settings() -> Dict[str, Any]:
    """Load app settings from disk, returning empty dict on failure."""
    try:
        if not APP_CONFIG_FILE.exists():
            return {}
        with APP_CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_app_settings(settings: Dict[str, Any]) -> None:
    """Save app settings to disk; failures are ignored to keep GUI responsive."""
    try:
        APP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with APP_CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=True, indent=2)
    except Exception:
        pass


def normalize_filename_stem(stem: str) -> str:
    """Normalize file stem for robust matching across Image / GT / Pred / Overlay."""
    s = stem.strip()
    changed = True
    while changed:
        changed = False
        lower_s = s.lower()
        for suffix in NORMALIZE_SUFFIXES:
            if lower_s.endswith(suffix):
                s = s[: -len(suffix)]
                changed = True
                break
    return s


def list_image_files(folder: Path) -> List[Path]:
    """List image files in a folder by allowed extensions (case-insensitive)."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ])


def build_key_index(folder: Path) -> Dict[str, Path]:
    """Build normalized key -> best matching file index for one folder."""
    idx: Dict[str, Path] = {}
    for p in list_image_files(folder):
        key = normalize_filename_stem(p.stem)
        current = idx.get(key)
        # Prefer shorter stem when there are collisions (usually cleaner base file).
        if current is None or len(p.stem) < len(current.stem):
            idx[key] = p
    return idx


def load_model_from_overlay_folder(overlay_folder: Path, display_name: str) -> ModelSelection:
    """
    Build one model selection from overlay folder.

    Expected result layout:
    result_dir/
      overlays/
      masks/
    """
    overlay_index = build_key_index(overlay_folder)
    result_dir = overlay_folder.parent
    mask_folder = result_dir / "masks"
    pred_index: Dict[str, Path] = {}
    if mask_folder.exists() and mask_folder.is_dir():
        pred_index = build_key_index(mask_folder)
    else:
        mask_folder = None

    return ModelSelection(
        display_name=display_name,
        overlay_folder=overlay_folder,
        mask_folder=mask_folder,
        overlay_index=overlay_index,
        pred_index=pred_index,
    )


def sanitize_label_for_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "model"


def _read_for_plot(path: Path, keep_color: bool = True):
    """Load image with Pillow and return RGB or grayscale image for matplotlib."""
    with Image.open(path) as im:
        if keep_color:
            return im.convert("RGB")
        return im.convert("L")


def generate_comparison_figure(
    columns_data: List[Dict[str, Path]],
    row_specs: List[Tuple[str, str]],
    wspace: float = 0.003,
    hspace: float = 0.003,
    cell_height: float = 2.1,
):
    """
    Build aligned figure.

    Parameters:
    - columns_data: one dict per selected sample column
    - row_specs: ordered list of (row_label, row_key)
    """
    if not columns_data:
        raise ValueError("No matched items to plot.")
    if not row_specs:
        raise ValueError("No rows selected for plotting.")

    n_cols = len(columns_data)
    n_rows = len(row_specs)

    # Adapt figure size to selected image ratio and selected row/column counts.
    aspect_samples: List[float] = []
    for sample in columns_data[: min(5, len(columns_data))]:
        image_path = sample.get("image")
        if image_path is None:
            continue
        try:
            with Image.open(image_path) as im:
                w, h = im.size
                if h > 0:
                    aspect_samples.append(w / h)
        except Exception:
            continue
    image_aspect = sum(aspect_samples) / len(aspect_samples) if aspect_samples else 1.0

    cell_h = max(1.2, float(cell_height))
    cell_w = max(1.2, cell_h * image_aspect)

    # Reserve room for left-side labels while keeping the rest adaptive.
    left_label_inches = 1.35
    fig_w = max(left_label_inches + (cell_w * n_cols), 6.0)
    fig_h = max(cell_h * n_rows, 4.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=100)

    if n_cols == 1 and n_rows == 1:
        axes = [[axes]]
    elif n_cols == 1:
        axes = [[axes[r]] for r in range(n_rows)]
    elif n_rows == 1:
        axes = [list(axes)]

    for c, sample in enumerate(columns_data):
        for r, (row_label, row_key) in enumerate(row_specs):
            ax = axes[r][c]
            path = sample[row_key]
            keep_color = row_key.startswith("image") or row_key.startswith("overlay")
            img = _read_for_plot(path, keep_color=keep_color)
            if getattr(img, "mode", "") == "L":
                ax.imshow(img, cmap="gray", aspect="auto")
            else:
                ax.imshow(img, aspect="auto")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if c == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=10,
                    rotation=0,
                    labelpad=18,
                    va="center",
                    ha="right",
                    fontname="Times New Roman",
                )
                ax.yaxis.set_label_coords(-0.03, 0.5)

    # Make horizontal and vertical gaps equally tight.
    ws = max(0.0, float(wspace))
    hs = max(0.0, float(hspace))
    plt.subplots_adjust(left=0.12, right=0.998, top=0.995, bottom=0.01, wspace=ws, hspace=hs)
    return fig


def save_figure(fig, out_path: Path):
    """Save figure with publication-quality resolution."""
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.01)


class SemAogComparisonApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SEM AOG Comparison Figure Tool (Multi-model)")
        self.root.geometry("1320x800")

        self.models: List[ModelSelection] = []
        self.sample_keys: List[str] = []
        self.sample_display_lines: List[str] = []

        # Selection-order tracking for columns.
        self.selection_order: List[int] = []
        self.current_selected_set: Set[int] = set()
        self.last_preview_idx: Optional[int] = None

        self.preview_photo = None

        self.gt_var = tk.StringVar(value=DEFAULT_GT_FOLDER)
        self.img_var = tk.StringVar(value=DEFAULT_IMAGE_FOLDER)
        self.save_var = tk.StringVar(value=str(Path.cwd() / "comparison_figure.png"))
        self.preview_model_var = tk.StringVar(value="")

        self.include_pred_var = tk.BooleanVar(value=True)
        self.include_overlay_var = tk.BooleanVar(value=True)
        self.wspace_var = tk.StringVar(value="0.003")
        self.hspace_var = tk.StringVar(value="0.003")
        self.cell_height_var = tk.StringVar(value="2.1")
        self.row_label_overrides: Dict[str, str] = {}

        self._build_ui()
        self._restore_last_models()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(main, text="Folders & Figure Options", padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="GT Folder:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(controls, textvariable=self.gt_var, width=95).grid(row=0, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        ttk.Button(controls, text="Browse", command=self.on_browse_gt_folder).grid(row=0, column=5, padx=4, pady=4)

        ttk.Label(controls, text="Image Folder:").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(controls, textvariable=self.img_var, width=95).grid(row=1, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        ttk.Button(controls, text="Browse", command=self.on_browse_image_folder).grid(row=1, column=5, padx=4, pady=4)

        ttk.Label(controls, text="Save PNG:").grid(row=2, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(controls, textvariable=self.save_var, width=95).grid(row=2, column=1, columnspan=4, padx=4, pady=4, sticky="we")
        ttk.Button(controls, text="Choose", command=self.on_choose_save_path).grid(row=2, column=5, padx=4, pady=4)

        options = ttk.Frame(controls)
        options.grid(row=3, column=0, columnspan=6, sticky="w", padx=2, pady=(6, 2))
        ttk.Label(options, text="Row output:").pack(side="left")
        ttk.Checkbutton(options, text="Pred rows", variable=self.include_pred_var).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(options, text="Overlay rows", variable=self.include_overlay_var).pack(side="left", padx=(8, 0))
        ttk.Label(options, text="wspace:").pack(side="left", padx=(12, 2))
        ttk.Entry(options, textvariable=self.wspace_var, width=6).pack(side="left")
        ttk.Label(options, text="hspace:").pack(side="left", padx=(8, 2))
        ttk.Entry(options, textvariable=self.hspace_var, width=6).pack(side="left")
        ttk.Label(options, text="cellH:").pack(side="left", padx=(8, 2))
        ttk.Entry(options, textvariable=self.cell_height_var, width=6).pack(side="left")
        ttk.Button(options, text="Edit Row Labels", command=self.on_edit_row_labels).pack(side="left", padx=(12, 0))
        ttk.Button(options, text="Generate Figure", command=self.on_generate_figure).pack(side="left", padx=(16, 0))

        body = ttk.Frame(main)
        body.pack(fill="both", expand=True, pady=(8, 0))
        self.body_frame = body

        # Left: model configuration (intentionally narrower)
        model_panel = ttk.LabelFrame(body, text="Models (Add multiple overlay folders)", padding=8)
        model_panel.pack(side="right", fill="both", expand=False, padx=(8, 0))
        self.model_panel = model_panel

        self.model_tree = ttk.Treeview(model_panel, columns=("name", "overlay", "mask"), show="headings", height=11)
        self.model_tree.heading("name", text="Model Label")
        self.model_tree.heading("overlay", text="Overlay Folder")
        self.model_tree.heading("mask", text="Mask Folder")
        self.model_tree.column("name", width=110, anchor="w")
        self.model_tree.column("overlay", width=180, anchor="w")
        self.model_tree.column("mask", width=165, anchor="w")
        self.model_tree.pack(fill="both", expand=True)

        model_tree_xsb = ttk.Scrollbar(model_panel, orient="horizontal", command=self.model_tree.xview)
        model_tree_xsb.pack(fill="x", pady=(4, 0))
        self.model_tree.configure(xscrollcommand=model_tree_xsb.set)

        model_btns = ttk.Frame(model_panel)
        model_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(model_btns, text="Add Model", command=self.on_add_model).pack(side="left")
        ttk.Button(model_btns, text="Remove Selected", command=self.on_remove_model).pack(side="left", padx=(6, 0))
        ttk.Button(model_btns, text="Rename Label", command=self.on_rename_model_label).pack(side="left", padx=(6, 0))

        # Middle: overlay selection list
        sample_panel = ttk.LabelFrame(body, text="Overlay Selection (select in desired left->right order)", padding=8)
        sample_panel.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.sample_panel = sample_panel

        hint = ttk.Label(sample_panel, text="Tip: click overlay rows to toggle selection. Figure columns follow your click order.")
        hint.pack(anchor="w", pady=(0, 6))

        self.listbox = tk.Listbox(sample_panel, selectmode=tk.MULTIPLE, exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_selection_change)
        self.listbox.bind("<ButtonRelease-1>", self.on_list_click_release)

        sample_sb = ttk.Scrollbar(sample_panel, orient="vertical", command=self.listbox.yview)
        sample_sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sample_sb.set)

        sample_xsb = ttk.Scrollbar(sample_panel, orient="horizontal", command=self.listbox.xview)
        sample_xsb.pack(fill="x", pady=(4, 0))
        self.listbox.config(xscrollcommand=sample_xsb.set)

        sample_btns = ttk.Frame(sample_panel)
        sample_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(sample_btns, text="Select All", command=self.on_select_all).pack(side="left")
        ttk.Button(sample_btns, text="Clear Selection", command=self.on_clear_selection).pack(side="left", padx=(6, 0))
        self.selected_count_label = ttk.Label(sample_btns, text="Selected samples: 0")
        self.selected_count_label.pack(side="left", padx=(12, 0))

        # Right: preview
        preview_panel = ttk.LabelFrame(body, text="Overlay Preview", padding=8)
        preview_panel.pack(side="left", fill="both", expand=False, padx=(8, 0))
        self.preview_panel = preview_panel

        # Keep preview readable when the main window is narrowed.
        self.preview_panel.configure(width=430)
        self.preview_panel.pack_propagate(False)
        # Models panel is allowed to shrink first during horizontal resize.
        self.model_panel.configure(width=300)
        self.model_panel.pack_propagate(False)
        self.body_frame.bind("<Configure>", self.on_body_resize)

        preview_model_row = ttk.Frame(preview_panel)
        preview_model_row.pack(fill="x", pady=(0, 6))
        ttk.Label(preview_model_row, text="Preview model:").pack(side="left")
        self.preview_model_combo = ttk.Combobox(
            preview_model_row,
            textvariable=self.preview_model_var,
            state="readonly",
            width=22,
        )
        self.preview_model_combo.pack(side="left", padx=(6, 0))
        self.preview_model_combo.bind("<<ComboboxSelected>>", self.on_preview_model_changed)

        self.preview_label = ttk.Label(preview_panel, text="No preview")
        self.preview_label.pack(fill="both", expand=True)

        # Status
        status_frame = ttk.LabelFrame(main, text="Status", padding=8)
        status_frame.pack(fill="x", pady=(8, 0))

        self.status_text = tk.Text(status_frame, height=8, wrap="word")
        self.status_text.pack(fill="x")
        self.status_text.config(state="disabled")

    def _log(self, msg: str):
        self.status_text.config(state="normal")
        self.status_text.insert("end", msg + "\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")

    def on_body_resize(self, event):
        """When narrowing the window, shrink the model panel before shrinking preview."""
        total_w = max(200, int(event.width))

        preview_pref = 430
        preview_min = 360
        model_pref = 300
        model_min = 145
        sample_min = 320
        spacing_budget = 40

        preview_w = preview_pref
        room_for_model = total_w - preview_w - sample_min - spacing_budget
        model_w = min(model_pref, room_for_model)

        if model_w < model_min:
            deficit = model_min - model_w
            preview_w = max(preview_min, preview_w - deficit)
            room_for_model = total_w - preview_w - sample_min - spacing_budget
            model_w = min(model_pref, room_for_model)

        model_w = max(model_min, model_w)
        self.model_panel.configure(width=int(model_w))
        self.preview_panel.configure(width=int(preview_w))

    def _sync_selection_order(self, current_idxs: List[int]):
        current_set = set(current_idxs)
        self.selection_order = [i for i in self.selection_order if i in current_set]
        for idx in current_idxs:
            if idx not in self.current_selected_set and idx not in self.selection_order:
                self.selection_order.append(idx)
        self.current_selected_set = current_set

    def _get_selected_indices_in_order(self) -> List[int]:
        return [i for i in self.selection_order if i in self.current_selected_set]

    def _get_selected_keys_in_order(self) -> List[str]:
        return [self.sample_keys[i] for i in self._get_selected_indices_in_order() if i < len(self.sample_keys)]

    def _reset_sample_selection_state(self):
        self.selection_order = []
        self.current_selected_set = set()
        self.last_preview_idx = None
        self.selected_count_label.config(text="Selected samples: 0")
        self.preview_label.config(text="No preview", image="")

    def _refresh_model_tree(self):
        for iid in self.model_tree.get_children():
            self.model_tree.delete(iid)
        for i, m in enumerate(self.models):
            mask_text = str(m.mask_folder) if m.mask_folder else "(missing masks folder)"
            self.model_tree.insert("", "end", iid=str(i), values=(m.display_name, str(m.overlay_folder), mask_text))

    def _update_preview_model_options(self):
        labels = [m.display_name for m in self.models]
        self.preview_model_combo["values"] = labels
        if labels:
            if self.preview_model_var.get() not in labels:
                self.preview_model_var.set(labels[0])
        else:
            self.preview_model_var.set("")

    def _build_non_base_row_defaults(self) -> List[Tuple[str, str]]:
        """Return default (row_key, row_label) for rows that users can rename."""
        rows: List[Tuple[str, str]] = []
        if self.include_pred_var.get():
            for m in self.models:
                rows.append((f"pred::{m.display_name}", f"Pred ({m.display_name})"))
        if self.include_overlay_var.get():
            for m in self.models:
                rows.append((f"overlay::{m.display_name}", f"Overlay ({m.display_name})"))
        return rows

    def on_edit_row_labels(self):
        if not self.models:
            messagebox.showinfo("Edit Row Labels", "Please add at least one model first.")
            return

        defaults = self._build_non_base_row_defaults()
        if not defaults:
            messagebox.showinfo("Edit Row Labels", "Enable Pred rows and/or Overlay rows first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Edit Row Labels (Image/GT fixed)")
        win.geometry("620x420")
        win.transient(self.root)
        win.grab_set()

        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Customize left-side row labels for non-base rows. Leave empty to use default.",
        ).pack(anchor="w", pady=(0, 8))

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrolly = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)

        content.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrolly.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrolly.pack(side="right", fill="y")

        entry_vars: Dict[str, tk.StringVar] = {}
        for i, (row_key, default_label) in enumerate(defaults):
            ttk.Label(content, text=default_label).grid(row=i, column=0, sticky="w", padx=(0, 10), pady=4)
            cur = self.row_label_overrides.get(row_key, default_label)
            v = tk.StringVar(value=cur)
            entry_vars[row_key] = v
            ttk.Entry(content, textvariable=v, width=40).grid(row=i, column=1, sticky="we", pady=4)

        content.columnconfigure(1, weight=1)

        btns = ttk.Frame(win, padding=(10, 0, 10, 10))
        btns.pack(fill="x")

        def _save_and_close():
            # Refresh overrides only for current non-base row keys.
            new_overrides: Dict[str, str] = {}
            for row_key, default_label in defaults:
                txt = entry_vars[row_key].get().strip()
                if txt and txt != default_label:
                    new_overrides[row_key] = txt
            self.row_label_overrides = new_overrides
            self._log(f"Row label overrides updated: {len(new_overrides)} row(s).")
            win.destroy()

        ttk.Button(btns, text="Save", command=_save_and_close).pack(side="right")
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=(0, 6))

    def _auto_update_save_filename(self):
        labels = [sanitize_label_for_filename(m.display_name) for m in self.models]
        model_part = "_".join(labels[:4]) if labels else "no_model"
        base_name = f"comparison_{model_part}.png"

        cur = Path(self.save_var.get().strip())
        if not cur.name:
            cur = Path.cwd() / "comparison_figure.png"
        new_path = cur.with_name(base_name)
        self.save_var.set(str(new_path))

    def _save_last_models_setting(self):
        settings = load_app_settings()
        settings["last_models"] = [
            {
                "label": m.display_name,
                "overlay_folder": str(m.overlay_folder),
            }
            for m in self.models
        ]
        save_app_settings(settings)

    def _restore_last_models(self):
        settings = load_app_settings()
        last_models = settings.get("last_models", [])
        if not isinstance(last_models, list):
            return

        restored = 0
        for item in last_models:
            if not isinstance(item, dict):
                continue
            folder = str(item.get("overlay_folder", "")).strip()
            label = str(item.get("label", "")).strip() or "Model"
            p = Path(folder)
            if not p.exists() or not p.is_dir():
                continue
            try:
                model = load_model_from_overlay_folder(p, label)
            except Exception:
                continue
            if not model.overlay_index:
                continue
            self.models.append(model)
            restored += 1

        if restored > 0:
            self._refresh_model_tree()
            self._update_preview_model_options()
            self._rebuild_sample_list()
            self._auto_update_save_filename()
            self._log(f"Restored {restored} model(s) from previous session.")

    def _rebuild_sample_list(self):
        self.listbox.delete(0, "end")
        self.sample_keys = []
        self.sample_display_lines = []
        self._reset_sample_selection_state()

        if not self.models:
            return

        common_keys: Set[str] = set(self.models[0].overlay_index.keys())
        for m in self.models[1:]:
            common_keys &= set(m.overlay_index.keys())

        self.sample_keys = sorted(common_keys)
        preview_model = self._find_model_by_label(self.preview_model_var.get().strip())
        if preview_model is None and self.models:
            preview_model = self.models[0]

        for key in self.sample_keys:
            overlay_name = ""
            if preview_model is not None:
                p = preview_model.overlay_index.get(key)
                overlay_name = p.name if p is not None else ""
            line = f"{key}    |    {overlay_name}"
            self.sample_display_lines.append(line)
            self.listbox.insert("end", line)

        self._log(f"Samples available across all selected models: {len(self.sample_keys)}")

    def on_add_model(self):
        folder = filedialog.askdirectory(title="Choose overlay folder for a model")
        if not folder:
            return

        overlay_folder = Path(folder)
        default_label = overlay_folder.parent.name or overlay_folder.name or "Model"
        label = simpledialog.askstring(
            "Model Label",
            "Enter display label for this model (used in row names and file name):",
            initialvalue=default_label,
        )
        if label is None:
            return
        label = label.strip() or default_label

        try:
            model = load_model_from_overlay_folder(overlay_folder, label)
        except Exception as e:
            messagebox.showerror("Load model failed", f"Failed to load model folder:\n{e}")
            return

        if not model.overlay_index:
            messagebox.showwarning("No overlay files", "No supported overlay images found in selected folder.")
            return

        self.models.append(model)
        self._refresh_model_tree()
        self._update_preview_model_options()
        self._rebuild_sample_list()
        self._auto_update_save_filename()
        self._save_last_models_setting()

        self._log(f"Added model: {label}")
        self._log(f"Overlay folder: {overlay_folder}")
        if model.mask_folder is None:
            self._log("Warning: masks folder missing (expected sibling 'masks').")
        else:
            self._log(f"Mask folder: {model.mask_folder}")

    def on_remove_model(self):
        selected = self.model_tree.selection()
        if not selected:
            messagebox.showinfo("Remove model", "Please select a model row to remove.")
            return

        idx = int(selected[0])
        removed = self.models.pop(idx)
        self._refresh_model_tree()
        self._update_preview_model_options()
        self._rebuild_sample_list()
        self._auto_update_save_filename()
        self._save_last_models_setting()
        self._log(f"Removed model: {removed.display_name}")

    def on_rename_model_label(self):
        selected = self.model_tree.selection()
        if not selected:
            messagebox.showinfo("Rename label", "Please select a model row to rename.")
            return

        idx = int(selected[0])
        cur_label = self.models[idx].display_name
        new_label = simpledialog.askstring("Rename Label", "Enter new model label:", initialvalue=cur_label)
        if new_label is None:
            return
        new_label = new_label.strip()
        if not new_label:
            return

        self.models[idx].display_name = new_label
        self._refresh_model_tree()
        self._update_preview_model_options()
        self._auto_update_save_filename()
        self._save_last_models_setting()
        self._log(f"Renamed model label: {cur_label} -> {new_label}")

    def on_browse_gt_folder(self):
        folder = filedialog.askdirectory(title="Choose GT folder")
        if folder:
            self.gt_var.set(folder)
            self._log(f"GT folder set: {folder}")

    def on_browse_image_folder(self):
        folder = filedialog.askdirectory(title="Choose original image folder")
        if folder:
            self.img_var.set(folder)
            self._log(f"Image folder set: {folder}")

    def on_choose_save_path(self):
        path = filedialog.asksaveasfilename(
            title="Choose output PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if path:
            self.save_var.set(path)
            self._log(f"Save path set: {path}")

    def on_list_selection_change(self, _event=None):
        idxs = list(self.listbox.curselection())
        self._sync_selection_order(idxs)
        ordered_idxs = self._get_selected_indices_in_order()

        self.selected_count_label.config(text=f"Selected samples: {len(idxs)}")
        if not ordered_idxs:
            self.last_preview_idx = None
            self.preview_label.config(text="No preview", image="")
            return

        active_idx = self.listbox.index("active")
        if active_idx in self.current_selected_set:
            self.last_preview_idx = active_idx
        elif self.last_preview_idx not in self.current_selected_set:
            self.last_preview_idx = ordered_idxs[-1]

        self._update_preview_from_current_state()

    def on_list_click_release(self, event):
        if not self.sample_keys:
            return
        idx = self.listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.sample_keys):
            return
        if idx in self.current_selected_set:
            self.last_preview_idx = idx
            self._update_preview_from_current_state()

    def on_select_all(self):
        if not self.sample_keys:
            return
        self.listbox.select_set(0, "end")
        self.on_list_selection_change()
        self._log(f"Selected all samples: {len(self.sample_keys)}")

    def on_clear_selection(self):
        self.listbox.selection_clear(0, "end")
        self.selection_order = []
        self.current_selected_set = set()
        self.last_preview_idx = None
        self.on_list_selection_change()
        self._log("Selection cleared.")

    def _find_model_by_label(self, label: str) -> Optional[ModelSelection]:
        for m in self.models:
            if m.display_name == label:
                return m
        return None

    def _show_preview(self, image_path: Path):
        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                im.thumbnail((360, 360))
                self.preview_photo = ImageTk.PhotoImage(im)
                self.preview_label.config(image=self.preview_photo, text="")
        except Exception as e:
            self.preview_label.config(text=f"Preview failed:\n{e}", image="")

    def _update_preview_from_current_state(self):
        if self.last_preview_idx is None or self.last_preview_idx >= len(self.sample_keys):
            self.preview_label.config(text="No preview", image="")
            return

        key = self.sample_keys[self.last_preview_idx]
        model_label = self.preview_model_var.get().strip()
        model = self._find_model_by_label(model_label)
        if model is None and self.models:
            model = self.models[0]

        if model is None:
            self.preview_label.config(text="No preview", image="")
            return

        overlay_path = model.overlay_index.get(key)
        if overlay_path is None:
            self.preview_label.config(text="No overlay for selected model/key", image="")
            return

        self._show_preview(overlay_path)

    def on_preview_model_changed(self, _event=None):
        # Rebuild list text so it shows overlay filenames from the chosen model.
        if self.models:
            selected_in_order = self._get_selected_indices_in_order()
            self._rebuild_sample_list()
            for idx in selected_in_order:
                if 0 <= idx < len(self.sample_keys):
                    self.listbox.select_set(idx)
            self.on_list_selection_change()
        self._update_preview_from_current_state()

    def _build_columns_data(
        self,
        selected_keys: List[str],
        gt_folder: Path,
        img_folder: Path,
        include_pred: bool,
        include_overlay: bool,
    ) -> Tuple[List[Dict[str, Path]], List[Tuple[str, str]], List[str]]:
        """
        Build plotting data with robust missing checks.

        Returns:
        - columns_data
        - row_specs
        - warnings
        """
        warnings: List[str] = []
        columns_data: List[Dict[str, Path]] = []

        img_idx = build_key_index(img_folder)
        gt_idx = build_key_index(gt_folder)

        row_specs: List[Tuple[str, str]] = [("Image", "image"), ("GT", "gt")]

        if include_pred:
            for m in self.models:
                row_key = f"pred::{m.display_name}"
                default_label = f"Pred ({m.display_name})"
                row_label = self.row_label_overrides.get(row_key, default_label)
                row_specs.append((row_label, row_key))
        if include_overlay:
            for m in self.models:
                row_key = f"overlay::{m.display_name}"
                default_label = f"Overlay ({m.display_name})"
                row_label = self.row_label_overrides.get(row_key, default_label)
                row_specs.append((row_label, row_key))

        for key in selected_keys:
            col: Dict[str, Path] = {}

            img_path = img_idx.get(key)
            gt_path = gt_idx.get(key)
            if img_path is None or gt_path is None:
                miss = []
                if img_path is None:
                    miss.append("Image")
                if gt_path is None:
                    miss.append("GT")
                warnings.append(f"[{key}] missing -> {', '.join(miss)}")
                continue

            col["image"] = img_path
            col["gt"] = gt_path

            failed = False
            if include_pred:
                for m in self.models:
                    pred_path = m.pred_index.get(key)
                    if pred_path is None:
                        warnings.append(f"[{key}] missing -> Pred ({m.display_name})")
                        failed = True
                        break
                    col[f"pred::{m.display_name}"] = pred_path

            if failed:
                continue

            if include_overlay:
                for m in self.models:
                    ov_path = m.overlay_index.get(key)
                    if ov_path is None:
                        warnings.append(f"[{key}] missing -> Overlay ({m.display_name})")
                        failed = True
                        break
                    col[f"overlay::{m.display_name}"] = ov_path

            if failed:
                continue

            columns_data.append(col)

        return columns_data, row_specs, warnings

    def on_generate_figure(self):
        if not self.models:
            messagebox.showwarning("No model", "Please add at least one model overlay folder.")
            return

        include_pred = bool(self.include_pred_var.get())
        include_overlay = bool(self.include_overlay_var.get())
        if not include_pred and not include_overlay:
            messagebox.showwarning("No output rows", "Please enable Pred rows and/or Overlay rows.")
            return

        gt_folder = Path(self.gt_var.get().strip())
        img_folder = Path(self.img_var.get().strip())
        if not gt_folder.exists() or not gt_folder.is_dir():
            messagebox.showerror("Invalid GT folder", "GT folder does not exist.")
            return
        if not img_folder.exists() or not img_folder.is_dir():
            messagebox.showerror("Invalid image folder", "Original image folder does not exist.")
            return

        selected_keys = self._get_selected_keys_in_order()
        if not selected_keys:
            messagebox.showwarning("No selection", "Please select one or more samples.")
            return

        try:
            wspace = float(self.wspace_var.get().strip())
            hspace = float(self.hspace_var.get().strip())
            cell_height = float(self.cell_height_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid spacing", "wspace / hspace / cellH must be numeric values.")
            return
        if wspace < 0 or hspace < 0:
            messagebox.showerror("Invalid spacing", "wspace and hspace must be >= 0.")
            return
        if cell_height <= 0:
            messagebox.showerror("Invalid cellH", "cellH must be > 0.")
            return

        self._log(f"Selected samples: {len(selected_keys)}")

        columns_data, row_specs, warnings = self._build_columns_data(
            selected_keys=selected_keys,
            gt_folder=gt_folder,
            img_folder=img_folder,
            include_pred=include_pred,
            include_overlay=include_overlay,
        )

        if warnings:
            self._log("---- Missing file report ----")
            for w in warnings:
                self._log(w)

        if not columns_data:
            messagebox.showerror("No matched samples", "No complete matched sample set found. Check status messages.")
            return

        try:
            fig = generate_comparison_figure(
                columns_data=columns_data,
                row_specs=row_specs,
                wspace=wspace,
                hspace=hspace,
                cell_height=cell_height,
            )
        except Exception as e:
            messagebox.showerror("Figure error", f"Failed to generate figure:\n{e}")
            return

        out_path = Path(self.save_var.get().strip())
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        try:
            save_figure(fig, out_path)
            plt.close(fig)
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save figure:\n{e}")
            return

        self._log(f"Figure saved successfully: {out_path}")
        self._log(f"Matched samples used: {len(columns_data)}")
        if warnings:
            self._log(f"Skipped due to missing files: {len(warnings)}")

        messagebox.showinfo("Done", f"Figure saved:\n{out_path}")


def main():
    root = tk.Tk()
    app = SemAogComparisonApp(root)
    app._log("GUI ready. Add one or more model overlay folders.")
    root.mainloop()


if __name__ == "__main__":
    main()
