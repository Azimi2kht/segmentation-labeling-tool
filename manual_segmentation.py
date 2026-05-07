import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider


@dataclass
class LabelInfo:
    label_id: int
    name: str
    color: Tuple[float, float, float, float]


class ManualHSISegmenter:
    def __init__(self, npy_path: str, output_mask_path: str):
        self.npy_path = npy_path
        self.output_mask_path = output_mask_path

        self.cube = np.load(self.npy_path).astype(np.float32)  # (H, W, B)
        if self.cube.ndim != 3:
            raise ValueError(f"Expected HSI cube with shape (H, W, B), got {self.cube.shape}")
        self.h, self.w, self.bands = self.cube.shape

        # Final mask shared for all bands. 0 = background.
        self.mask = np.zeros((self.h, self.w), dtype=np.int32)

        self.current_band = 0
        self.current_label = 1
        self.brush_radius = 0  # 0 means single pixel.
        self.mouse_down = False
        self.mouse_button = None
        self.is_panning = False
        self.pan_mode = False
        self.last_pan_xy = None
        self.undo_stack = []
        self.max_undo_steps = 30
        self.max_labels = 255

        self.labels: Dict[int, LabelInfo] = {
            0: LabelInfo(0, "background", (0.0, 0.0, 0.0, 0.0))
        }
        for label_id in range(1, 11):
            self.labels[label_id] = LabelInfo(
                label_id,
                f"label_{label_id}",
                self._next_label_color(label_id),
            )

        # Prevent Matplotlib default "s" action (save figure) from overriding mask save.
        plt.rcParams["keymap.save"] = []
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(left=0.24, bottom=0.22)
        self.base_im = None
        self.mask_im = None
        self.band_slider = None
        self._is_updating_slider = False
        self.buttons = {}
        self._draw_initial()
        self._create_controls()
        self._connect_events()

    def _normalize_band(self, band_img: np.ndarray) -> np.ndarray:
        mn = np.min(band_img)
        mx = np.max(band_img)
        return (band_img - mn) / (mx - mn + 1e-8)

    def _make_overlay_cmap(self) -> ListedColormap:
        max_label = max(self.labels.keys())
        colors = []
        for i in range(max_label + 1):
            if i in self.labels:
                colors.append(self.labels[i].color)
            else:
                colors.append((1.0, 1.0, 1.0, 0.45))
        return ListedColormap(colors)

    def _draw_initial(self):
        band_img = self._normalize_band(self.cube[:, :, self.current_band])
        self.base_im = self.ax.imshow(band_img, cmap="gray", interpolation="nearest")

        cmap = self._make_overlay_cmap()
        self.mask_im = self.ax.imshow(
            self.mask,
            cmap=cmap,
            vmin=0,
            vmax=max(self.labels.keys()),
            interpolation="nearest",
        )
        self.ax.set_title(self._title_text())
        self.ax.set_axis_off()

    def _create_controls(self):
        self.fig.text(
            0.02,
            0.92,
            self._shortcut_bullets(),
            fontsize=9,
            va="top",
            ha="left",
            family="monospace",
        )

        slider_ax = self.fig.add_axes([0.28, 0.09, 0.68, 0.04])
        self.band_slider = Slider(
            ax=slider_ax,
            label="Band / Frame",
            valmin=0,
            valmax=self.bands - 1,
            valinit=self.current_band,
            valstep=1,
        )
        self.band_slider.on_changed(self._on_slider_change)

        btn_w = 0.085
        btn_h = 0.05
        y = 0.02
        x_positions = [0.24, 0.315, 0.39, 0.465, 0.54, 0.615, 0.69, 0.765, 0.84, 0.915]
        specs = [
            ("prev_band", "◀ Band", self._prev_band),
            ("next_band", "Band ▶", self._next_band),
            ("brush_minus", "− Brush", self._brush_minus),
            ("brush_plus", "+ Brush", self._brush_plus),
            ("prev_label", "◀ Label", self._prev_label),
            ("next_label", "Label ▶", self._next_label),
            ("add_label", "+ Label", self._add_label_button),
            ("undo", "↶ Undo", self._undo_button),
            ("pan_mode", "Pan Off", self._toggle_pan_mode_button),
            ("save", "Save", self._save_button),
        ]

        for x, (key, text, callback) in zip(x_positions, specs):
            ax_btn = self.fig.add_axes([x, y, btn_w, btn_h])
            btn = Button(ax_btn, text)
            btn.on_clicked(callback)
            self.buttons[key] = btn

    def _title_text(self) -> str:
        return (
            f"Band {self.current_band + 1}/{self.bands} | "
            f"Label {self.current_label} ({self.labels[self.current_label].name}) | "
            f"Brush radius={self.brush_radius}px | Pan={'ON' if self.pan_mode else 'OFF'}"
        )

    def _shortcut_bullets(self) -> str:
        return "\n".join(
            [
                "Shortcuts",
                "- Left drag: paint",
                "- Right drag: erase",
                "- [ / ] : prev / next band",
                "- +/- : brush size",
                "- n : add label",
                "- u : undo",
                "- m : pan mode",
                "- s : save mask",
                "- q : quit",
                "- Wheel : zoom in/out",
            ]
        )

    def _refresh(self):
        band_img = self._normalize_band(self.cube[:, :, self.current_band])
        self.base_im.set_data(band_img)
        self.mask_im.set_data(self.mask)
        self.mask_im.set_cmap(self._make_overlay_cmap())
        self.mask_im.set_clim(0, max(self.labels.keys()))
        if self.band_slider is not None:
            self._is_updating_slider = True
            self.band_slider.set_val(self.current_band)
            self._is_updating_slider = False
        self.ax.set_title(self._title_text())
        self.fig.canvas.draw_idle()

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _apply_brush(self, x: int, y: int, label_value: int):
        r = self.brush_radius
        x_min = max(0, x - r)
        x_max = min(self.w - 1, x + r)
        y_min = max(0, y - r)
        y_max = min(self.h - 1, y + r)

        if r == 0:
            self.mask[y, x] = label_value
            return

        yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= r**2
        patch = self.mask[y_min : y_max + 1, x_min : x_max + 1]
        patch[circle] = label_value

    def _event_to_xy(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return None
        return x, y

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.pan_mode and event.button == 1:
            self.is_panning = True
            self.last_pan_xy = (event.xdata, event.ydata)
            return

        pos = self._event_to_xy(event)
        if pos is None:
            return
        self.mouse_down = True
        self.mouse_button = event.button
        if event.button in (1, 3):
            self._push_undo_state()
        x, y = pos
        if event.button == 1:  # left
            self._apply_brush(x, y, self.current_label)
        elif event.button == 3:  # right
            self._apply_brush(x, y, 0)
        self._refresh()

    def _on_mouse_release(self, event):
        self.mouse_down = False
        self.mouse_button = None
        self.is_panning = False
        self.last_pan_xy = None

    def _on_mouse_move(self, event):
        if self.is_panning and self.pan_mode and event.inaxes == self.ax:
            if self.last_pan_xy is None or event.xdata is None or event.ydata is None:
                return
            prev_x, prev_y = self.last_pan_xy
            dx = event.xdata - prev_x
            dy = event.ydata - prev_y
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
            self.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
            self.last_pan_xy = (event.xdata, event.ydata)
            self.fig.canvas.draw_idle()
            return

        if not self.mouse_down:
            return
        pos = self._event_to_xy(event)
        if pos is None:
            return
        x, y = pos
        if self.mouse_button == 1:
            self._apply_brush(x, y, self.current_label)
        elif self.mouse_button == 3:
            self._apply_brush(x, y, 0)
        self._refresh()

    def _zoom(self, event, scale_factor: float):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        x_center, y_center = event.xdata, event.ydata

        x_left = x_center - (x_center - cur_xlim[0]) / scale_factor
        x_right = x_center + (cur_xlim[1] - x_center) / scale_factor
        y_bottom = y_center - (y_center - cur_ylim[0]) / scale_factor
        y_top = y_center + (cur_ylim[1] - y_center) / scale_factor

        self.ax.set_xlim([x_left, x_right])
        self.ax.set_ylim([y_bottom, y_top])
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.button == "up":
            self._zoom(event, 1.2)  # zoom in
        elif event.button == "down":
            self._zoom(event, 1 / 1.2)  # zoom out

    def _on_slider_change(self, value):
        if self._is_updating_slider:
            return
        self.current_band = int(value)
        self._refresh()

    def _prev_band(self, _event=None):
        self.current_band = max(0, self.current_band - 1)
        self._refresh()

    def _next_band(self, _event=None):
        self.current_band = min(self.bands - 1, self.current_band + 1)
        self._refresh()

    def _brush_minus(self, _event=None):
        self.brush_radius = max(0, self.brush_radius - 1)
        self._refresh()

    def _brush_plus(self, _event=None):
        self.brush_radius = min(20, self.brush_radius + 1)
        self._refresh()

    def _available_labels(self):
        return sorted(self.labels.keys())

    def _prev_label(self, _event=None):
        labels = self._available_labels()
        idx = labels.index(self.current_label)
        self.current_label = labels[(idx - 1) % len(labels)]
        self._refresh()

    def _next_label(self, _event=None):
        labels = self._available_labels()
        idx = labels.index(self.current_label)
        self.current_label = labels[(idx + 1) % len(labels)]
        self._refresh()

    def _add_label_button(self, _event=None):
        self._add_label_interactive()

    def _save_button(self, _event=None):
        self._save_outputs()

    def _push_undo_state(self):
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

    def _undo(self):
        if not self.undo_stack:
            print("Undo stack is empty.")
            return
        self.mask = self.undo_stack.pop()
        self._refresh()

    def _undo_button(self, _event=None):
        self._undo()

    def _set_pan_mode(self, enabled: bool):
        self.pan_mode = enabled
        if "pan_mode" in self.buttons:
            self.buttons["pan_mode"].label.set_text("Pan On" if enabled else "Pan Off")
        self._refresh()

    def _toggle_pan_mode_button(self, _event=None):
        self._set_pan_mode(not self.pan_mode)

    def _next_label_color(self, label_id: int):
        palette = [
            (1.0, 0.0, 0.0, 0.45),
            (0.0, 1.0, 0.0, 0.45),
            (0.0, 0.4, 1.0, 0.45),
            (1.0, 0.65, 0.0, 0.45),
            (0.6, 0.2, 1.0, 0.45),
            (0.0, 0.8, 0.8, 0.45),
            (1.0, 0.1, 0.6, 0.45),
            (0.6, 0.6, 0.0, 0.45),
        ]
        return palette[(label_id - 1) % len(palette)]

    def _add_label_interactive(self):
        if max(self.labels.keys()) >= self.max_labels:
            print(f"Cannot add more labels. Maximum is {self.max_labels}.")
            return
        new_id = max(self.labels.keys()) + 1
        name = f"label_{new_id}"
        color = self._next_label_color(new_id)

        self.labels[new_id] = LabelInfo(new_id, name, color)
        self.current_label = new_id
        print(f"Added label {new_id}: {name} with color {color}")
        self._refresh()

    def _save_outputs(self):
        abs_path = os.path.abspath(self.output_mask_path)
        root, ext = os.path.splitext(abs_path)
        if not ext:
            ext = ".npy"
        band_path = f"{root}_band{self.current_band:03d}{ext}"
        os.makedirs(os.path.dirname(band_path), exist_ok=True)
        np.save(band_path, self.mask.astype(np.int32))
        print(f"Saved final segmentation mask to: {band_path}")

    def _on_key_press(self, event):
        if event.key in ["]", "right"]:
            self._next_band()
            return
        elif event.key in ["[", "left"]:
            self._prev_band()
            return
        elif event.key in ["+", "="]:
            self._brush_plus()
            return
        elif event.key in ["-", "_"]:
            self._brush_minus()
            return
        elif event.key == "n":
            self._add_label_interactive()
        elif event.key == "u":
            self._undo()
            return
        elif event.key == "m":
            self._set_pan_mode(not self.pan_mode)
            return
        elif event.key == "s":
            self._save_outputs()
        elif event.key == "q":
            plt.close(self.fig)
            return
        elif event.key and event.key.isdigit():
            digit = int(event.key)
            if digit in self.labels:
                self.current_label = digit
            else:
                print(f"Label {digit} not defined. Press 'n' to add a new label.")
        self._refresh()

    def run(self):
        print(f"Loaded cube from {self.npy_path} with shape (H={self.h}, W={self.w}, B={self.bands})")
        print("Manual segmentation started.")
        print("Controls:")
        print("  Left drag = paint current label, Right drag = erase to background")
        print("  [ / ] or left/right arrows = previous/next band")
        print("  Slider = jump directly to a specific band/frame")
        print("  +/- = decrease/increase brush radius (0 means single-pixel brush)")
        print("  m = toggle pan mode, u = undo last stroke")
        print("  Buttons = previous/next band, brush +/- , label +/- , add label, undo, pan mode, save")
        print("  0..9 = select existing label id")
        print("  n = add new label (auto color)")
        print("  s = save mask only (.npy)")
        print("  q = quit")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Manual HSI segmentation tool")
    parser.add_argument(
        "--npy_path",
        type=str,
        default="./exp6_003_preprocessed_cropped_rotated/frame_0160_radiance_cropped_rotated.npy",
        help="Path to HSI cube in .npy format with shape (H, W, B)",
    )
    parser.add_argument(
        "--out_mask",
        type=str,
        default=None,
        help="Output .npy path for final class mask (H, W). Defaults to ./output/<same_input_filename>.npy",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.npy_path):
        raise FileNotFoundError(f"Input file not found: {args.npy_path}")

    if args.out_mask:
        out_mask = args.out_mask
    else:
        output_dir = os.path.join(os.getcwd(), "output")
        input_filename = os.path.basename(args.npy_path)
        out_mask = os.path.join(output_dir, input_filename)

    app = ManualHSISegmenter(
        npy_path=args.npy_path,
        output_mask_path=out_mask,
    )
    app.run()


if __name__ == "__main__":
    main()
