from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABEL_H = 20

try:
    FONT = ImageFont.load_default(size=14)
except TypeError:  # older Pillow
    FONT = ImageFont.load_default()


def _tile(frame, label, vmin, vmax):
    f = frame.astype(np.float32)
    a = vmin if vmin is not None else np.nanmin(f)
    b = vmax if vmax is not None else np.nanmax(f)
    u8 = (np.clip((f - a) / ((b - a) or 1.0), 0, 1) * 255).astype(np.uint8)

    H, W = u8.shape
    tile = np.zeros((H + LABEL_H, W, 3), np.uint8)
    tile[:H] = u8[..., None]

    img = Image.fromarray(tile)
    ImageDraw.Draw(img).text(
        (W // 2, H + LABEL_H // 2), label, fill=(255, 255, 255), anchor="mm", font=FONT
    )
    return np.asarray(img)


def save_trials_grid_video(
    data,  # shape (M, Trials, T, H, W)
    save_path,
    mice=(1, 2),  # 1-indexed like your UI
    trials=((1, 2, 3), (1, 2, 3)),  # 1-indexed
    fps=25,
    global_scale=True,
    label="",
):
    data = np.asarray(data)
    T, H, W = data.shape[2:]
    ncols = max(len(tr) for tr in trials)

    grid = [
        [(f"M{m} T{t}", data[m - 1, t - 1]) for t in tr] + [None] * (ncols - len(tr))
        for m, tr in zip(mice, trials)
    ]
    selected = [cell[1] for row in grid for cell in row if cell]

    vmin = min(np.nanmin(s) for s in selected) if global_scale else None
    vmax = max(np.nanmax(s) for s in selected) if global_scale else None

    blank = np.zeros((H + LABEL_H, W, 3), np.uint8)

    save_path = Path(save_path)

    with imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8) as w:
        for k in range(T):
            frame = np.concatenate(
                [
                    np.concatenate(
                        [_tile(c[1][k], label, vmin, vmax) if c else blank for c in row],
                        axis=1,
                    )
                    for row in grid
                ],
                axis=0,
            )
            frame = np.pad(
                frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0))
            )
            w.append_data(frame)

    return save_path