\
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict
import os

def _get_font(size: int = 16):
    # Use a default font that exists. PIL will fallback if not available.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def line_plot(y: np.ndarray,
              title: str,
              xlabel: str,
              ylabel: str,
              path: str,
              width: int = 900,
              height: int = 520,
              margin: int = 60):
    """
    Minimal line plot without matplotlib, using PIL.
    y: shape (K,) or (K,S)
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    K, S = y.shape

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fontT = _get_font(18)
    font = _get_font(14)

    # Axes area
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin

    # compute y range
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    # draw axes
    d.line([(x0, y0), (x1, y0)], fill=(0, 0, 0), width=2)
    d.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=2)

    # ticks
    for i in range(6):
        xt = x0 + (x1-x0)*i/5
        d.line([(xt, y0), (xt, y0+5)], fill=(0,0,0), width=1)
        label = str(int(round((K-1)*i/5)))
        d.text((xt-10, y0+8), label, fill=(0,0,0), font=font)
    for i in range(6):
        yt = y0 - (y0-y1)*i/5
        d.line([(x0-5, yt), (x0, yt)], fill=(0,0,0), width=1)
        val = ymin + (ymax-ymin)*i/5
        d.text((5, yt-8), f"{val:.2f}", fill=(0,0,0), font=font)

    # plot series
    colors = [(0,90,160), (160,60,0), (0,140,60), (120,0,140), (90,90,0), (0,0,0)]
    for s in range(S):
        pts = []
        for k in range(K):
            x = x0 + (x1-x0)*k/max(1, K-1)
            yv = y[k, s]
            yy = y0 - (y0-y1)*(yv - ymin)/(ymax - ymin)
            pts.append((x, yy))
        d.line(pts, fill=colors[s % len(colors)], width=2)

    d.text((margin, 10), title, fill=(0,0,0), font=fontT)
    d.text((width//2-20, height-margin+30), xlabel, fill=(0,0,0), font=font)
    d.text((10, margin-40), ylabel, fill=(0,0,0), font=font)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def voltage_profile_plot(V: np.ndarray, title: str, path: str):
    """V shape (N,)"""
    V = np.asarray(V).reshape(-1)
    N = V.size
    line_plot(V, title=title, xlabel="bus index", ylabel="V(pu)", path=path)

def bar_plot(values: np.ndarray, title: str, xlabel: str, ylabel: str, path: str,
             width: int = 900, height: int = 520, margin: int = 60):
    values = np.asarray(values).reshape(-1)
    n = values.size
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fontT = _get_font(18); font = _get_font(14)
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    d.line([(x0, y0), (x1, y0)], fill=(0, 0, 0), width=2)
    d.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=2)
    vmin = float(np.min(values)); vmax = float(np.max(values))
    vmax = max(vmax, 1e-6)
    # zero line
    y_zero = y0 - (y0-y1)*(0 - vmin)/(vmax - vmin) if vmin < 0 else y0
    d.line([(x0, y_zero), (x1, y_zero)], fill=(200,200,200), width=1)

    barw = (x1-x0)/max(1,n)
    for i,val in enumerate(values):
        x_left = x0 + i*barw + 2
        x_right = x0 + (i+1)*barw - 2
        y_val = y0 - (y0-y1)*(val - vmin)/(vmax - vmin)
        d.rectangle([(x_left, y_val), (x_right, y_zero)], outline=(0,0,0), fill=(0,90,160))
    d.text((margin, 10), title, fill=(0,0,0), font=fontT)
    d.text((width//2-20, height-margin+30), xlabel, fill=(0,0,0), font=font)
    d.text((10, margin-40), ylabel, fill=(0,0,0), font=font)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
