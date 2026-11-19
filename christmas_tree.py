"""Unified 3D ASCII Christmas Tree, Procedural Soundtrack & 9:16 Reel Exporter."""

from __future__ import annotations

import argparse
import io
import math
import os
import shutil
import subprocess
import tempfile
import time
import tokenize
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio_ffmpeg
import numpy as np
import pygame

try:
    import cv2
except ImportError:
    cv2 = None

TAU = 2.0 * np.pi

# Scene & animation choreography (seconds)
DURATION = 20.0
T_GROW = (0.7, 5.6)
T_GARLAND = (6.4, 8.6)
T_BALLS = (7.4, 9.5)
T_LIGHTS = (8.3, 11.3)
T_GIFTS = (10.0, 11.5)
T_STAR = 11.8
T_MSG = 13.0

# 3D world bounds
GROUND = -1.12
Y0 = -0.76
Y1 = 1.24
ROT_SPEED = 0.31  # rad/s (~355 deg over 20s)

# Material kinds
NEEDLE, SNOW, CONE, TRUNK, TINSEL, BALL, LIGHT, GIFT, RIBBON = range(9)

# Colors (normalized 0..1 RGB)
WHITE = np.array([1.00, 1.00, 1.00], dtype=np.float32)
SNOW_COL = np.array([0.88, 0.94, 1.00], dtype=np.float32)
GOLD = np.array([1.00, 0.84, 0.28], dtype=np.float32)
BAUBLE_GOLD = np.array([1.00, 0.78, 0.24], dtype=np.float32)
RED = np.array([0.96, 0.20, 0.24], dtype=np.float32)
WARM = np.array([1.00, 0.88, 0.50], dtype=np.float32)
GREEN_BOX = np.array([0.18, 0.78, 0.34], dtype=np.float32)
BLUE_BOX = np.array([0.22, 0.48, 0.98], dtype=np.float32)
SILVER = np.array([0.86, 0.90, 0.98], dtype=np.float32)

# Luminance-ordered ASCII ramp
RAMP = " .,:;-~=+*^oxv%&#W@"
RAMP_ARR = np.frombuffer(RAMP.encode("ascii"), dtype=np.uint8)

# Default window layout
TITLE = "ASCII Christmas Tree"
CW, CH, FONT_PX = 9, 14, 15
COLS, ROWS = 88, 74
W, H = COLS * CW, ROWS * CH
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
HINT = "Space pause · R restart · S screenshot · Left/Right scrub · Esc quit"
BG_RGB = (5, 8, 15)
DIM, ACCENT, SIGN = (96, 104, 122), (255, 196, 90), (176, 168, 196)
DEFAULT_AUTHOR = "@cfs.ascii"
DEFAULT_MESSAGE = "M E R R Y   C H R I S T M A S   2 0 2 6"

# Split-screen code pane palette
CODE_BG = (8, 11, 20)
HEADER_BG = (15, 18, 29)
BORDER_COL = (26, 32, 48)
HIGHLIGHT_BG = (20, 26, 44)
LINE_NUM_COL = (74, 83, 102)
HEADER_TEXT_COL = (214, 222, 238)

SYNTAX_COLORS = {
    "keyword": (244, 114, 152),
    "builtin": (125, 185, 252),
    "call": (125, 185, 252),
    "string": (134, 224, 148),
    "number": (245, 178, 102),
    "comment": (102, 113, 134),
    "op": (168, 178, 198),
    "ident": (224, 232, 246),
}

PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
}

PY_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "ord", "chr", "print", "range", "round", "set",
    "str", "sum", "tuple", "zip", "np", "pygame", "math", "cv2",
}

# Audio synthesis constants
SAMPLE_RATE = 44100

NOTE_MAP = {
    "A1": 55.00, "B1": 61.74,
    "C#2": 69.30, "D2": 73.42, "E2": 82.41, "F#2": 92.50, "G2": 98.00, "A2": 110.00, "B2": 123.47,
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00, "A3": 220.00, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00, "A4": 440.00, "Bb4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G5": 783.99, "A5": 880.00, "B5": 987.77,
    "C6": 1046.50, "C#6": 1108.73, "D6": 1174.66, "E6": 1318.51, "F#6": 1479.98, "G6": 1567.98, "A6": 1760.00,
}


# ---------------------------------------------------------------------------
# 3D ASCII Scene Engine
# ---------------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-8)


def smooth(x: float | np.ndarray) -> float | np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = i * 2.39996323
    return np.stack([np.cos(phi) * r, z, np.sin(phi) * r], axis=1)


def envelope(y: float | np.ndarray) -> float | np.ndarray:
    u = np.clip((y - Y0) / (Y1 - Y0), 0.0, 1.0)
    tiers = 1.0 + 0.07 * np.sin(u * TAU * 5.0) * (1.0 - u)
    return 0.96 * ((1.0 - u) ** 0.90) * tiers


def grow_time(y: float | np.ndarray) -> float | np.ndarray:
    u = np.clip((y - GROUND) / (Y1 - GROUND), 0.0, 1.0)
    return T_GROW[0] + (T_GROW[1] - T_GROW[0]) * u


def point_in_poly(px: np.ndarray, py: np.ndarray, poly: Sequence[tuple[float, float]]) -> np.ndarray:
    res = np.zeros_like(px, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (y1 > py) != (y2 > py)
        x_int = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
        res ^= cross & (px < x_int)
    return res


class Parts:
    """Collects 3D points and compiles them into contiguous NumPy arrays."""

    def __init__(self) -> None:
        self._pos: list[np.ndarray] = []
        self._nrm: list[np.ndarray] = []
        self._col: list[np.ndarray] = []
        self._kind: list[np.ndarray] = []
        self._t0: list[np.ndarray] = []
        self._base: list[np.ndarray] = []
        self._centre: list[np.ndarray] = []
        self._phase: list[np.ndarray] = []
        self._pop: list[np.ndarray] = []

    def add(
        self,
        pos: np.ndarray,
        nrm: np.ndarray,
        col: np.ndarray | tuple[float, float, float],
        kind: int | np.ndarray,
        t0: float | np.ndarray,
        base: float | np.ndarray = 1.0,
        centre: np.ndarray | None = None,
        phase: float | np.ndarray = 0.0,
        pop: bool = False,
    ) -> None:
        pos = np.asarray(pos, dtype=np.float32)
        n = len(pos)
        nrm = unit(np.asarray(nrm, dtype=np.float32))
        col_arr = np.asarray(col, dtype=np.float32)
        if col_arr.ndim == 1:
            col_arr = np.broadcast_to(col_arr, (n, 3)).copy()
        kind_arr = np.broadcast_to(np.asarray(kind, dtype=np.int32), (n,)).copy()
        t0_arr = np.broadcast_to(np.asarray(t0, dtype=np.float32), (n,)).copy()
        base_arr = np.broadcast_to(np.asarray(base, dtype=np.float32), (n,)).copy()
        if centre is None:
            cen_arr = pos.copy()
        else:
            cen_arr = np.broadcast_to(np.asarray(centre, dtype=np.float32), (n, 3)).copy()
        phase_arr = np.broadcast_to(np.asarray(phase, dtype=np.float32), (n,)).copy()
        pop_arr = np.full(n, pop, dtype=bool)

        self._pos.append(pos)
        self._nrm.append(nrm)
        self._col.append(col_arr)
        self._kind.append(kind_arr)
        self._t0.append(t0_arr)
        self._base.append(base_arr)
        self._centre.append(cen_arr)
        self._phase.append(phase_arr)
        self._pop.append(pop_arr)

    def compile(self) -> SceneData:
        return SceneData(
            pos=np.concatenate(self._pos, axis=0),
            nrm=np.concatenate(self._nrm, axis=0),
            col=np.concatenate(self._col, axis=0),
            kind=np.concatenate(self._kind, axis=0),
            t0=np.concatenate(self._t0, axis=0),
            base=np.concatenate(self._base, axis=0),
            centre=np.concatenate(self._centre, axis=0),
            phase=np.concatenate(self._phase, axis=0),
            pop=np.concatenate(self._pop, axis=0),
        )


@dataclass
class SceneData:
    pos: np.ndarray
    nrm: np.ndarray
    col: np.ndarray
    kind: np.ndarray
    t0: np.ndarray
    base: np.ndarray
    centre: np.ndarray
    phase: np.ndarray
    pop: np.ndarray


def build() -> SceneData:
    rng = np.random.default_rng(7)
    q = Parts()
    tips: list[np.ndarray] = []

    # needles: whorls of drooping branches, denser and longer at the bottom
    for i, y in enumerate(np.linspace(Y0 + 0.02, Y1 - 0.08, 25)):
        u = (y - Y0) / (Y1 - Y0)
        R = envelope(y) + 0.05
        nb = int(7 + 12 * (1.0 - u))
        for b in range(nb):
            a = TAU * (b + 0.5 * rng.random()) / nb + i * 0.77
            L = R * rng.uniform(0.85, 1.08)
            n = int(26 + 240 * L)
            s = rng.random(n) ** 0.65  # 0 trunk .. 1 tip
            d = np.array([np.cos(a), 0.0, np.sin(a)], dtype=np.float32)
            side = np.array([-np.sin(a), 0.0, np.cos(a)], dtype=np.float32)
            spread = (0.025 + 0.09 * s) * rng.normal(0.0, 1.0, n)
            sl = s * L
            p = d * sl[:, None] + side * spread[:, None]
            p[:, 1] = y + 0.06 * sl - 0.20 * (sl ** 2) + 0.012 * rng.normal(0.0, 1.0, n)
            nrm = unit(d * 0.75 + np.array([0.0, 0.55, 0.0]) + 0.3 * rng.normal(0.0, 1.0, (n, 3)))
            ao = (0.30 + 0.70 * (s ** 1.3)) * rng.uniform(0.75, 1.1, n)
            snow = (s > 0.5) & (rng.random(n) < 0.07 * s) & (nrm[:, 1] > 0.5)
            green = np.array([0.10, 0.48, 0.18]) + np.array([0.12, 0.38, 0.14]) * rng.random((n, 1))
            q.add(
                p,
                nrm,
                np.where(snow[:, None], SNOW_COL, green),
                np.where(snow, SNOW, NEEDLE),
                grow_time(y) + 0.45 * s,
                base=np.where(snow, 1.0, ao),
            )
            tl = 0.93 * L
            tips.append(d * tl + np.array([0.0, y + 0.06 * tl - 0.20 * (tl ** 2), 0.0]))

    # dark inner cone so nothing shows through the branches
    m = 4200
    y = Y0 + (Y1 - 0.1 - Y0) * rng.random(m)
    phi = rng.random(m) * TAU
    out = np.stack([np.cos(phi), np.full(m, 0.3), np.sin(phi)], axis=1)
    r = 0.6 * envelope(y)
    q.add(np.stack([np.cos(phi) * r, y, np.sin(phi) * r], axis=1), out, (0.06, 0.28, 0.10), CONE, grow_time(y), base=0.55)

    # trunk
    m = 1000
    y, phi = GROUND + rng.random(m) * (Y0 + 0.1 - GROUND), rng.random(m) * TAU
    n = np.stack([np.cos(phi), np.zeros(m), np.sin(phi)], axis=1)
    bark = 0.75 + 0.25 * np.sign(np.sin(phi * 30.0 + y * 5.0))
    q.add(
        n * 0.08 + np.stack([np.zeros(m), y, np.zeros(m)], axis=1),
        n,
        np.array([0.42, 0.24, 0.12]) * bark[:, None],
        TRUNK,
        grow_time(y),
    )

    # tinsel garland: a gold spiral drooping in swags between hanging points
    u = np.linspace(0.0, 1.0, 1800)
    turns = 3.6
    yg = Y0 + 0.15 + (Y1 - Y0 - 0.45) * u
    ag = TAU * turns * u + 0.9
    swag = -0.07 * np.abs(np.sin(ag * 2.5))
    rg = envelope(yg) + 0.07
    line = np.stack([np.cos(ag) * rg, yg + swag, np.sin(ag) * rg], axis=1)
    for _ in range(3):
        nrm = np.stack([np.cos(ag), np.full_like(u, 0.3), np.sin(ag)], axis=1) + 0.6 * rng.normal(0.0, 1.0, line.shape)
        q.add(
            line + 0.016 * rng.normal(0.0, 1.0, line.shape),
            nrm,
            GOLD,
            TINSEL,
            T_GARLAND[0] + u * (T_GARLAND[1] - T_GARLAND[0]),
            phase=rng.random(len(u)) * TAU,
        )

    # baubles: red, gold, and silver spheres nestled on the tree silhouette
    sphere = fib_sphere(160)
    for k in range(34):
        u = (k + rng.random()) / 34.0
        y = Y0 + 0.08 + u * (Y1 - Y0 - 0.38)
        a = k * 2.39996 + rng.random() * 0.4
        c = np.array([np.cos(a), 0.0, np.sin(a)]) * (envelope(y) + 0.02) + np.array([0.0, y - 0.03, 0.0])
        rad = 0.058 + 0.032 * (1.0 - u)
        col = RED if k % 2 == 0 else (BAUBLE_GOLD if k % 4 == 1 else SILVER)
        q.add(
            c + sphere * rad,
            sphere,
            col,
            BALL,
            T_BALLS[0] + rng.random() * (T_BALLS[1] - T_BALLS[0]),
            centre=c,
            pop=True,
        )

    # fairy lights on the branch tips
    tips_arr = np.array(tips, dtype=np.float32)
    pick = rng.choice(len(tips_arr), min(170, len(tips_arr)), replace=False)
    bulb = fib_sphere(16)
    light_t0 = T_LIGHTS[0] + rng.random(len(pick)) * (T_LIGHTS[1] - T_LIGHTS[0])
    for c, t0 in zip(tips_arr[pick], light_t0):
        col = WARM if rng.random() < 0.8 else np.array([1.0, 0.95, 0.85])
        q.add(
            c + bulb * 0.03,
            bulb,
            col,
            LIGHT,
            float(t0),
            centre=c,
            phase=rng.random() * TAU,
        )

    # presents: boxes with a crossed ribbon and a bow around the base
    gifts = [
        (0.3, 0.96, 0.25, 0.20, 0.24, 0.4, RED, GOLD),
        (1.4, 1.02, 0.23, 0.23, 0.22, 0.9, BLUE_BOX, SILVER),
        (2.5, 0.95, 0.28, 0.16, 0.22, 1.0, GOLD, RED),
        (3.6, 1.00, 0.22, 0.22, 0.22, 0.3, RED, WHITE),
        (4.6, 0.95, 0.25, 0.19, 0.25, 0.1, GREEN_BOX, RED),
        (5.5, 1.02, 0.22, 0.20, 0.21, 0.7, BLUE_BOX, GOLD),
    ]
    loop = fib_sphere(64)
    for g_idx, (a, dist, w, h, d, rot, body, rib) in enumerate(gifts):
        areas = np.array([w * d, h * d, h * d, h * w, h * w])
        m = int(2600 * areas.sum())
        face = rng.choice(5, m, p=areas / areas.sum())
        fa, fb = rng.random(m) - 0.5, rng.random(m) - 0.5
        lx = np.select([face == 0, face == 1, face == 2], [fa * w, np.full(m, w / 2), np.full(m, -w / 2)], default=fa * w)
        ly = np.where(face == 0, h, (fb + 0.5) * h)
        lz = np.select([face == 0, face <= 2, face == 3], [fb * d, fa * d, np.full(m, d / 2)], default=np.full(m, -d / 2))
        n = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)[face]
        on_x, on_z = np.abs(lx) < 0.024, np.abs(lz) < 0.024
        ribbon = np.select([face == 0, face <= 2], [on_x | on_z, on_z], default=on_x)
        ca, sa = np.cos(rot), np.sin(rot)
        center = np.array([np.cos(a) * dist, GROUND, np.sin(a) * dist], dtype=np.float32)
        world_p = np.stack([lx * ca - lz * sa, ly, lx * sa + lz * ca], axis=1) + center
        world_n = np.stack([n[:, 0] * ca - n[:, 2] * sa, n[:, 1], n[:, 0] * sa + n[:, 2] * ca], axis=1)
        t_gift = T_GIFTS[0] + (g_idx / len(gifts)) * (T_GIFTS[1] - T_GIFTS[0])
        q.add(
            world_p,
            world_n,
            np.where(ribbon[:, None], rib, body),
            np.where(ribbon, RIBBON, GIFT),
            t_gift,
            centre=center + np.array([0.0, h * 0.5, 0.0], dtype=np.float32),
            pop=True,
        )
        # Crossed bow loops on top of each gift
        for ba in (0.0, np.pi * 0.5):
            bow_off = np.array([np.cos(ba + rot) * 0.035, h + 0.03, np.sin(ba + rot) * 0.035], dtype=np.float32)
            bow_pts = loop * np.array([0.045, 0.026, 0.024], dtype=np.float32)
            bow_r = np.stack(
                [bow_pts[:, 0] * np.cos(ba + rot) - bow_pts[:, 2] * np.sin(ba + rot), bow_pts[:, 1], bow_pts[:, 0] * np.sin(ba + rot) + bow_pts[:, 2] * np.cos(ba + rot)],
                axis=1,
            )
            q.add(bow_r + center + bow_off, loop, rib, RIBBON, t_gift + 0.05, centre=center + bow_off, pop=True)

    return q.compile()


def project(pts: np.ndarray, rot: float, cols: int, rows: int, aspect: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ca, sa = math.cos(rot), math.sin(rot)
    x = pts[:, 0] * ca - pts[:, 2] * sa
    y = pts[:, 1] - 0.12
    z = pts[:, 0] * sa + pts[:, 2] * ca
    # Slight downward camera pitch
    pitch = 0.15
    cp, sp = math.cos(pitch), math.sin(pitch)
    yp = y * cp - z * sp
    zp = y * sp + z * cp + 4.85
    scale = 2.65 / np.maximum(zp, 0.25)
    sx = cols * 0.5 + x * scale * (rows * 0.40) / aspect
    sy = rows * 0.44 - yp * scale * (rows * 0.40)
    return sx, sy, zp


class Background:
    """Precomputed starfield, falling snow, and receding ground dots."""

    def __init__(self, cols: int, rows: int) -> None:
        rng = np.random.default_rng(42)
        self.cols = cols
        self.rows = rows
        self.horizon = int(rows * 0.53)

        # Twinkling night stars above horizon
        n_stars = int(cols * rows * 0.038)
        self.star_r = rng.integers(0, self.horizon + 4, size=n_stars)
        self.star_c = rng.integers(0, cols, size=n_stars)
        self.star_ch = rng.choice([ord("+"), ord("*"), ord(".")], size=n_stars, p=[0.45, 0.30, 0.25])
        self.star_phase = rng.random(n_stars) * TAU
        self.star_speed = rng.uniform(1.2, 3.2, size=n_stars)

        # Falling snowflakes across full sky + ground
        n_snow = 115
        self.snow_x0 = rng.uniform(0.0, cols, size=n_snow)
        self.snow_y0 = rng.uniform(0.0, rows, size=n_snow)
        self.snow_vy = rng.uniform(2.2, 5.4, size=n_snow)
        self.snow_vx = rng.uniform(-0.8, 0.8, size=n_snow)
        self.snow_phase = rng.random(n_snow) * TAU
        self.snow_ch = rng.choice([ord("+"), ord("*"), ord(".")], size=n_snow, p=[0.35, 0.45, 0.20])

        # Perspective ground grid points in 3D world space (dense enough to fill rows cleanly)
        gx = np.linspace(-4.2, 4.2, 130)
        gz = np.linspace(-2.6, 5.4, 110)
        mx, mz = np.meshgrid(gx, gz)
        jitter_x = rng.uniform(-0.025, 0.025, mx.size)
        jitter_z = rng.uniform(-0.025, 0.025, mx.size)
        self.ground_pts = np.stack(
            [mx.ravel() + jitter_x, np.full(mx.size, GROUND), mz.ravel() + jitter_z], axis=1
        ).astype(np.float32)

    def paint(self, char: np.ndarray, col: np.ndarray, t: float, rot: float, aspect: float = CW / CH) -> None:
        # 1. Perspective ground plane
        gx, gy, gz = project(self.ground_pts, rot * 0.35, self.cols, self.rows, aspect)
        ix = np.rint(gx).astype(np.int32)
        iy = np.rint(gy).astype(np.int32)
        mask = (iy >= self.horizon) & (iy < self.rows) & (ix >= 0) & (ix < self.cols) & (gz > 0.6)
        ix, iy, gz = ix[mask], iy[mask], gz[mask]
        depth_fade = np.clip(1.30 - gz / 8.5, 0.16, 0.62)
        g_col = np.array([0.55, 0.62, 0.78], dtype=np.float32) * depth_fade[:, None]
        g_ch = np.where(depth_fade > 0.40, ord("-"), ord("."))
        char[iy, ix] = g_ch
        col[iy, ix] = g_col

        # 2. Night stars (avoid top-right timer HUD zone)
        tw = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(t * self.star_speed + self.star_phase))
        in_sky = (self.star_r < self.horizon) & ~((self.star_r < 4) & (self.star_c >= self.cols - 16))
        sr = self.star_r[in_sky]
        sc = self.star_c[in_sky]
        char[sr, sc] = self.star_ch[in_sky]
        col[sr, sc] = np.array([0.78, 0.85, 0.98], dtype=np.float32) * tw[in_sky, None]

        # 3. Falling snowflakes (keep HUD corners clean)
        sx = np.rint((self.snow_x0 + self.snow_vx * t + 0.8 * np.sin(t * 1.1 + self.snow_phase)) % self.cols).astype(np.int32)
        sy = np.rint((self.snow_y0 + self.snow_vy * t) % self.rows).astype(np.int32)
        sx = np.clip(sx, 0, self.cols - 1)
        sy = np.clip(sy, 0, self.rows - 1)
        hud_clear = ~((sy < 4) & (sx >= self.cols - 16))
        sx, sy = sx[hud_clear], sy[hud_clear]
        fade = 0.60 + 0.40 * np.sin(t * 2.5 + self.snow_phase[hud_clear])
        char[sy, sx] = self.snow_ch[hud_clear]
        col[sy, sx] = np.array([0.86, 0.92, 1.00], dtype=np.float32) * fade[:, None]


def star(
    char: np.ndarray,
    col: np.ndarray,
    cx: int,
    cy: int,
    size: float,
    t: float,
    power: float,
    aspect: float,
    flash: float = 0.0,
) -> None:
    # thin rays first, then a filled five-pointed star on top;
    # when it lights up it bursts white with long rays, then settles to gold
    tint = GOLD + (1.0 - GOLD) * flash
    power = power * (1.0 + 1.6 * flash)
    for k in range(8):
        ang = k * TAU / 8.0 + 0.10 * math.sin(t * 0.7)
        if math.sin(ang) < -0.2:  # no rays down into the tree
            continue
        horiz = 0.55 if abs(math.sin(ang)) < 0.25 else 1.0
        reach = size * (2.6 + 0.6 * math.sin(t * 3.0 + k * 1.3)) * (1.0 if k % 2 == 0 else 0.75) * horiz * (1.0 + 1.6 * flash)
        ch = ord("-|/\\"[int(round((math.degrees(ang) % 180.0) / 45.0)) % 4])
        for s in np.arange(size * 1.15, reach, 0.5):
            y = int(round(cy - math.sin(ang) * s))
            x = int(round(cx + math.cos(ang) * s / aspect))
            if 0 <= y < char.shape[0] and 0 <= x < char.shape[1] and col[y, x].max() < 0.75:
                fade = max(0.0, 1.0 - s / reach) * horiz
                char[y, x] = ch
                col[y, x] = tint * (power * 0.65 * fade)

    # Filled 5-pointed star polygon
    verts: list[tuple[float, float]] = []
    for i in range(10):
        a = -math.pi / 2.0 + i * TAU / 10.0
        r = size if i % 2 == 0 else size * 0.42
        verts.append((cx + math.cos(a) * r / aspect, cy + math.sin(a) * r))

    y0 = max(0, int(math.floor(cy - size - 1)))
    y1 = min(char.shape[0], int(math.ceil(cy + size + 2)))
    x0 = max(0, int(math.floor(cx - size / aspect - 1)))
    x1 = min(char.shape[1], int(math.ceil(cx + size / aspect + 2)))
    if y1 <= y0 or x1 <= x0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = point_in_poly(xx.astype(np.float32), yy.astype(np.float32), verts)
    dist = np.hypot((xx - cx) * aspect, yy - cy) / max(size, 0.5)
    core = inside & (dist < 0.52)
    mid = inside & ~core
    char[y0:y1, x0:x1][mid] = ord("#")
    char[y0:y1, x0:x1][core] = ord("@")
    col[y0:y1, x0:x1][inside] = tint * min(1.35, power)
    # Crisp outer tips
    for vx, vy in verts[::2]:
        ix, iy = int(round(vx)), int(round(vy))
        if 0 <= iy < char.shape[0] and 0 <= ix < char.shape[1]:
            char[iy, ix] = ord("*")
            col[iy, ix] = tint * min(1.35, power)


def put(char: np.ndarray, col: np.ndarray, r: int, c: int, ch: str, tint: np.ndarray, lum: float) -> None:
    if 0 <= r < char.shape[0] and 0 <= c < char.shape[1]:
        char[r, c] = ord(ch)
        col[r, c] = np.clip(tint * lum, 0.0, 1.0)


def step_frame(
    scene: SceneData,
    bg: Background,
    t: float,
    cols: int = COLS,
    rows: int = ROWS,
    aspect: float = CW / CH,
    message: str = DEFAULT_MESSAGE,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    char = np.zeros((rows, cols), dtype=np.uint8)
    col = np.zeros((rows, cols, 3), dtype=np.float32)

    rot = t * ROT_SPEED
    bg.paint(char, col, t, rot, aspect)

    active = t >= scene.t0
    pos = scene.pos[active]
    nrm = scene.nrm[active]
    p_col = scene.col[active]
    kind = scene.kind[active]
    t0 = scene.t0[active]
    base = scene.base[active]
    centre = scene.centre[active]
    phase = scene.phase[active]
    pop = scene.pop[active]

    age = t - t0
    if np.any(pop):
        s_pop = smooth(age[pop] / 0.35)[:, None]
        pos = pos.copy()
        pos[pop] = centre[pop] + (pos[pop] - centre[pop]) * s_pop

    sx, sy, zp = project(pos, rot, cols, rows, aspect)
    ix = np.rint(sx).astype(np.int32)
    iy = np.rint(sy).astype(np.int32)
    in_bounds = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows) & (zp > 0.5)

    ix, iy, zp = ix[in_bounds], iy[in_bounds], zp[in_bounds]
    nrm = nrm[in_bounds]
    p_col = p_col[in_bounds]
    kind = kind[in_bounds]
    age = age[in_bounds]
    base = base[in_bounds]
    phase = phase[in_bounds]

    # Rotate normals into view space for lighting
    ca, sa = math.cos(rot), math.sin(rot)
    nx = nrm[:, 0] * ca - nrm[:, 2] * sa
    ny = nrm[:, 1]
    nz = nrm[:, 0] * sa + nrm[:, 2] * ca
    light_dir = np.array([0.42, 0.56, -0.71], dtype=np.float32)
    ndotl = np.clip(nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2], 0.0, 1.0)

    lum = base * (0.28 + 0.72 * ndotl)
    lum = np.where(kind == CONE, base * (0.18 + 0.25 * ndotl), lum)
    lum = np.where(kind == SNOW, 0.72 + 0.28 * ndotl, lum)
    lum = np.where(kind == TINSEL, 0.65 + 0.35 * np.sin(t * 4.5 + phase) * 0.5 + 0.35 * ndotl, lum)
    lum = np.where(kind == BALL, 0.38 + 0.62 * (ndotl ** 1.3), lum)
    lum = np.where(kind == GIFT, 0.42 + 0.58 * ndotl, lum)
    lum = np.where(kind == RIBBON, 0.62 + 0.38 * ndotl, lum)

    # Twinkling fairy lights
    is_light = kind == LIGHT
    twinkle = 0.55 + 0.45 * np.sin(t * 6.0 + phase)
    lum = np.where(is_light, 0.55 + 0.65 * twinkle, lum)

    # Growth frontier frost-white flash as points appear
    grow_mask = (kind == NEEDLE) | (kind == SNOW) | (kind == TRUNK)
    flash_w = np.clip(1.0 - age / 0.38, 0.0, 1.0) * grow_mask
    rgb = p_col * (1.0 - 0.85 * flash_w[:, None]) + WHITE * (0.85 * flash_w[:, None])
    lum = np.maximum(lum, 1.05 * flash_w)

    # Z-buffer resolve (frontmost point wins per cell)
    flat_idx = iy * cols + ix
    order = np.argsort(zp, kind="mergesort")
    _, unique_pos = np.unique(flat_idx[order], return_index=True)
    winners = order[unique_pos]

    wy, wx = iy[winners], ix[winners]
    w_lum = np.clip(lum[winners], 0.0, 1.0)
    w_kind = kind[winners]
    w_rgb = np.clip(rgb[winners] * np.clip(lum[winners], 0.12, 1.2)[:, None], 0.0, 1.0)

    ramp_idx = np.clip((w_lum * (len(RAMP_ARR) - 1)).astype(np.int32), 1, len(RAMP_ARR) - 1)
    ch_vals = RAMP_ARR[ramp_idx]
    ch_vals = np.where(w_kind == LIGHT, ord("@"), ch_vals)
    ch_vals = np.where(w_kind == GIFT, np.where(w_lum > 0.65, ord("#"), ord("%")), ch_vals)
    ch_vals = np.where(w_kind == RIBBON, ord("@"), ch_vals)

    char[wy, wx] = ch_vals
    col[wy, wx] = w_rgb

    # Top star: dim silhouette before T_STAR, then bursts white and settles gold
    tx, ty, _ = project(np.array([[0.0, Y1 + 0.13, 0.0]], dtype=np.float32), rot, cols, rows, aspect)
    if t >= T_GROW[1] - 0.5:
        on = float(smooth((t - T_STAR) / 0.3)) if t >= T_STAR else 0.0
        flash = float(np.clip(1.0 - (t - T_STAR) / 1.0, 0.0, 1.0) ** 2) if t >= T_STAR else 0.0
        power = 0.25 + 0.85 * on
        star_sz = (1.6 + 1.5 * on) * (1.0 + 0.05 * math.sin(t * 3.0))
        star(char, col, int(round(float(tx[0]))), int(round(float(ty[0]))), star_sz, t, power, aspect, flash=flash)

    # Typed holiday message near the bottom
    if t > T_MSG and message:
        row = rows - 9
        x0 = max(0, (cols - len(message)) // 2)
        shown = min(len(message), int((t - T_MSG) * 10.0))
        char[row - 1 : row + 2, max(x0 - 3, 0) : min(cols, x0 + shown + 3)] = 0
        for i, ch in enumerate(message[:shown]):
            if ch != " ":
                tint = (RED, GOLD, GREEN_BOX)[(i // 2 + int(t * 1.5)) % 3]
                put(char, col, row, x0 + i, ch, tint, 0.85 + 0.25 * math.sin(t * 5.0 + i))

    lights_on = int((scene.kind[active] == LIGHT).sum() // 16)
    info = {
        "rotation_deg": (math.degrees(rot)) % 360.0,
        "lights": lights_on,
        "letters": int((char > 0).sum()),
    }
    return char, col, info


class Renderer:
    """Fast glyph-atlas ASCII renderer with multi-scale bloom and HUD."""

    def __init__(
        self,
        cols: int = COLS,
        rows: int = ROWS,
        cw: int = CW,
        ch: int = CH,
        font_px: int = FONT_PX,
        hud_px: int = 15,
        author: str = DEFAULT_AUTHOR,
        show_hint: bool = True,
    ) -> None:
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.w, self.h = cols * cw, rows * ch
        self.author = author
        self.show_hint = show_hint

        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont(MONO, font_px)
        self.hud_font = pygame.font.SysFont(MONO, hud_px)

        atlas = []
        for code in range(32, 127):
            s = pygame.Surface((cw, ch))
            s.fill((0, 0, 0))
            g = font.render(chr(code), True, (255, 255, 255))
            s.blit(g, ((cw - g.get_width()) // 2, (ch - g.get_height()) // 2))
            gray = pygame.surfarray.array3d(s).transpose(1, 0, 2)[:, :, :1].astype(np.float32) / 255.0
            atlas.append(gray)
        self.atlas = np.stack(atlas, axis=0)  # (95, ch, cw, 1)

    def render_surface(self, char: np.ndarray, col: np.ndarray, t: float, info: dict[str, float | int]) -> pygame.Surface:
        # Sparse glyph composite: only touch non-empty cells (~40% of grid)
        canvas = np.zeros((self.rows, self.cols, self.ch, self.cw, 3), dtype=np.uint8)
        rr, cc = np.nonzero(char >= 32)
        if len(rr):
            idx = np.clip(char[rr, cc] - 32, 0, 94)
            glyphs = self.atlas[idx]  # (k, ch, cw, 1)
            rgb_vals = (col[rr, cc][:, None, None, :] * 255.0)
            canvas[rr, cc] = np.clip(glyphs * rgb_vals, 0.0, 255.0).astype(np.uint8)
        img = canvas.transpose(0, 2, 1, 3, 4).reshape(self.h, self.w, 3)

        # Fast multi-scale additive bloom in uint8
        if cv2 is not None:
            bright = cv2.subtract(img, (35, 35, 35, 0))
            s1 = cv2.resize(bright, (max(1, self.w // 2), max(1, self.h // 2)), interpolation=cv2.INTER_AREA)
            s1 = cv2.GaussianBlur(s1, (0, 0), sigmaX=2.4, sigmaY=2.4)
            g1 = cv2.resize(s1, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            s2 = cv2.resize(bright, (max(1, self.w // 6), max(1, self.h // 6)), interpolation=cv2.INTER_AREA)
            s2 = cv2.GaussianBlur(s2, (0, 0), sigmaX=2.6, sigmaY=2.6)
            g2 = cv2.resize(s2, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            out_u8 = cv2.add(img, np.array(BG_RGB, dtype=np.uint8))
            out_u8 = cv2.addWeighted(out_u8, 1.0, g1, 0.50, 0.0)
            out_u8 = cv2.addWeighted(out_u8, 1.0, g2, 0.45, 0.0)
        else:
            out = img.astype(np.float32) + np.array(BG_RGB, dtype=np.float32)
            surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
            small1 = pygame.transform.smoothscale(surf, (max(1, self.w // 4), max(1, self.h // 4)))
            blur1 = pygame.transform.smoothscale(small1, (self.w, self.h))
            small2 = pygame.transform.smoothscale(surf, (max(1, self.w // 12), max(1, self.h // 12)))
            blur2 = pygame.transform.smoothscale(small2, (self.w, self.h))
            b1 = pygame.surfarray.array3d(blur1).transpose(1, 0, 2).astype(np.float32)
            b2 = pygame.surfarray.array3d(blur2).transpose(1, 0, 2).astype(np.float32)
            out_u8 = np.clip(out + 0.45 * b1 + 0.55 * b2, 0, 255).astype(np.uint8)

        surf = pygame.surfarray.make_surface(out_u8.transpose(1, 0, 2))
        self._draw_hud(surf, t, info)
        return surf

    def render_rgb(self, char: np.ndarray, col: np.ndarray, t: float, info: dict[str, float | int]) -> np.ndarray:
        surf = self.render_surface(char, col, t, info)
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def _draw_hud(self, surf: pygame.Surface, t: float, info: dict[str, float | int]) -> None:
        pad_x = max(16, int(self.cw * 2.0))
        pad_y = max(14, int(self.ch * 1.1))

        # Top-right timer badge
        t_lbl = self.hud_font.render(f"t {t:05.2f}s", True, ACCENT)
        tx = self.w - pad_x - t_lbl.get_width()
        ty = pad_y
        dot_r = max(3, self.hud_font.get_height() // 5)
        dot_dx = max(13, int(self.hud_font.get_height() * 0.85))
        pygame.draw.circle(surf, ACCENT, (tx - dot_dx, ty + t_lbl.get_height() // 2 + 1), dot_r)
        surf.blit(t_lbl, (tx, ty))

        # Bottom-left signature & telemetry
        line_h = int(self.hud_font.get_height() * 1.32)
        by = self.h - pad_y - line_h * 3
        s1 = self.hud_font.render(self.author, True, SIGN)
        s2 = self.hud_font.render(f"rotation {info['rotation_deg']:06.2f}°     lights {info['lights']:>3d}", True, DIM)
        s3 = self.hud_font.render(f"letters {info['letters']:,}", True, DIM)
        surf.blit(s1, (pad_x, by))
        surf.blit(s2, (pad_x, by + line_h))
        surf.blit(s3, (pad_x, by + line_h * 2))

        if self.show_hint:
            h_lbl = self.hud_font.render(HINT, True, (68, 76, 92))
            surf.blit(h_lbl, (self.w - pad_x - h_lbl.get_width(), self.h - pad_y - h_lbl.get_height()))


# ---------------------------------------------------------------------------
# Procedural Festive Soundtrack Synthesizer
# ---------------------------------------------------------------------------

def _bell_voice(freq: float, dur: float, amp: float = 0.25) -> np.ndarray:
    n = max(1, int(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    atk = np.minimum(t / 0.006, 1.0)
    env1 = atk * np.exp(-t * 3.1)
    env2 = atk * np.exp(-t * 6.5)
    env3 = atk * np.exp(-t * 11.0)
    sig = (
        0.65 * np.sin(2.0 * np.pi * freq * t) * env1
        + 0.24 * np.sin(2.0 * np.pi * freq * 2.0 * t) * env2
        + 0.11 * np.sin(2.0 * np.pi * freq * 3.0 * t) * env3
        + 0.05 * np.sin(2.0 * np.pi * freq * 4.02 * t) * env3
    )
    return sig * amp


def _synth_lead_voice(freq: float, dur: float, amp: float = 0.24) -> np.ndarray:
    """Warm 80s synth-bell vocal lead with sustained expression."""
    tail = 0.22
    total_dur = dur + tail
    n = max(1, int(total_dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    atk = np.minimum(t / 0.010, 1.0)
    hold = np.where(t <= dur, 1.0, np.exp(-(t - dur) / 0.075))
    decay = np.exp(-t * (0.95 / max(dur, 0.35)))
    env = atk * hold * decay

    vib = 0.003 * np.clip((t - 0.18) / 0.25, 0.0, 1.0) * np.sin(2.0 * np.pi * 5.4 * t)
    phase = 2.0 * np.pi * freq * (t + np.cumsum(vib) / SAMPLE_RATE)

    chime = np.exp(-t * 11.0)
    sig = (
        0.58 * np.sin(phase)
        + 0.24 * np.sin(2.0 * phase) * (0.55 + 0.45 * np.exp(-t * 4.0))
        + 0.10 * np.sin(3.0 * phase) * np.exp(-t * 6.0)
        + 0.08 * np.sin(4.0 * phase) * chime
    )
    return (sig * env * amp).astype(np.float32)


def _synth_bass_voice(freq: float, dur: float = 0.23, amp: float = 0.21) -> np.ndarray:
    """Punchy staccato 80s Juno/Prophet synth bass."""
    n = max(1, int(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    atk = np.minimum(t / 0.004, 1.0)
    env = atk * np.exp(-t * 11.5)
    bright_env = atk * np.exp(-t * 22.0)
    sig = (
        0.66 * np.sin(2.0 * np.pi * freq * t) * env
        + 0.26 * np.sin(2.0 * np.pi * freq * 2.0 * t) * bright_env
        + 0.11 * np.sin(2.0 * np.pi * freq * 3.0 * t) * bright_env
    )
    return (sig * amp).astype(np.float32)


def _pad_chord(freqs: list[float], dur: float, amp: float = 0.08) -> np.ndarray:
    n = max(1, int(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    fade_in = np.minimum(t / 0.45, 1.0)
    fade_out = np.minimum((dur - t) / 0.55, 1.0)
    env = np.clip(fade_in * fade_out, 0.0, 1.0)
    sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        sig += np.sin(2.0 * np.pi * f * t) + 0.28 * np.sin(2.0 * np.pi * f * 2.0 * t)
        sig += 0.15 * np.sin(2.0 * np.pi * (f * 1.003) * t)
    return (sig / max(1, len(freqs))) * env * amp


def _juno_pad_chord(freqs: list[float], dur: float, amp: float = 0.11) -> np.ndarray:
    """Stereo chorus-detuned 80s synth pad."""
    n = max(1, int(dur * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    fade_in = np.minimum(t / 0.28, 1.0)
    fade_out = np.minimum((dur - t) / 0.35, 1.0)
    env = np.clip(fade_in * fade_out, 0.0, 1.0)
    stereo = np.zeros((n, 2), dtype=np.float32)
    for f in freqs:
        for ch, detune in ((0, 0.9972), (1, 1.0028)):
            f_d = f * detune
            voice = np.zeros(n, dtype=np.float32)
            for k, wt in ((1, 1.0), (2, 0.38), (3, 0.18), (4, 0.09), (5, 0.04)):
                voice += wt * np.sin(2.0 * np.pi * (f_d * k) * t)
            stereo[:, ch] += voice
    stereo /= max(1, len(freqs))
    stereo *= env[:, None] * amp
    return stereo


def _pop_kick(amp: float = 0.22) -> np.ndarray:
    dur = 0.28
    t = np.arange(int(dur * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    freq = 44.0 + 78.0 * np.exp(-t * 32.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    env = np.minimum(t / 0.002, 1.0) * np.exp(-t * 11.0)
    return (np.sin(phase) * env * amp).astype(np.float32)


def _snare_80s(rng: np.random.Generator, amp: float = 0.13) -> np.ndarray:
    dur = 0.22
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = rng.normal(0.0, 1.0, n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    bp = np.where((freqs >= 900.0) & (freqs <= 7200.0), 1.0, 0.12)
    filt = np.fft.irfft(np.fft.rfft(noise) * bp, n=n).astype(np.float32)
    body = np.sin(2.0 * np.pi * (185.0 * np.exp(-t * 15.0) + 125.0) * t) * np.exp(-t * 24.0)
    gated = np.where(t < 0.12, np.exp(-t * 7.5), np.exp(-0.12 * 7.5) * np.exp(-(t - 0.12) * 32.0))
    return ((0.75 * filt * gated + 0.35 * body) * amp).astype(np.float32)


def _add_stereo(buf: np.ndarray, start_s: float, mono: np.ndarray, pan: float = 0.0) -> None:
    i0 = int(start_s * SAMPLE_RATE)
    if i0 >= buf.shape[0] or i0 + len(mono) <= 0:
        return
    src_start = max(0, -i0)
    dst_start = max(0, i0)
    length = min(len(mono) - src_start, buf.shape[0] - dst_start)
    if length <= 0:
        return
    l_gain = np.sqrt(0.5 * (1.0 - np.clip(pan, -1.0, 1.0)))
    r_gain = np.sqrt(0.5 * (1.0 + np.clip(pan, -1.0, 1.0)))
    chunk = mono[src_start : src_start + length]
    buf[dst_start : dst_start + length, 0] += chunk * l_gain
    buf[dst_start : dst_start + length, 1] += chunk * r_gain


def _add_stereo_pair(buf: np.ndarray, start_s: float, stereo: np.ndarray) -> None:
    i0 = int(start_s * SAMPLE_RATE)
    if i0 >= buf.shape[0] or i0 + stereo.shape[0] <= 0:
        return
    src_start = max(0, -i0)
    dst_start = max(0, i0)
    length = min(stereo.shape[0] - src_start, buf.shape[0] - dst_start)
    if length <= 0:
        return
    buf[dst_start : dst_start + length] += stereo[src_start : src_start + length]


def _apply_reverb(stereo: np.ndarray) -> np.ndarray:
    out = stereo.copy()
    taps = [
        (int(0.037 * SAMPLE_RATE), 0.25, 0.14),
        (int(0.061 * SAMPLE_RATE), 0.16, 0.23),
        (int(0.097 * SAMPLE_RATE), 0.17, 0.11),
        (int(0.149 * SAMPLE_RATE), 0.09, 0.15),
    ]
    n = stereo.shape[0]
    for delay, g_same, g_cross in taps:
        if delay >= n:
            continue
        out[delay:, 0] += stereo[: n - delay, 0] * g_same + stereo[: n - delay, 1] * g_cross
        out[delay:, 1] += stereo[: n - delay, 1] * g_same + stereo[: n - delay, 0] * g_cross
    return out


def _render_last_christmas(buf: np.ndarray, duration: float, star_time: float) -> None:
    """80s synth-pop + chime arrangement in D major at 108 BPM (D -> Bm -> Em -> A)."""
    beat = 60.0 / 108.0  # 0.5556s per beat; 36 beats = 20.00s
    bar = 4.0 * beat
    rng = np.random.default_rng(1984)

    chords = [
        (0.0, bar + 0.15, ["D3", "F#3", "A3", "C#4"]),
        (1 * bar, 2 * bar + 0.15, ["D3", "F#3", "A3", "C#4"]),
        (3 * bar, 2 * bar + 0.15, ["B2", "D3", "F#3", "A3"]),
        (5 * bar, 2 * bar + 0.15, ["E3", "G3", "B3", "D4"]),
        (7 * bar, 1.5 * bar + 0.1, ["A2", "D3", "E3", "G3"]),
        (8.5 * bar, 0.25 * bar + 0.1, ["A2", "C#3", "E3", "G3"]),
        (8.75 * bar, max(0.5, duration - 8.75 * bar), ["D3", "F#3", "A3", "E4"]),
    ]
    for t0, dur_c, notes in chords:
        if t0 >= duration:
            break
        pad = _juno_pad_chord([NOTE_MAP[n] for n in notes], min(dur_c, duration - t0 + 0.2), amp=0.095)
        _add_stereo_pair(buf, t0, pad)

    bass_roots = [
        (0, 12, "D2", "C#2"),
        (12, 20, "B1", "F#2"),
        (20, 28, "E2", "G2"),
        (28, 34, "A1", "A1"),
    ]
    for b_start, b_end, root, pickup in bass_roots:
        total_eighths = (b_end - b_start) * 2
        for e in range(total_eighths):
            t_e = (b_start + e * 0.5) * beat
            if t_e >= duration - 0.1:
                break
            note = pickup if (e == total_eighths - 1 and b_end < 34) else root
            _add_stereo(buf, t_e, _synth_bass_voice(NOTE_MAP[note], dur=0.21, amp=0.17), pan=0.0)

    kick = _pop_kick(amp=0.19)
    snare = _snare_80s(rng, amp=0.075)
    t_sleigh = np.arange(int(0.11 * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    sleigh = rng.normal(0.0, 1.0, len(t_sleigh)).astype(np.float32) * np.sin(2.0 * np.pi * 6400.0 * t_sleigh) * np.exp(-t_sleigh * 32.0) * 0.032

    total_beats = int(duration / beat)
    for b in range(total_beats):
        tb = b * beat
        if b < 35:
            if b % 2 == 0:
                _add_stereo(buf, tb, kick, pan=0.0)
            else:
                _add_stereo(buf, tb, snare, pan=0.04)
            _add_stereo(buf, tb, sleigh, pan=0.25)
            if tb + 0.5 * beat < duration - 0.15:
                _add_stereo(buf, tb + 0.5 * beat, sleigh * 0.65, pan=-0.25)

    for i, note in enumerate(["D5", "F#5", "A5", "C#6", "A5", "F#5"]):
        ti = 0.15 + i * 0.5 * beat
        if ti < duration - 0.2:
            _add_stereo(buf, ti, _bell_voice(NOTE_MAP[note], 0.85, amp=0.15), pan=-0.35 + 0.14 * i)

    melody = [
        (4.00, "E5", 1.35), (5.50, "E5", 0.45), (6.00, "D5", 0.85), (7.00, "A4", 0.80),
        (8.00, "E5", 0.45), (8.50, "E5", 0.45), (9.00, "F#5", 0.80), (10.00, "D5", 1.80),
        (12.00, "B4", 0.45), (12.50, "B4", 0.45), (13.00, "E5", 0.45), (13.50, "E5", 0.45),
        (14.00, "F#5", 0.48), (14.50, "D5", 1.30), (16.00, "B4", 0.45), (16.50, "C#5", 0.45),
        (17.00, "D5", 0.45), (17.50, "C#5", 0.45), (18.00, "B4", 1.80),
        (20.00, "F#5", 0.90), (21.00, "E5", 1.25), (22.50, "B4", 0.45), (23.00, "F#5", 0.48),
        (23.50, "G5", 0.48), (24.00, "F#5", 0.48), (24.50, "E5", 2.50),
        (28.00, "D5", 0.45), (28.50, "C#5", 0.45), (29.00, "D5", 0.45), (29.50, "C#5", 0.45),
        (30.00, "C#5", 0.48), (30.50, "D5", 0.48), (31.00, "C#5", 0.85), (32.00, "A4", 2.00),
        (34.50, "D5", 1.45),
    ]
    for idx, (b_pos, note, b_dur) in enumerate(melody):
        t_note = b_pos * beat
        if t_note >= duration - 0.15:
            continue
        dur_s = min(b_dur * beat, max(0.15, duration - t_note))
        lead = _synth_lead_voice(NOTE_MAP[note], dur_s, amp=0.28)
        bell_layer = _bell_voice(NOTE_MAP[note], max(0.50, dur_s), amp=0.13)
        pan = -0.14 if idx % 2 == 0 else 0.14
        _add_stereo(buf, t_note, lead, pan=pan)
        _add_stereo(buf, t_note, bell_layer, pan=-pan)

    if star_time < duration - 0.2:
        sparkle = ["D5", "F#5", "A5", "C#6", "D6", "E6", "F#6", "A6"]
        for i, note in enumerate(sparkle):
            ti = star_time + i * 0.052
            if ti < duration - 0.15:
                _add_stereo(
                    buf,
                    ti,
                    _bell_voice(NOTE_MAP[note], 1.5, amp=0.13),
                    pan=-0.6 + 1.2 * (i / max(1, len(sparkle) - 1)),
                )


def _render_traditional(buf: np.ndarray, duration: float, star_time: float) -> None:
    """Traditional 'We Wish You a Merry Christmas' music-box & chime arrangement."""
    chords = [
        (0.0, 2.6, ["C3", "G3", "C4", "E4"]),
        (2.5, 2.6, ["F3", "C4", "F4", "A4"]),
        (5.0, 2.6, ["D3", "A3", "D4", "F#4"]),
        (7.5, 2.6, ["G3", "B3", "D4", "G4"]),
        (10.0, 2.0, ["E3", "B3", "E4", "G4"]),
        (11.8, 2.8, ["F3", "C4", "F4", "A4", "C5"]),
        (14.5, 2.8, ["G3", "D4", "G4", "B4"]),
        (17.2, 2.8, ["C3", "G3", "C4", "E4", "G4"]),
    ]
    for t0, dur, notes in chords:
        pad = _pad_chord([NOTE_MAP[n] for n in notes], dur, amp=0.10)
        _add_stereo(buf, t0, pad, pan=0.0)

    beat = 0.44
    melody = [
        (0.4, "D4", 1.0), (0.4 + beat, "G4", 1.0), (0.4 + 2 * beat, "G4", 0.5), (0.4 + 2.5 * beat, "A4", 0.5),
        (0.4 + 3 * beat, "G4", 0.5), (0.4 + 3.5 * beat, "F#4", 0.5), (0.4 + 4 * beat, "E4", 1.0), (0.4 + 5 * beat, "E4", 1.0),
        (0.4 + 6 * beat, "E4", 1.0), (0.4 + 7 * beat, "A4", 1.0), (0.4 + 8 * beat, "A4", 0.5), (0.4 + 8.5 * beat, "B4", 0.5),
        (0.4 + 9 * beat, "A4", 0.5), (0.4 + 9.5 * beat, "G4", 0.5), (0.4 + 10 * beat, "F#4", 1.0), (0.4 + 11 * beat, "D4", 1.0),
        (0.4 + 12 * beat, "D4", 1.0), (0.4 + 13 * beat, "B4", 1.0), (0.4 + 14 * beat, "B4", 0.5), (0.4 + 14.5 * beat, "C5", 0.5),
        (0.4 + 15 * beat, "B4", 0.5), (0.4 + 15.5 * beat, "A4", 0.5), (0.4 + 16 * beat, "G4", 1.0), (0.4 + 17 * beat, "E4", 1.0),
        (0.4 + 18 * beat, "D4", 0.5), (0.4 + 18.5 * beat, "D4", 0.5), (0.4 + 19 * beat, "E4", 1.0),
        (0.4 + 20 * beat, "A4", 1.0), (0.4 + 21 * beat, "F#4", 1.0), (0.4 + 22 * beat, "G4", 2.0),
    ]
    for idx, (t0, note, b_len) in enumerate(melody):
        voice = _bell_voice(NOTE_MAP[note], max(0.8, b_len * 0.9), amp=0.22)
        pan = -0.25 if idx % 2 == 0 else 0.25
        _add_stereo(buf, t0, voice, pan=pan)

    sparkle_notes = ["C5", "E5", "G5", "B5", "C6", "D6", "E6", "G6"]
    for i, note in enumerate(sparkle_notes):
        voice = _bell_voice(NOTE_MAP[note], 1.6, amp=0.16)
        _add_stereo(buf, star_time + i * 0.055, voice, pan=-0.6 + 1.2 * (i / max(1, len(sparkle_notes) - 1)))

    t_p2 = 12.6
    phrase2 = [
        (0.0, "G4", 1.0), (1 * beat, "C5", 1.0), (2 * beat, "C5", 1.0), (3 * beat, "C5", 1.0),
        (4 * beat, "B4", 2.0), (6 * beat, "B4", 1.0), (7 * beat, "C5", 1.0),
        (8 * beat, "B4", 1.0), (9 * beat, "A4", 1.0), (10 * beat, "G4", 2.0),
        (12 * beat, "D5", 1.0), (13 * beat, "E5", 1.0), (14 * beat, "D5", 1.0), (15 * beat, "C5", 1.0),
        (16 * beat, "G5", 1.2), (17 * beat, "E5", 1.5), (18.5 * beat, "C5", 2.5),
    ]
    for idx, (dt_s, note, b_len) in enumerate(phrase2):
        t_note = t_p2 + dt_s
        if t_note < duration - 0.5:
            voice = _bell_voice(NOTE_MAP[note], max(0.9, b_len * 0.95), amp=0.23)
            _add_stereo(buf, t_note, voice, pan=0.25 if idx % 2 == 0 else -0.25)

    rng = np.random.default_rng(12)
    t_tick = np.arange(int(0.18 * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    noise = rng.normal(0.0, 1.0, len(t_tick)).astype(np.float32)
    shimmer = noise * np.sin(2.0 * np.pi * 5800.0 * t_tick) * np.exp(-t_tick * 24.0) * 0.035
    for beat_i in range(1, int(duration / (2 * beat))):
        _add_stereo(buf, beat_i * (2 * beat), shimmer, pan=0.35 * np.sin(beat_i))


def generate_festive_audio(
    out_wav: str | Path,
    duration: float = 20.0,
    star_time: float = T_STAR,
    theme: str = "last_christmas",
) -> Path:
    """Synthesize a 20s stereo holiday soundtrack synced to star_time."""
    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = max(1, int(duration * SAMPLE_RATE))
    buf = np.zeros((total_samples, 2), dtype=np.float32)

    if theme == "traditional":
        _render_traditional(buf, duration=duration, star_time=star_time)
    elif theme == "last_christmas":
        _render_last_christmas(buf, duration=duration, star_time=star_time)
    else:
        raise ValueError(f"Unknown audio theme: {theme!r}")

    wet = _apply_reverb(buf)
    fade_samples = min(total_samples, int(0.85 * SAMPLE_RATE))
    if fade_samples > 0:
        wet[-fade_samples:, :] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]

    peak = float(np.max(np.abs(wet)))
    if peak > 0.0:
        wet = wet / max(peak, 1.0) * 0.88
    pcm = (np.tanh(wet) * 32767.0).astype(np.int16)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return out_path


def _prepare_playback_wav(
    tmp_dir: Path,
    duration: float,
    audio_path: Path | None,
    audio_start: float,
    audio_theme: str,
) -> Path:
    """Prepare a seekable stereo 44.1 kHz WAV for interactive Pygame playback."""
    out_wav = tmp_dir / "playback.wav"
    if audio_path is None:
        return generate_festive_audio(out_wav, duration=duration, star_time=T_STAR, theme=audio_theme)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() == ".wav" and audio_start <= 0.0:
        return audio_path

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y"]
    if audio_start > 0.0:
        cmd.extend(["-ss", f"{audio_start:.3f}"])
    cmd.extend([
        "-i", str(audio_path),
        "-t", f"{duration:.3f}",
        "-vn",
        "-ac", "2",
        "-ar", str(SAMPLE_RATE),
        "-af", f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0.0, duration - 1.0):.2f}:d=1.0",
        str(out_wav),
    ])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed:\n{res.stderr}")
    return out_wav


# ---------------------------------------------------------------------------
# Vertical 9:16 Split-Screen Reel Exporter
# ---------------------------------------------------------------------------

def _tokenize_lines(source: str) -> list[list[tuple[int, int, tuple[int, int, int]]]]:
    """Return per-line (start_col, end_col, rgb) spans for syntax highlighting."""
    lines = source.splitlines()
    per_line: list[list[tuple[int, int, tuple[int, int, int]]]] = [[] for _ in lines]
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return per_line

    for idx, tok in enumerate(tokens):
        ttype, tstr, (srow, scol), (erow, ecol), _ = tok
        if ttype in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        if ttype == tokenize.COMMENT:
            col = SYNTAX_COLORS["comment"]
        elif ttype == tokenize.STRING:
            col = SYNTAX_COLORS["string"]
        elif ttype == tokenize.NUMBER:
            col = SYNTAX_COLORS["number"]
        elif ttype == tokenize.OP:
            col = SYNTAX_COLORS["op"]
        elif ttype == tokenize.NAME:
            if tstr in PY_KEYWORDS:
                col = SYNTAX_COLORS["keyword"]
            elif tstr in PY_BUILTINS:
                col = SYNTAX_COLORS["builtin"]
            elif idx + 1 < len(tokens) and tokens[idx + 1].string == "(":
                col = SYNTAX_COLORS["call"]
            else:
                col = SYNTAX_COLORS["ident"]
        else:
            col = SYNTAX_COLORS["ident"]

        for r in range(srow, erow + 1):
            if not (1 <= r <= len(lines)):
                continue
            c0 = scol if r == srow else 0
            c1 = ecol if r == erow else len(lines[r - 1])
            if c1 > c0:
                per_line[r - 1].append((c0, c1, col))
    return per_line


def _prerender_code_sheet(
    source_path: Path,
    width: int,
    font_px: int,
    line_h: int,
) -> tuple[np.ndarray, list[str]]:
    """Pre-render the entire source file onto an RGB sheet for fast scrolling."""
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    spans_by_line = _tokenize_lines(source)

    font = pygame.font.SysFont(MONO, font_px)
    total_h = max(1, len(lines) * line_h)
    sheet = pygame.Surface((width, total_h))
    sheet.fill(CODE_BG)

    num_x_right = int(width * 0.058)
    code_x0 = int(width * 0.082)

    for i, text in enumerate(lines):
        y = i * line_h + (line_h - font.get_height()) // 2
        num_surf = font.render(str(i + 1), True, LINE_NUM_COL)
        sheet.blit(num_surf, (num_x_right - num_surf.get_width(), y))

        spans = spans_by_line[i]
        if not spans:
            if text:
                txt_surf = font.render(text, True, SYNTAX_COLORS["ident"])
                sheet.blit(txt_surf, (code_x0, y))
            continue

        cursor = 0
        for c0, c1, col in spans:
            if c0 > cursor:
                gap = text[cursor:c0]
                if gap.strip():
                    gs = font.render(gap, True, SYNTAX_COLORS["ident"])
                    sheet.blit(gs, (code_x0 + font.size(text[:cursor])[0], y))
            seg = text[c0:c1]
            if seg:
                ss = font.render(seg, True, col)
                sheet.blit(ss, (code_x0 + font.size(text[:c0])[0], y))
            cursor = max(cursor, c1)

    rgb = pygame.surfarray.array3d(sheet).transpose(1, 0, 2).copy()
    return rgb, lines


def _find_line(lines: list[str], needle: str, fallback: int) -> int:
    for idx, line in enumerate(lines):
        if needle in line:
            return idx
    return min(fallback, max(0, len(lines) - 1))


def _build_scroll_keyframes(lines: list[str], duration: float) -> list[tuple[float, float]]:
    anchors = [
        (0.0, "# needles:"),
        (3.8, "# dark inner cone"),
        (5.1, "# trunk"),
        (6.4, "# tinsel garland:"),
        (7.5, "# baubles:"),
        (8.6, "# fairy lights"),
        (10.0, "# presents:"),
        (11.8, "def star("),
        (13.2, "if t > T_MSG"),
        (16.2, "class Renderer:"),
        (duration, "def render_surface("),
    ]
    keyframes: list[tuple[float, float]] = []
    prev_idx = 0
    for t, key in anchors:
        idx = _find_line(lines, key, prev_idx)
        idx = max(prev_idx, idx)
        keyframes.append((t, float(idx)))
        prev_idx = idx
    return keyframes


def _interp_line(keyframes: list[tuple[float, float]], t: float) -> float:
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    for (t0, l0), (t1, l1) in zip(keyframes, keyframes[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / max(1e-6, t1 - t0)
            s = u * u * (3.0 - 2.0 * u)
            return l0 + (l1 - l0) * s
    return keyframes[-1][1]


def _render_header(width: int, height: int, fps: int, filename: str = "christmas_tree.py") -> np.ndarray:
    surf = pygame.Surface((width, height))
    surf.fill(HEADER_BG)
    pygame.draw.line(surf, BORDER_COL, (0, 0), (width, 0), 1)
    pygame.draw.line(surf, BORDER_COL, (0, height - 1), (width, height - 1), 1)

    font_px = max(13, int(height * 0.48))
    font = pygame.font.SysFont(MONO, font_px)
    pad_x = max(16, int(width * 0.022))

    left_lbl = font.render(filename, True, HEADER_TEXT_COL)
    right_lbl = font.render(f"running · {fps} fps", True, ACCENT)
    cy = (height - left_lbl.get_height()) // 2
    surf.blit(left_lbl, (pad_x, cy))

    rx = width - pad_x - right_lbl.get_width()
    dot_r = max(3, height // 10)
    dot_dx = max(11, int(font_px * 0.78))
    pygame.draw.circle(surf, ACCENT, (rx - dot_dx, height // 2), dot_r)
    surf.blit(right_lbl, (rx, (height - right_lbl.get_height()) // 2))
    return pygame.surfarray.array3d(surf).transpose(1, 0, 2).copy()


def render_video(
    output_path: Path,
    res: str = "1080p",
    fps: int = 30,
    duration: float = DURATION,
    author: str = DEFAULT_AUTHOR,
    message: str = DEFAULT_MESSAGE,
    audio_path: Path | None = None,
    no_audio: bool = False,
    audio_theme: str = "last_christmas",
    audio_start: float = 0.0,
) -> Path:
    """Export a 9:16 split-screen Reel MP4 combining ASCII animation, scrolling code, and audio."""
    if audio_path is not None and not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    prev_driver = os.environ.get("SDL_VIDEODRIVER")
    init_here = not pygame.get_init()
    if init_here:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()

    if res == "720p":
        vid_w, vid_h = 720, 1280
        art_h, hdr_h = 864, 30
        cw, ch, font_px, hud_px = 8, 11, 13, 14
        code_font_px, code_line_h = 13, 21
    else:
        vid_w, vid_h = 1080, 1920
        art_h, hdr_h = 1296, 44
        cw, ch, font_px, hud_px = 12, 17, 19, 21
        code_font_px, code_line_h = 20, 32

    code_h = vid_h - art_h - hdr_h
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene = build()
    bg = Background(COLS, ROWS)
    renderer = Renderer(
        cols=COLS,
        rows=ROWS,
        cw=cw,
        ch=ch,
        font_px=font_px,
        hud_px=hud_px,
        author=author,
        show_hint=False,
    )
    art_ox = (vid_w - renderer.w) // 2
    art_oy = (art_h - renderer.h) // 2

    src_path = Path(__file__)
    header_img = _render_header(vid_w, hdr_h, fps, filename=src_path.name)
    code_sheet, code_lines = _prerender_code_sheet(
        src_path, width=vid_w, font_px=code_font_px, line_h=code_line_h
    )
    keyframes = _build_scroll_keyframes(code_lines, duration)

    fade_rows = max(12, code_line_h)
    code_fade = np.ones((code_h, 1, 1), dtype=np.float32)
    code_fade[:fade_rows, 0, 0] = np.linspace(0.15, 1.0, fade_rows, dtype=np.float32)
    code_fade[-fade_rows:, 0, 0] = np.linspace(1.0, 0.25, fade_rows, dtype=np.float32)
    code_bg_arr = np.array(CODE_BG, dtype=np.float32)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_dir = Path(tempfile.mkdtemp(prefix="cfs_reel_"))
    silent_mp4 = tmp_dir / "video_only.mp4"

    cmd_enc = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{vid_w}x{vid_h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "medium",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        str(silent_mp4),
    ]

    total_frames = int(round(duration * fps))
    frame_buf = np.empty((vid_h, vid_w, 3), dtype=np.uint8)
    frame_buf[:art_h] = BG_RGB
    frame_buf[art_h : art_h + hdr_h] = header_img

    t_start = time.perf_counter()
    proc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    max_scroll = max(0, code_sheet.shape[0] - code_h)
    active_offset_px = int(code_h * 0.24)

    try:
        for f_idx in range(total_frames):
            t = f_idx / fps
            char, col, info = step_frame(scene, bg, t, COLS, ROWS, cw / ch, message=message)
            art_rgb = renderer.render_rgb(char, col, t, info)
            frame_buf[art_oy : art_oy + renderer.h, art_ox : art_ox + renderer.w] = art_rgb

            target_line = _interp_line(keyframes, t)
            target_y = target_line * code_line_h
            scroll_y = int(round(np.clip(target_y - active_offset_px, 0.0, float(max_scroll))))
            pane = code_sheet[scroll_y : scroll_y + code_h].copy()

            hl_y0 = int(round(target_y)) - scroll_y
            hl_y1 = hl_y0 + code_line_h
            if 0 <= hl_y0 < code_h and hl_y1 > 0:
                y_a, y_b = max(0, hl_y0), min(code_h, hl_y1)
                band = pane[y_a:y_b].astype(np.int16)
                is_bg = np.all(band <= 15, axis=2)
                band[is_bg] = HIGHLIGHT_BG
                pane[y_a:y_b] = np.clip(band, 0, 255).astype(np.uint8)

            pane_f = pane.astype(np.float32) * code_fade + code_bg_arr * (1.0 - code_fade)
            frame_buf[art_h + hdr_h :] = np.clip(pane_f, 0, 255).astype(np.uint8)

            proc.stdin.write(frame_buf.tobytes())

            if (f_idx + 1) % max(1, fps * 4) == 0 or f_idx + 1 == total_frames:
                elapsed = time.perf_counter() - t_start
                print(f"Encoded {f_idx + 1}/{total_frames} frames ({(f_idx + 1)/elapsed:.1f} fps)")

        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        stderr = ""
        if proc.stderr:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            proc.stderr.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg video encode failed (exit {rc}):\n{stderr}")

        if no_audio:
            shutil.move(str(silent_mp4), str(output_path))
        else:
            if audio_path is None:
                audio_file = generate_festive_audio(
                    tmp_dir / "soundtrack.wav",
                    duration=duration,
                    star_time=T_STAR,
                    theme=audio_theme,
                )
            else:
                audio_file = audio_path

            audio_input_args = (
                ["-ss", f"{max(0.0, audio_start):.3f}", "-i", str(audio_file)]
                if (audio_path is not None and audio_start > 0.0)
                else ["-i", str(audio_file)]
            )
            cmd_mux = [
                ffmpeg_exe,
                "-y",
                "-i", str(silent_mp4),
                *audio_input_args,
                "-t", f"{duration:.3f}",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-af", f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0.0, duration - 1.0):.2f}:d=1.0",
                "-movflags", "+faststart",
                str(output_path),
            ]
            mux_res = subprocess.run(cmd_mux, capture_output=True, text=True)
            if mux_res.returncode != 0:
                raise RuntimeError(f"ffmpeg audio mux failed:\n{mux_res.stderr}")
    finally:
        if init_here:
            pygame.quit()
            if prev_driver is None:
                os.environ.pop("SDL_VIDEODRIVER", None)
            else:
                os.environ["SDL_VIDEODRIVER"] = prev_driver
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_path


# ---------------------------------------------------------------------------
# Unified CLI Entrypoint (Interactive Window or MP4 Reel Export)
# ---------------------------------------------------------------------------

def _run_interactive(args: argparse.Namespace) -> None:
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    scene = build()
    bg = Background(COLS, ROWS)
    renderer = Renderer(author=args.author, show_hint=True)

    tmp_audio_dir: Path | None = None
    music_ready = False
    if not args.no_audio and os.environ.get("SDL_AUDIODRIVER") != "dummy":
        try:
            pygame.mixer.init(frequency=SAMPLE_RATE, channels=2)
            tmp_audio_dir = Path(tempfile.mkdtemp(prefix="cfs_play_"))
            wav_file = _prepare_playback_wav(
                tmp_audio_dir,
                duration=DURATION,
                audio_path=args.audio,
                audio_start=args.audio_start,
                audio_theme=args.theme,
            )
            pygame.mixer.music.load(str(wav_file))
            pygame.mixer.music.play()
            music_ready = True
        except Exception as exc:
            print(f"Audio playback unavailable ({exc}); continuing visually.")

    def _sync_music(pos_s: float, is_paused: bool) -> None:
        if not music_ready:
            return
        try:
            pygame.mixer.music.play(start=max(0.0, min(pos_s, DURATION - 0.1)))
            if is_paused:
                pygame.mixer.music.pause()
        except pygame.error:
            pass

    t = 0.0
    paused = False
    running = True
    shot_idx = 1

    try:
        while running:
            dt = clock.tick(args.fps) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        paused = not paused
                        if music_ready:
                            if paused:
                                pygame.mixer.music.pause()
                            else:
                                pygame.mixer.music.unpause()
                    elif ev.key == pygame.K_r:
                        t = 0.0
                        _sync_music(t, paused)
                    elif ev.key == pygame.K_LEFT:
                        t = max(0.0, t - 1.0)
                        _sync_music(t, paused)
                    elif ev.key == pygame.K_RIGHT:
                        t = min(DURATION, t + 1.0)
                        _sync_music(t, paused)
                    elif ev.key == pygame.K_s:
                        os.makedirs("output", exist_ok=True)
                        out_path = os.path.join("output", f"screenshot_{shot_idx:02d}.png")
                        pygame.image.save(screen, out_path)
                        print(f"Saved {out_path}")
                        shot_idx += 1

            if not paused:
                prev_t = t
                t += dt
                if t > DURATION:
                    if args.no_loop:
                        t = DURATION
                    else:
                        t = t % DURATION
                        if prev_t <= DURATION:
                            _sync_music(t, False)

            char, col, info = step_frame(scene, bg, t, COLS, ROWS, CW / CH, message=args.message)
            frame_surf = renderer.render_surface(char, col, t, info)
            screen.blit(frame_surf, (0, 0))
            pygame.display.flip()
    finally:
        if music_ready:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except pygame.error:
                pass
        pygame.quit()
        if tmp_audio_dir is not None:
            shutil.rmtree(tmp_audio_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-in-one 3D ASCII Christmas Tree, Soundtrack Synthesizer & 9:16 Reel Exporter"
    )
    parser.add_argument("--author", default=DEFAULT_AUTHOR, help="Bottom-left handle/signature")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Holiday greeting text")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--no-loop", action="store_true", help="Hold at final frame instead of looping")
    parser.add_argument(
        "--export",
        nargs="?",
        const="output/christmas_tree_reel.mp4",
        default=None,
        metavar="MP4",
        help="Export 9:16 split-screen MP4 Reel (default path: output/christmas_tree_reel.mp4)",
    )
    parser.add_argument("--res", choices=["1080p", "720p"], default="1080p", help="Reel resolution (--export)")
    parser.add_argument("--duration", type=float, default=DURATION, help="Reel duration in seconds (--export)")
    parser.add_argument("--audio", type=Path, default=None, help="Custom audio/video track (.mp3, .m4a, .wav, .mp4)")
    parser.add_argument("--audio-start", type=float, default=0.0, help="Start offset in seconds for --audio")
    parser.add_argument(
        "--theme",
        choices=["last_christmas", "traditional"],
        default="last_christmas",
        help="Built-in soundtrack theme when --audio is not provided",
    )
    parser.add_argument("--no-audio", action="store_true", help="Disable audio playback or export silent MP4")
    args = parser.parse_args()

    if args.export is not None:
        out = render_video(
            output_path=Path(args.export),
            res=args.res,
            fps=args.fps,
            duration=args.duration,
            author=args.author,
            message=args.message,
            audio_path=args.audio,
            no_audio=args.no_audio,
            audio_theme=args.theme,
            audio_start=args.audio_start,
        )
        print(f"Done! Saved Reel video to {out.resolve()}")
    else:
        _run_interactive(args)


if __name__ == "__main__":
    main()
