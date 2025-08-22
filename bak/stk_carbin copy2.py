#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stock position sizing from indicator (2000macd) with exponential-like curve.

Requirements from user story (中文):
- 指标 2000macd 范围 [-100, 100]
- 当指标为 0 -> 持仓 100%; 指标为 100 -> 持仓 0%
- 关系不是线性的，期望指数变化：
  * 接近 100 时，持仓快速下降到 0%
  * 接近 0 时，持仓快速上升到 100%
- 提供一个参数来调节变化速度

Implementation:
Two modes selectable via --mode:
1) symmetric (default): S-curve (logistic) mapping，满足“中间段变化快、两端变化慢”。
	支持 --midband a b（默认 0.25 0.75）把陡峭区集中在 [a,b]；
	令 x = 0.5 + (t - center)/width，其中 center=(a+b)/2，width=(b-a)，
	base(x)=1/(1+exp(k*(x-0.5)))，并用 base(x) 在 t=0 与 t=1 处做归一化，确保端点 0→1、1→0。
2) exp: the original one-sided exponential decay (steeper near 100).

CLI:
- Run this file to generate a PNG chart showing curves for one or multiple k values.
  Example:
	python stk_carbin.py --k 0.5 1 2 5 --out position_curve.png
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Iterable, List

try:
	import numpy as np
except Exception:  # numpy may be missing; we'll fallback to pure python arrays
	np = None  # type: ignore


def _pos_symmetric(t: float, k: float, band: tuple[float, float] | None = None) -> float:
	"""Symmetric S-curve (logistic) on t ∈ [0,1] -> pos ∈ [0,1].

	Fast change in the middle (around 0.5) and slow near the ends (0 and 1).
	"""
	if k <= 1e-8:
		# near-zero steepness, use linear fallback (can switch to smoothstep if desired)
		return 1.0 - t

	def base(x: float) -> float:
		# Logistic centered at 0.5; larger k => steeper center, slower ends
		return 1.0 / (1.0 + math.exp(k * (x - 0.5)))

	# Optional band scaling to focus slope between [a,b]
	if band is None:
		band = (0.25, 0.75)
	a, b = band
	# Accept 0..100 inputs too
	if a > 1.0 or b > 1.0:
		a, b = a / 100.0, b / 100.0
	# Clamp and ensure order
	a = max(0.0, min(1.0, a))
	b = max(0.0, min(1.0, b))
	if a > b:
		a, b = b, a
	width = max(b - a, 1e-6)
	center = 0.5 * (a + b)

	def scale(u: float) -> float:
		# Affine map so that [a,b] maps to a width-1 window around 0.5
		return 0.5 + (u - center) / width

	lo = base(scale(0.0))
	hi = base(scale(1.0))
	den = lo - hi
	if abs(den) < 1e-12:
		return 1.0 - t
	val = base(scale(t))
	# Normalize so t=0 -> 1, t=1 -> 0
	return (val - hi) / den


def _pos_exp(t: float, k: float) -> float:
	"""One-sided exponential decay on t ∈ [0,1] -> pos ∈ [0,1]."""
	if k <= 1e-8:
		return 1.0 - t
	denom = 1.0 - math.exp(-k)
	if abs(denom) < 1e-12:
		return 1.0 - t
	return (1.0 - math.exp(-k * (1.0 - t))) / denom


def position_from_indicator(
	indicator: float,
	k: float = 2.0,
	mode: str = "symmetric",
	midband: tuple[float, float] | None = None,
) -> float:
	"""Map indicator in [-100,100] to position percent in [0,100] with exponential-like curve.

	Args:
		indicator: 2000macd value in [-100, 100]. Values will be clamped into range.
		k: positive steepness parameter (>0). Larger k => faster changes near ends.
		mode: 'symmetric' (default) or 'exp'.

	Returns:
		Position percent in [0,100].
	"""
	# Clamp external indicator to [-100, 100] and map to t ∈ [0,1]
	val = max(-100.0, min(100.0, float(indicator)))
	# map: -100 -> 0.0, 0 -> 0.5, 100 -> 1.0
	t = (val + 100.0) / 200.0
	k = float(k)
	mode = (mode or "symmetric").lower()

	if mode == "exp":
		pos_frac = _pos_exp(t, k)
	else:
		pos_frac = _pos_symmetric(t, k, band=midband)
	# Convert to percent and clamp
	return max(0.0, min(100.0, pos_frac * 100.0))


def vectorized_positions(
	indicators: Iterable[float],
	k: float,
	mode: str = "symmetric",
	midband: tuple[float, float] | None = None,
) -> List[float]:
	"""Vectorized helper over an iterable of indicators."""
	return [position_from_indicator(x, k, mode=mode, midband=midband) for x in indicators]


def generate_chart(out_path: str, ks: List[float], mode: str = "symmetric", midband: tuple[float, float] | None = None) -> str:
	"""Generate a PNG chart of position vs indicator for given k values.

	Args:
		out_path: path to save the image (PNG recommended).
		ks: list of positive k values to plot.

	Returns:
		The absolute path to the saved image.
	"""
	# Import matplotlib lazily to allow running core logic without it
	import matplotlib
	# Use a non-interactive backend for headless environments
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	# X axis: indicator -100..100
	if np is not None:
		xs = np.linspace(-100, 100, 501)
	else:
		# step 0.4 to get 501 points across 200 range
		xs = [-100.0 + i * 0.4 for i in range(501)]

	# Square figure for equal visual space on both axes
	plt.figure(figsize=(6, 6))
	for k in ks:
		ys = vectorized_positions(xs, k, mode=mode, midband=midband)
		if mode == "symmetric" and midband is not None:
			label = f"{mode}, k={k:g}, band={midband[0]:.2f}-{midband[1]:.2f}"
		else:
			label = f"{mode}, k={k:g}"
		if np is not None:
			plt.plot(xs, ys, label=label)
		else:
			plt.plot(list(xs), ys, label=label)

	plt.title("Position vs 2000macd (Exponential-like mapping)")
	plt.xlabel("2000macd (-100 → 100)")
	plt.ylabel("Position % (100 → 0)")
	# Fix axis limits and enforce equal scaling so 1 unit in x equals 1 unit in y
	plt.xlim(-100, 100)
	plt.ylim(0, 100)
	ax = plt.gca()
	ax.set_aspect('equal', adjustable='box')
	# Optional nicer ticks
	ax.set_xticks([-100, -50, 0, 50, 100])
	ax.set_yticks([0, 20, 40, 60, 80, 100])
	plt.grid(True, alpha=0.3)
	plt.legend(title="Steepness", loc="upper right")
	plt.tight_layout()

	out_dir = os.path.dirname(out_path) or os.getcwd()
	os.makedirs(out_dir, exist_ok=True)
	abs_path = os.path.abspath(out_path)
	plt.savefig(abs_path, dpi=160)
	plt.close()
	return abs_path


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Compute position from indicator (-100..100) and draw chart.")
	p.add_argument(
		"--k",
		"-k",
		type=float,
		nargs="+",
		default=[2.0, 5.0, 10.0],
		help="Steepness parameter(s) > 0. Larger => faster drop near 100. Default: 2 5 10",
	)
	p.add_argument(
		"--mode",
		choices=["symmetric", "exp"],
		default="symmetric",
		help="Mapping mode: 'symmetric' (steep at both ends) or 'exp' (steeper near 100).",
	)
	p.add_argument(
		"--midband",
		nargs=2,
		type=float,
		default=[0.25, 0.75],
		help="For symmetric mode: the band [a b] (0..1 or 0..100) where changes are fastest; default 0.25 0.75.",
	)
	p.add_argument(
		"--out",
		"-o",
		type=str,
		default="position_curve.png",
		help="Output image file path (PNG recommended). Default: position_curve.png",
	)
	p.add_argument(
		"--sample",
		type=float,
		default=None,
		help="If provided, print position for a single indicator value (-100..100) and exit.",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()

	# If --sample is provided, just print the computed position with k=first value
	if args.sample is not None:
		k = float(args.k[0])
		band = (args.midband[0], args.midband[1]) if args.mode == "symmetric" else None
		pos = position_from_indicator(args.sample, k=k, mode=args.mode, midband=band)
		if band:
			print(f"indicator={args.sample:g}, mode={args.mode}, k={k:g}, band={band} -> position={pos:.4f}%")
		else:
			print(f"indicator={args.sample:g}, mode={args.mode}, k={k:g} -> position={pos:.4f}%")
		return

	# Generate chart
	ks = [float(x) for x in args.k if float(x) > 0]
	if not ks:
		raise SystemExit("No valid k values provided. Use --k with positive numbers.")
	band = (args.midband[0], args.midband[1]) if args.mode == "symmetric" else None
	out = generate_chart(args.out, ks, mode=args.mode, midband=band)
	print(f"Chart saved to: {out}")


if __name__ == "__main__":
	main()


