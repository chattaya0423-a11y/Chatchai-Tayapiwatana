"""
CLI entry point for CytoSoftware-RPQ.

Example:
  python -m rpq.cli --video input.mp4 --out out_dir

You can override selected parameters via flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import RPQConfig, analyze_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CytoSoftware-RPQ: red particle quantification on MP4 videos.")
    p.add_argument("--video", required=True, help="Path to input MP4 video.")
    p.add_argument("--out", default="rpq_out", help="Output directory.")
    p.add_argument("--every-n", type=int, default=None, help="Process every N frames.")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames to analyze (after subsampling).")

    # Key detection knobs (common tuning parameters)
    p.add_argument("--bg-method", choices=["gaussian", "rolling_ball"], default=None)
    p.add_argument("--bg-sigma", type=float, default=None)
    p.add_argument("--rb-radius", type=int, default=None)

    p.add_argument("--red-dom", type=float, default=None)
    p.add_argument("--red-min", type=float, default=None)
    p.add_argument("--snr-min", type=float, default=None)

    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--num-sigmas", type=int, default=None)
    p.add_argument("--log-thr", type=float, default=None)
    p.add_argument("--robust-k", type=float, default=None)

    p.add_argument("--area-min", type=int, default=None)
    p.add_argument("--area-max", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = RPQConfig()
    if args.every_n is not None:
        cfg.process_every_n_frames = args.every_n
    if args.max_frames is not None:
        cfg.max_frames_to_analyze = args.max_frames

    if args.bg_method is not None:
        cfg.background_method = args.bg_method
    if args.bg_sigma is not None:
        cfg.bg_blur_sigma = args.bg_sigma
    if args.rb_radius is not None:
        cfg.rolling_ball_radius = args.rb_radius

    if args.red_dom is not None:
        cfg.red_dom_ratio = args.red_dom
    if args.red_min is not None:
        cfg.red_min_abs = args.red_min
    if args.snr_min is not None:
        cfg.snr_min = args.snr_min

    if args.sigma_min is not None:
        cfg.log_sigma_min = args.sigma_min
    if args.sigma_max is not None:
        cfg.log_sigma_max = args.sigma_max
    if args.num_sigmas is not None:
        cfg.log_num_sigmas = args.num_sigmas
    if args.log_thr is not None:
        cfg.log_thresh = args.log_thr
    if args.robust_k is not None:
        cfg.robust_k = args.robust_k

    if args.area_min is not None:
        cfg.min_particle_area = args.area_min
    if args.area_max is not None:
        cfg.max_particle_area = args.area_max

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = analyze_video(args.video, out_dir, cfg)
    print("Done.")
    print("Outputs:")
    print(" -", res["csv_path"])
    print(" -", res["blobs_csv_path"])
    print(" -", res["annotated_mp4_path"])
    print(" -", res["run_config_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
