#!/usr/bin/env python3
"""Gaussian local filter followed by pattern-anchor correction."""

from classic_pattern_anchor_filter import load_filter_module, run_classic_pattern_anchor


def main():
    gaussian_filter = load_filter_module("7_gaussian_filter.py", "gaussian_filter_module")

    def apply_filter(track_df, args):
        return gaussian_filter.apply_gaussian_filter(track_df, sigma=args.sigma)

    run_classic_pattern_anchor(
        filter_name="gaussian",
        apply_filter=apply_filter,
        default_suffix="gaussian_pattern_anchor_filtered",
        filter_args={
            "--sigma": {
                "type": float,
                "default": 1.5,
                "help": "Standard deviation for the Gaussian kernel",
            }
        },
    )


if __name__ == "__main__":
    main()
