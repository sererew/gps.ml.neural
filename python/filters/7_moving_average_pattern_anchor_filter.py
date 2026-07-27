#!/usr/bin/env python3
"""Moving-average local filter followed by pattern-anchor correction."""

from classic_pattern_anchor_filter import load_filter_module, run_classic_pattern_anchor


def main():
    moving_average_filter = load_filter_module("7_moving_average_filter.py", "moving_average_filter_module")

    def apply_filter(track_df, args):
        return moving_average_filter.apply_moving_average_filter(track_df, window_size=args.window)

    run_classic_pattern_anchor(
        filter_name="moving_average",
        apply_filter=apply_filter,
        default_suffix="moving_average_pattern_anchor_filtered",
        filter_args={
            "--window": {
                "type": int,
                "default": 5,
                "help": "Window size for the moving average",
            }
        },
    )


if __name__ == "__main__":
    main()
