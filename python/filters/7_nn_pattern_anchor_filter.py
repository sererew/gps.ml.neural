#!/usr/bin/env python3
"""
Experimental v3 + pattern-anchor slow correction filter.

This wrapper keeps the regular `nn` filter as the frozen v3 baseline and exposes
an oracle-style experimental filter for pipeline step 7. It infers the clean
pattern from the input path and samples sparse anchors from that pattern.
"""

import importlib.util
from pathlib import Path


def load_nn_filter_module():
    """Load 7_nn_filter.py even though its filename is not importable normally."""
    module_path = Path(__file__).with_name("7_nn_filter.py")
    spec = importlib.util.spec_from_file_location("nn_filter_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Run the base NN filter with pattern-anchor correction enabled by default."""
    nn_filter = load_nn_filter_module()
    nn_filter.main(
        default_slow_correction="pattern-anchor",
        default_anchor_trim_to_pattern=True,
        default_anchor_edge_skip_points=180,
    )


if __name__ == "__main__":
    main()
