"""Resolve feature metadata from a run manifest (unfused vs fusion)."""

from __future__ import annotations


def category_levels_for(manifest: dict) -> dict:
    """Return frozen categorical level order for the feature set in ``manifest``."""
    from features import CATEGORY_LEVELS as base

    cats = manifest.get("categorical_features", [])
    if "fusion_kind" in cats:
        from features_fusion import CATEGORY_LEVELS as fusion

        return fusion
    return base
