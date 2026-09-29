"""Deterministic sharding for multi-node Slurm runs."""

from __future__ import annotations

import hashlib


def config_compile_shard(name: str, num_shards: int) -> int:
    """Shard index for compile partitioning (by config name)."""
    if num_shards <= 1:
        return 0
    digest = hashlib.sha256(name.encode()).hexdigest()
    return int(digest, 16) % num_shards


def config_on_compile_shard(name: str, shard_index: int, num_shards: int) -> bool:
    """True if this shard compiles the config (hash of its name)."""
    return config_compile_shard(name, num_shards) == shard_index


def eval_pair_shard(name: str, M: int, N: int, K: int, num_shards: int) -> int:
    """Return shard index in [0, num_shards) for one (config, shape) eval pair."""
    if num_shards <= 1:
        return 0
    key = f"{name}:{M}:{N}:{K}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest, 16) % num_shards


def pair_on_shard(name: str, M: int, N: int, K: int, shard_index: int, num_shards: int) -> bool:
    """True if this shard benchmarks the (config, M, N, K) pair."""
    return eval_pair_shard(name, M, N, K, num_shards) == shard_index
