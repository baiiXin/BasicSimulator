"""Standalone Newton-style triangle-mesh self-contact detection.

Runtime dependencies: NumPy + NVIDIA Warp. No Newton import is required.
"""

from .collision_detector import (
    CollisionBufferCapacityError,
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
)
from .mesh_topology import MeshTopology, build_mesh_topology

__all__ = [
    "CollisionBufferCapacityError",
    "MeshTopology",
    "TriMeshCollisionDetector",
    "TriMeshCollisionInfo",
    "build_mesh_topology",
]
