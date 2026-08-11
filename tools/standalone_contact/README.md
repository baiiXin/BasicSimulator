# Standalone Contact

### 建议代码结构

```
standalone_contact/
│
├── mesh_topology.py
│      triangles -> Newton-compatible edge_indices
│      vertex/edge adjacency
│      n-ring topological filter
│
├── geometry.py
│      triangle_closest_point
│      binary_search
│
├── collision_kernels.py
│      compute_tri_aabbs
│      compute_edge_aabbs
│      VT detection kernel
│      EE detection kernel
│
├── collision_detector.py
│      TriMeshCollisionInfo
│      TriMeshCollisionDetector
│      refit()
│      detect_vertex_triangle()
│      detect_edge_edge()
│
├── dat_kernels.py
│      division plane
│      planar_truncation_t
│      apply_planar_truncation_parallel_by_collision
│      apply_truncation_ts
│
├── penetration_free.py
│      collision_detection()
│      penetration_free_truncation()
│
└── tests/
       test_collision_vs_newton.py
       test_dat_vs_newton.py
       benchmark_collision.py
```

### 求解器使用：

```
detector = TriMeshCollisionDetector(
    vertices=q_rest,
    triangles=faces,
    ...
)

detector.detect(
    q=current_q,
    margin=margin,
    rest_q=q_rest,
    rest_exclusion_radius=rest_exclusion_radius,
)

dq_safe = penetration_free_truncation(
    q_snapshot=q_collision,
    dq=dq,
    collision_info=detector.collision_info,
    triangles=faces,
    edges=detector.edges,
)
```

### 代码处理，建议尽量复制

```
wp.Bvh
same AABB construction
same VT kernel
same EE kernel
same block_dim = 16
same fixed collision buffers
same 4 threads / collision primitive in DAT
same atomic_min
same topology filters
```


# 版权
### newton

项目地址: https://github.com/newton-physics/newton
版权声明: 

```
SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
SPDX-License-Identifier: Apache-2.0
```

### Standalone Contact
建议我们统一用类似: 

```
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from newton-physics/newton.
# This file has been modified to remove dependencies on the Newton framework
# and provide a standalone Warp/NumPy implementation.
#
# Modifications Copyright (c) 2026 <your name / project>
```

项目再保留 Apache-2.0 license, 并加一个 THIRD_PARTY_NOTICES.md 记录: 

```
Newton Physics
Source: newton-physics/newton
Baseline commit: 71fe5e63889e1368e186882dfb000623b4c9a434

Adapted components:
- tri_mesh_collision.py
- geometry/kernels.py collision utilities
- particle_vbd_kernels.py planar DAT utilities
```

# 环境

```
# 1. 新建环境
conda create -n bslo python=3.11 -y

# 2. 激活
conda activate bslo

# 3. 更新基础安装工具
python -m pip install -U pip setuptools wheel

# 4. 安装你本地这份 Newton，以及它要求的 Warp 等依赖
python -m pip install -e ./newton

# 5. 安装测试工具
python -m pip install pytest
```