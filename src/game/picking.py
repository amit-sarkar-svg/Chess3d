"""
src/game/picking.py

Utilities for click-to-select tile + simple piece picking.

Functions:
- screen_point_to_world_ray(x, y, width, height, projection, view)
- intersect_ray_plane(origin, dir, plane_y=0.0)
- world_pos_to_tile(hit_pos, tile_size=1.0)
- ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius)
- pick_at_screen(x, y, width, height, projection, view, tile_size, pieces, piece_radius)

`pieces` is expected to be an iterable of:
    ((file, rank), (piece_type, color))

Example usage:
    result = pick_at_screen(x, y, win.width, win.height, proj, view, 1.0, pieces_dict.items(), 0.35)
    # result -> {"tile": (f,r) or None, "piece": ((f,r, type, color), dist) or None, "world_pos": np.array(...) }
"""

from typing import Tuple, Optional, Iterable, Any, Dict
import numpy as np


def screen_point_to_world_ray(x: float, y: float, width: int, height: int,
                              projection: np.ndarray, view: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert screen coordinates (x,y) to a world-space ray (origin, direction).
    - x,y are in window coordinates (origin top-left, as provided by GLFW).
    - projection and view are 4x4 numpy arrays.

    Returns:
        ray_origin (3,), ray_dir (3,) normalized
    """
    # Normalized Device Coordinates (NDC)
    ndc_x = (2.0 * x) / float(width) - 1.0
    ndc_y = 1.0 - (2.0 * y) / float(height)   # invert Y because window origin is top-left

    ndc_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
    ndc_far = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64)

    inv_vp = np.linalg.inv(np.matmul(projection, view))

    # Unproject
    world_near = inv_vp @ ndc_near
    world_far = inv_vp @ ndc_far

    # divide by w
    if abs(world_near[3]) > 1e-6:
        world_near = world_near / world_near[3]
    if abs(world_far[3]) > 1e-6:
        world_far = world_far / world_far[3]

    ray_origin = world_near[:3]
    ray_dir = world_far[:3] - world_near[:3]
    norm = np.linalg.norm(ray_dir)
    if norm < 1e-9:
        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        ray_dir = ray_dir / norm

    return ray_origin.astype(np.float64), ray_dir.astype(np.float64)


def intersect_ray_plane(ray_origin: np.ndarray, ray_dir: np.ndarray, plane_y: float = 0.0) -> Optional[np.ndarray]:
    """
    Intersect ray with horizontal plane y = plane_y.
    Returns world-space hit point (3,) or None if parallel / behind.
    """
    oy = float(ray_origin[1])
    dy = float(ray_dir[1])

    if abs(dy) < 1e-9:
        return None

    t = (plane_y - oy) / dy
    if t < 0:
        return None

    hit = ray_origin + ray_dir * t
    return hit


def world_pos_to_tile(hit_pos: np.ndarray, tile_size: float = 1.0) -> Optional[Tuple[int, int]]:
    """
    Convert a world-space 3D position (x,z) on plane to board tile (file, rank).
    Board layout assumptions (same as BoardRenderer/PieceRenderer):
        tile centers are at:
            x = (file - 3.5) * tile_size
            z = (rank - 3.5) * tile_size
        tile extents go from x = -4*tile_size ... +4*tile_size (edges)

    Returns (file, rank) in 0..7 or None if outside board extents.
    """
    x = float(hit_pos[0])
    z = float(hit_pos[2])

    # Convert to index using edge-offset mapping:
    # left-most edge at x = -4.0 * tile_size corresponds to file 0
    file_f = np.floor((x / tile_size) + 4.0)
    rank_f = np.floor((z / tile_size) + 4.0)

    if np.isnan(file_f) or np.isnan(rank_f):
        return None

    file_i = int(file_f)
    rank_i = int(rank_f)

    if 0 <= file_i <= 7 and 0 <= rank_i <= 7:
        return (file_i, rank_i)
    return None


def ray_sphere_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                         sphere_center: np.ndarray, sphere_radius: float) -> Optional[float]:
    """
    Ray-sphere intersection. Returns distance t from ray_origin to first hit,
    or None if no intersection.
    """
    # Solve |o + t d - c|^2 = r^2
    o = ray_origin.astype(np.float64)
    d = ray_dir.astype(np.float64)
    c = sphere_center.astype(np.float64)
    r = float(sphere_radius)

    oc = o - c
    b = 2.0 * np.dot(d, oc)
    c_term = np.dot(oc, oc) - r * r
    disc = b * b - 4.0 * c_term
    if disc < 0.0:
        return None
    sqrt_disc = np.sqrt(disc)
    t0 = (-b - sqrt_disc) / 2.0
    t1 = (-b + sqrt_disc) / 2.0

    # We want smallest positive t
    ts = [t for t in (t0, t1) if t >= 0.0]
    if not ts:
        return None
    return min(ts)


def pick_at_screen(x: float, y: float, width: int, height: int,
                   projection: np.ndarray, view: np.ndarray,
                   tile_size: float,
                   pieces: Iterable[Tuple[Tuple[int, int], Tuple[str, str]]],
                   piece_sphere_radius: float = 0.35) -> Dict[str, Any]:
    """
    High-level pick function.
    - pieces: iterable of ((file,rank), (piece_type, color))
    - piece_sphere_radius: radius in world units used for each piece bounding sphere.

    Returns dict:
    {
        "tile": (file, rank) or None,
        "world_pos": np.array([x,y,z]) or None,
        "piece": ((file,rank,piece_type,color), distance) or None
    }
    """
    ray_o, ray_d = screen_point_to_world_ray(x, y, width, height, projection, view)

    # Intersect board plane
    hit = intersect_ray_plane(ray_o, ray_d, plane_y=0.0)
    tile = None
    world_pos = None
    if hit is not None:
        t = world_pos_to_tile(hit, tile_size)
        if t is not None:
            tile = t
            world_pos = hit

    # Check pieces using bounding-sphere test (returns nearest hit)
    nearest = None
    nearest_dist = float("inf")

    # pieces: e.g. dict.items() -> ((file,rank), (piece_type, color))
    for (file_rank, (ptype, color)) in pieces:
        file_idx, rank_idx = file_rank
        # compute piece center on board (same layout as PieceRenderer._piece_transform)
        center_x = (file_idx - 3.5) * tile_size
        center_z = (rank_idx - 3.5) * tile_size
        center_y = 0.02  # same slight lift used for pieces
        center = np.array([center_x, center_y, center_z], dtype=np.float64)

        t = ray_sphere_intersect(ray_o, ray_d, center, piece_sphere_radius)
        if t is not None and t < nearest_dist:
            nearest = ((file_idx, rank_idx, ptype, color), float(t))
            nearest_dist = float(t)

    return {"tile": tile, "world_pos": world_pos, "piece": nearest}
