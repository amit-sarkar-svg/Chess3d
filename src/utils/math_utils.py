"""
src/utils/math_utils.py

General-purpose math utilities for Chess3D Engine.

Provides:
- vector normalization
- clamp
- lerp (linear interpolation)
- smoothstep
- distance / length
- direction vector
- reflection
- ray helpers

All operations are NumPy-based for speed and consistency.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def smoothstep(t):
    """Smooth nonlinear interpolation (0..1)."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# -----------------------------------------------------------------------------
# Vector utilities
# -----------------------------------------------------------------------------
def length(v):
    return float(np.linalg.norm(v))


def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def normalize(v, eps=1e-8):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def direction(a, b):
    """Vector pointing from a → b."""
    return normalize(np.array(b) - np.array(a))


def dot(a, b):
    return float(np.dot(a, b))


def reflect(vector, normal):
    """Reflect vector against normal."""
    vector = np.array(vector, dtype=np.float32)
    normal = normalize(normal)
    return vector - 2 * np.dot(vector, normal) * normal


# -----------------------------------------------------------------------------
# Ray utilities
# -----------------------------------------------------------------------------
def ray_point(origin, direction, distance):
    """Get world position at distance along ray."""
    return np.array(origin) + np.array(direction) * distance


def ray_from_screen(x, y, width, height, projection, view):
    """
    Convert screen x,y → world ray.

    projection: 4x4 matrix
    view:       4x4 matrix

    Returns:
        (origin, direction)
    """

    # Convert to NDC range [-1, +1]
    ndc_x = (2.0 * x) / width - 1.0
    ndc_y = 1.0 - (2.0 * y) / height  # invert Y

    # In clip space
    ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)

    # Inverse matrices
    inv_proj = np.linalg.inv(projection)
    inv_view = np.linalg.inv(view)

    # Convert clip → view
    ray_eye = inv_proj @ ray_clip
    ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0], dtype=np.float32)

    # Convert view → world
    ray_world = inv_view @ ray_eye
    ray_world = normalize(ray_world[:3])

    # Camera position is origin of ray
    cam_pos = inv_view @ np.array([0, 0, 0, 1], dtype=np.float32)

    return cam_pos[:3], ray_world
