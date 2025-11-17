"""
src/core/transformations.py

Matrix / vector transformation utilities for Chess3D Engine.
Provides:
- perspective projection
- orthographic projection
- lookAt camera matrix
- translation, rotation, scaling
- matrix composition helpers

All matrices are 4x4 numpy float32 arrays.
"""

from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# Basic Transformation Matrices
# -----------------------------------------------------------------------------
def translation(x: float, y: float, z: float) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[3, 0] = x
    M[3, 1] = y
    M[3, 2] = z
    return M


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def rotation_x(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    M = np.eye(4, dtype=np.float32)
    M[1, 1] = c
    M[1, 2] = s
    M[2, 1] = -s
    M[2, 2] = c
    return M


def rotation_y(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = c
    M[0, 2] = -s
    M[2, 0] = s
    M[2, 2] = c
    return M


def rotation_z(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = c
    M[0, 1] = s
    M[1, 0] = -s
    M[1, 1] = c
    return M


# -----------------------------------------------------------------------------
# Projection Matrices
# -----------------------------------------------------------------------------
def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Standard perspective projection matrix.
    """
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    nf = 1.0 / (near - far)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) * nf
    M[2, 3] = -1.0
    M[3, 2] = 2 * far * near * nf
    return M


def orthographic(left, right, bottom, top, near, far) -> np.ndarray:
    """
    Orthographic projection (used for orthographic camera mode).
    """
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (far - near)
    M[3, 0] = -(right + left) / (right - left)
    M[3, 1] = -(top + bottom) / (top - bottom)
    M[3, 2] = -(far + near) / (far - near)
    return M


# -----------------------------------------------------------------------------
# Camera LookAt
# -----------------------------------------------------------------------------
def look_at(eye, target, up) -> np.ndarray:
    """
    Creates a right-handed view matrix.
    eye:     camera position
    target:  point camera is looking at
    up:      camera up direction
    """

    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # forward = normalize(target - eye)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)

    # right = normalize(cross(f, up))
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-8)

    # up2 = cross(right, forward)
    u = np.cross(r, f)

    M = np.eye(4, dtype=np.float32)
    M[0, 0:3] = r
    M[1, 0:3] = u
    M[2, 0:3] = -f

    # translation
    M[3, 0] = -np.dot(r, eye)
    M[3, 1] = -np.dot(u, eye)
    M[3, 2] =  np.dot(f, eye)

    return M


# -----------------------------------------------------------------------------
# Helper: Combine transformations
# -----------------------------------------------------------------------------
def compose(*matrices) -> np.ndarray:
    """
    Compose multiple transformations (right to left multiplication).
    Example:
        M = compose(translation(1,0,0), rotation_y(45), scale(1,2,1))
    """
    M = np.eye(4, dtype=np.float32)
    for m in matrices:
        M = M @ m
    return M
