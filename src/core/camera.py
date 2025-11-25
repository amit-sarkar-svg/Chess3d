"""
src/core/camera.py

OrbitCamera supporting Perspective <-> Orthographic switching with smooth transitions.

Dependencies:
    - numpy

Usage sketch:
    cam = OrbitCamera(mode="perspective", radius=8.0, target=np.array([0.0, 0.0, 0.0]))
    # in update loop:
    cam.update(dt)
    view = cam.get_view_matrix()
    proj = cam.get_projection_matrix(aspect=width/height)
    # on input:
    cam.process_mouse_move(dx, dy, buttons=...)   # implement orbit/pan depending on button
    cam.process_scroll(yoffset)                   # zoom in/out
    cam.transition_to_mode("orthographic", duration=0.8)
"""

from typing import Tuple, Optional
import numpy as np
import math
import time


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a right-handed look-at view matrix (4x4)."""
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    z = normalize(eye - target)         # forward (camera looks towards -z in view space)
    x = normalize(np.cross(up, z))      # right
    y = np.cross(z, x)                  # true up

    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = x
    mat[1, :3] = y
    mat[2, :3] = z
    mat[0, 3] = -np.dot(x, eye)
    mat[1, 3] = -np.dot(y, eye)
    mat[2, 3] = -np.dot(z, eye)
    return mat


def perspective(fov_y_rad: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    """Perspective projection matrix (right-handed, column-major-like layout for GL)."""
    f = 1.0 / math.tan(fov_y_rad * 0.5)
    depth = z_near - z_far

    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / depth
    m[2, 3] = (2.0 * z_far * z_near) / depth
    m[3, 2] = -1.0
    return m


def orthographic(left: float, right: float, bottom: float, top: float, z_near: float, z_far: float) -> np.ndarray:
    """Orthographic projection matrix."""
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (z_far - z_near)
    m[3, 3] = 1.0
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(z_far + z_near) / (z_far - z_near)
    return m


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * np.array(a, dtype=np.float64) + t * np.array(b, dtype=np.float64)


class CameraParams:
    """Simple container for camera-relevant parameters that we want to interpolate."""
    def __init__(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
        fov_y: float,
        ortho_half_size: float
    ):
        self.eye = np.array(eye, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up = normalize(np.array(up, dtype=np.float64))
        self.fov_y = float(fov_y)                  # radians, used in perspective
        self.ortho_half_size = float(ortho_half_size)  # half-size used for orthographic (in world units)

    def copy(self):
        return CameraParams(self.eye.copy(), self.target.copy(), self.up.copy(), self.fov_y, self.ortho_half_size)


class OrbitCamera:
    """
    OrbitCamera supports both perspective and orthographic behaviors.
    - Orbit around a target using spherical coords (theta, phi) derived from eye-target vector.
    - Zoom in/out: adjusts radius (perspective) or ortho_half_size (orthographic).
    - Pan: shift target in camera plane.
    """

    def __init__(
        self,
        mode: str = "perspective",                # "perspective" or "orthographic"
        eye: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        up: Optional[np.ndarray] = None,
        fov_y: float = math.radians(50.0),
        ortho_half_size: float = 6.0,
        min_radius: float = 1.0,
        max_radius: float = 50.0
    ):
        self.mode = mode
        self.target = np.array(target if target is not None else np.array([0.0, 0.0, 0.0]), dtype=np.float64)
        if eye is None:
            # default to a nice angled view
            eye = np.array([0.0, 8.0, 8.0], dtype=np.float64)
        self.eye = np.array(eye, dtype=np.float64)
        self.up = np.array(up if up is not None else np.array([0.0, 1.0, 0.0]), dtype=np.float64)

        # spherical coords for orbiting
        self._update_spherical_from_eye()

        # zoom
        self.radius = max(1e-6, np.linalg.norm(self.eye - self.target))
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)

        # perspective settings
        self.fov_y = float(fov_y)
        # orthographic settings: half size (vertical)
        self.ortho_half_size = float(ortho_half_size)

        # input sensitivity
        self.orbit_sensitivity = 0.004
        self.pan_sensitivity = 0.002
        self.zoom_sensitivity = 1.0

        # smoothing / interpolation: by default instant
        self.smooth_enabled = True
        self.smooth_factor = 12.0  # higher -> snappier

        # internal target params for smoothing
        self._target_eye = self.eye.copy()
        self._target_target = self.target.copy()
        self._target_up = self.up.copy()
        self._target_fov_y = self.fov_y
        self._target_ortho_half_size = self.ortho_half_size

    def _update_spherical_from_eye(self):
        vec = self.eye - self.target
        self.radius = np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-9 else 1e-9
        # theta: azimuth around Y (0 = +X), phi: elevation from Y axis
        self.theta = math.atan2(vec[2], vec[0])
        self.phi = math.acos(max(-1.0, min(1.0, vec[1] / self.radius)))

    def _recompute_eye_from_spherical(self):
        # spherical -> cartesian
        sin_phi = math.sin(self.phi)
        x = self.radius * sin_phi * math.cos(self.theta)
        y = self.radius * math.cos(self.phi)
        z = self.radius * sin_phi * math.sin(self.theta)
        self._target_eye = self.target + np.array([x, y, z], dtype=np.float64)

    # ---------- Public API ----------
    def get_view_matrix(self) -> np.ndarray:
        """Returns the current view matrix (4x4)."""
        # If smoothing enabled, interpolate a bit towards target values
        if self.smooth_enabled:
            eye = lerp(self.eye, self._target_eye, 1.0 - math.exp(-self.smooth_factor * (1/60.0)))
            target = lerp(self.target, self._target_target, 1.0 - math.exp(-self.smooth_factor * (1/60.0)))
            up = normalize(lerp(self.up, self._target_up, 1.0 - math.exp(-self.smooth_factor * (1/60.0))))
        else:
            eye = self._target_eye.copy()
            target = self._target_target.copy()
            up = self._target_up.copy()

        return look_at(eye, target, up)

    def get_projection_matrix(self, aspect: float, z_near: float = 0.1, z_far: float = 100.0) -> np.ndarray:
        aspect = 1.0 if aspect == 0 or not np.isfinite(aspect) else aspect
        if self.mode == "perspective":
            return perspective(self.fov_y, aspect, z_near, z_far)
        else:
            h = self.ortho_half_size
            w = h * aspect
            left, right = -w, w
            bottom, top = -h, h
            return orthographic(left, right, bottom, top, z_near, z_far)

    def update(self, dt: float) -> None:
        """
        Must be called each frame with elapsed seconds (dt).
        This performs smoothing and updates internal matrices.
        """
        # If smoothing disabled: copy targets immediately
        if not self.smooth_enabled:
            self.eye = self._target_eye.copy()
            self.target = self._target_target.copy()
            self.up = self._target_up.copy()
            self.fov_y = self._target_fov_y
            self.ortho_half_size = self._target_ortho_half_size
            return

        # Exponential smoothing towards targets (frame-rate independent)
        t = 1.0 - math.exp(-self.smooth_factor * dt)
        self.eye = lerp(self.eye, self._target_eye, t)
        self.target = lerp(self.target, self._target_target, t)
        self.up = normalize(lerp(self.up, self._target_up, t))
        self.fov_y = (1.0 - t) * self.fov_y + t * self._target_fov_y
        self.ortho_half_size = (1.0 - t) * self.ortho_half_size + t * self._target_ortho_half_size

    # ---------- Input handlers ----------
    def process_mouse_move(self, dx: float, dy: float, button: str = "left", shift: bool = False) -> None:
        """
        Call with mouse deltas (pixels). button: "left"=orbit, "middle"=pan, "right"=pan
        If shift is True, invert pan behavior or reduce sensitivity.
        """
        if button == "left":
            # orbit
            self.theta -= dx * self.orbit_sensitivity
            self.phi += dy * self.orbit_sensitivity
            # clamp phi (avoid flipping)
            eps = 1e-3
            self.phi = max(eps, min(math.pi - eps, self.phi))
            # recompute desired eye from spherical coords
            self._recompute_eye_from_spherical()
        elif button in ("middle", "right"):
            # pan: move the target in camera local axes based on camera right/up
            # Use current (smoothed) view basis
            view = look_at(self._target_eye, self._target_target, self._target_up)
            right = view[0, :3]   # camera right vector
            up = view[1, :3]      # camera up vector
            sign = -1.0 if not shift else -0.5
            self._target_target += (right * (-dx) + up * (dy)) * self.pan_sensitivity * sign * (self.radius * 0.02)

            # After panning, update spherical center
            self._recompute_eye_from_spherical()
        else:
            # no-op for other buttons
            pass

    def process_scroll(self, yoffset: float) -> None:
        """Zoom in/out. Positive yoffset = scroll up (zoom in)."""
        if self.mode == "perspective":
            # change radius
            factor = 0.85 ** (yoffset * self.zoom_sensitivity)
            self.radius = max(self.min_radius, min(self.max_radius, self.radius * factor))
            self._recompute_eye_from_spherical()
        else:
            # orthographic: change half-size
            self._target_ortho_half_size = max(0.5, self.ortho_half_size * (0.9 ** (yoffset * self.zoom_sensitivity)))
            # Immediately apply for simplicity
            self.ortho_half_size = self._target_ortho_half_size

    def set_mode(self, mode: str) -> None:
        """Direct switch without transition. mode = 'perspective'|'orthographic'."""
        if mode not in ("perspective", "orthographic"):
            raise ValueError("mode must be 'perspective' or 'orthographic'")
        self.mode = mode

    # ---------- Transitions ----------
    def transition_to(self, params: CameraParams, duration: float = 0.8) -> None:
        """
        Start a smooth transition to the given CameraParams over duration seconds.
        The transition updates internal target_* values and can be stepped by update(dt).
        """
        # set end targets immediately; actual smoothing occurs in update()
        self._target_eye = params.eye.copy()
        self._target_target = params.target.copy()
        self._target_up = normalize(params.up.copy())
        self._target_fov_y = float(params.fov_y)
        self._target_ortho_half_size = float(params.ortho_half_size)
        # For orbit control consistency, recompute spherical from final eye/target
        self._update_spherical_from_eye()

    def snapshot_params(self) -> CameraParams:
        """Return a snapshot of current camera parameters (useful for transitions)."""
        return CameraParams(self._target_eye.copy(), self._target_target.copy(), self._target_up.copy(), self._target_fov_y, self._target_ortho_half_size)


class CameraManager:
    """
    Convenience wrapper to handle mode switching with an animated transition.
    Example usage:
        manager = CameraManager(camera)
        manager.to_mode("orthographic", duration=0.9)
        # in loop: manager.update(dt)
    """

    def __init__(self, camera: OrbitCamera):
        self.camera = camera
        self._transitioning = False
        self._transition_time = 0.0
        self._transition_duration = 0.0
        self._start_params: Optional[CameraParams] = None
        self._end_params: Optional[CameraParams] = None
        self._start_mode: Optional[str] = None
        self._end_mode: Optional[str] = None

    def to_mode(self, mode: str, duration: float = 0.8):
        """Begin a smooth transition to the requested mode."""
        if mode not in ("perspective", "orthographic", "isometric"):
            raise ValueError("mode must be 'perspective' or 'orthographic' or 'isometric'")

        self._start_mode = self.camera.mode
        self._end_mode = mode
        self._start_params = self.camera.snapshot_params()
        # Build sensible end params: keep eye/target/up, but adjust fov/ortho size defaults if switching types
        end_params = self._start_params.copy()
        if mode == "perspective":
            end_params.fov_y = math.radians(50.0)
        elif mode == "orthographic":
            dist = np.linalg.norm(self._start_params.eye - self._start_params.target)
            end_params.ortho_half_size = max(3.0, float(dist * 0.7))
        elif mode == "isometric":
            dist = np.linalg.norm(self._start_params.eye - self._start_params.target)
            end_params.ortho_half_size = max(3.0, float(dist * 0.7))
            phi = math.radians(54.7356)
            theta = math.radians(45.0)
            sin_phi = math.sin(phi)
            x = dist * sin_phi * math.cos(theta)
            y = dist * math.cos(phi)
            z = dist * sin_phi * math.sin(theta)
            end_params.eye = self._start_params.target + np.array([x, y, z], dtype=np.float64)

        self._end_params = end_params
        self._transition_time = 0.0
        self._transition_duration = max(1e-3, float(duration))
        self._transitioning = True

    def update(self, dt: float):
        """Call every frame with dt; steps the transition if active, otherwise lets camera.update(dt)."""
        if not self._transitioning:
            self.camera.update(dt)
            return

        self._transition_time += dt
        t = min(1.0, self._transition_time / self._transition_duration)
        # Smoothstep easing
        t_eased = t * t * (3.0 - 2.0 * t)

        # Interpolate relevant params
        sp = self._start_params
        ep = self._end_params
        interp_eye = lerp(sp.eye, ep.eye, t_eased)
        interp_target = lerp(sp.target, ep.target, t_eased)
        interp_up = normalize(lerp(sp.up, ep.up, t_eased))
        interp_fov = (1.0 - t_eased) * sp.fov_y + t_eased * ep.fov_y
        interp_ortho = (1.0 - t_eased) * sp.ortho_half_size + t_eased * ep.ortho_half_size

        # Apply to camera's target_* fields (so smoothing still applies)
        self.camera._target_eye = interp_eye
        self.camera._target_target = interp_target
        self.camera._target_up = interp_up
        self.camera._target_fov_y = interp_fov
        self.camera._target_ortho_half_size = interp_ortho

        # Interpolate mode switching near the end
        if t >= 0.98:
            # finalize mode and targets
            self.camera.mode = self._end_mode
            self.camera._target_fov_y = interp_fov
            self.camera._target_ortho_half_size = interp_ortho
            self._transitioning = False

        # Always also run camera.update to keep internal smoothing consistent
        self.camera.update(dt)
