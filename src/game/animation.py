"""
src/game/animation.py

AnimationManager for chess piece movement and capture animations.

Features:
- move_piece(from_pos, to_pos, duration)  : smooth slide, then commits to board_state
- capture_piece(at_pos, duration)         : fade-out capture (commits removal)
- spawn_piece(at_pos, piece_type, color)  : optional pop-in spawn animation
- update(dt)                              : advance time; must be called each frame
- get_animated_entries()                  : returns list of animated render items for drawing overlay

Integration:
- Pass a PieceRenderer instance to AnimationManager(...) so it can access models and modify board_state.
- In your main render loop:
    animation.update(dt)
    # Draw board & static pieces as usual
    board.draw(view, proj)
    pieces.draw(view, proj)   # draws current board_state (pieces not currently animated-from are present)
    # Then draw animated transient entries returned by animation.get_animated_entries()
    for entry in animation.get_animated_entries():
        shader.use(); shader.set_mat4("model", entry["model"]); shader.set_float("alpha", entry["alpha"])
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, entry["texture"]); shader.set_int("texture0",0)
        entry["mesh"].draw()
        glBindTexture(GL_TEXTURE_2D, 0)
        shader.stop()

Notes:
- This manager mutates PieceRenderer.board_state (removes pieces from source on move/capture immediately, places them in destination when move completes).
- Shapes/positions use the same conventions as PieceRenderer._piece_transform (tile center calc).
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import math


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in/out (0..1 -> 0..1)."""
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        u = (2.0 * t) - 2.0
        return 0.5 * u * u * u + 1.0


def make_model_matrix(pos: Tuple[float, float, float], scale: float = 1.0) -> np.ndarray:
    """Create a simple TRS 4x4 model matrix (column-major friendly for our shader.set_mat4)."""
    x, y, z = pos
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = scale
    M[1, 1] = scale
    M[2, 2] = scale
    M[3, 0] = x
    M[3, 1] = y
    M[3, 2] = z
    return M


def tile_center(file: int, rank: int, tile_size: float = 1.0, lift: float = 0.02) -> Tuple[float, float, float]:
    """Return piece center world position for a given tile (consistent with PieceRenderer)."""
    x = (file - 3.5) * tile_size
    z = (rank - 3.5) * tile_size
    y = lift
    return (x, y, z)


class _AnimEntry:
    """
    Internal structure describing one active animation.
    Types:
      - "move": moving a piece from src -> dst (keeps a reference to piece_type/color)
      - "capture": fading piece at pos (removal)
      - "spawn": pop-in at pos (add piece into board_state at end)
    """

    def __init__(self, kind: str, duration: float):
        self.kind = kind
        self.duration = max(1e-4, float(duration))
        self.time = 0.0
        # populated fields:
        self.piece_type: Optional[str] = None
        self.color: Optional[str] = None
        self.src: Optional[Tuple[int, int]] = None
        self.dst: Optional[Tuple[int, int]] = None
        self.pos: Optional[Tuple[float, float, float]] = None  # for capture/spawn
        self.on_complete = None  # optional callback


class AnimationManager:
    """
    AnimationManager ties into PieceRenderer (optional).

    Example:
        anim = AnimationManager(piece_renderer)
        anim.move_piece((4,1), (4,3), duration=0.6)
        anim.update(dt)
        overlay = anim.get_animated_entries()
        # draw overlay entries after drawing normal pieces
    """

    def __init__(self, piece_renderer=None, tile_size: float = 1.0):
        """
        piece_renderer: optional PieceRenderer instance this manager will operate on.
                        If given, manager will read/write piece_renderer.board_state and models.
        tile_size: tile world size for placement calculations
        """
        self.piece_renderer = piece_renderer
        self.tile_size = tile_size

        # Active animations (list of _AnimEntry)
        self._active: List[_AnimEntry] = []

    # ------------------------------
    # High-level API
    # ------------------------------
    def move_piece(self, src: Tuple[int, int], dst: Tuple[int, int], duration: float = 0.6) -> bool:
        """
        Start a move animation for a piece located at src to dst.
        Returns True if animation started; False if no piece at src.
        Immediately removes the piece from piece_renderer.board_state (so logical state doesn't double-draw).
        The piece is re-added to dst when animation finishes.
        """
        if self.piece_renderer is None:
            raise RuntimeError("AnimationManager requires a PieceRenderer instance to operate")

        board = self.piece_renderer.board_state
        if src not in board:
            return False

        piece_type, color = board.pop(src)  # remove logically now
        anim = _AnimEntry("move", duration)
        anim.piece_type = piece_type
        anim.color = color
        anim.src = src
        anim.dst = dst
        # compute start/end world positions
        anim.start_pos = tile_center(src[0], src[1], self.tile_size)
        anim.end_pos = tile_center(dst[0], dst[1], self.tile_size)
        self._active.append(anim)
        return True

    def capture_piece(self, at: Tuple[int, int], duration: float = 0.5) -> bool:
        """
        Start a capture (fade-out) animation for the piece at `at`.
        Immediately removes the piece from piece_renderer.board_state; at end of animation it is gone.
        Returns True if capture animation started, False if no piece at location.
        """
        if self.piece_renderer is None:
            raise RuntimeError("AnimationManager requires a PieceRenderer instance to operate")

        board = self.piece_renderer.board_state
        if at not in board:
            return False

        piece_type, color = board.pop(at)  # remove now (captured)
        anim = _AnimEntry("capture", duration)
        anim.piece_type = piece_type
        anim.color = color
        anim.src = at
        anim.pos = tile_center(at[0], at[1], self.tile_size)
        self._active.append(anim)
        return True

    def spawn_piece(self, at: Tuple[int, int], piece_type: str, color: str, duration: float = 0.45) -> bool:
        """
        Spawn/pop-in animation: at the end it inserts the piece into board_state at 'at'.
        If a piece already exists at 'at', returns False.
        """
        if self.piece_renderer is None:
            raise RuntimeError("AnimationManager requires a PieceRenderer instance to operate")
        board = self.piece_renderer.board_state
        if at in board:
            return False

        anim = _AnimEntry("spawn", duration)
        anim.piece_type = piece_type
        anim.color = color
        anim.dst = at
        anim.pos = tile_center(at[0], at[1], self.tile_size)
        self._active.append(anim)
        return True

    # ------------------------------
    # Update loop
    # ------------------------------
    def update(self, dt: float):
        """
        Advance all active animations. Call once per frame with elapsed seconds.
        Completed animations will commit final board_state changes (if needed).
        """
        finished = []
        for anim in self._active:
            anim.time += dt
            if anim.time >= anim.duration:
                finished.append(anim)

        # Process finished (commit effects)
        for anim in finished:
            self._complete_animation(anim)
            self._active.remove(anim)

    def _complete_animation(self, anim: _AnimEntry):
        """Called when an animation finishes; commit changes to piece_renderer.board_state as needed."""
        if self.piece_renderer is None:
            return

        board = self.piece_renderer.board_state

        if anim.kind == "move":
            # Place piece at destination
            if anim.dst in board:
                # if something occupies dst (shouldn't happen normally), overwrite
                board[anim.dst] = (anim.piece_type, anim.color)
            else:
                board[anim.dst] = (anim.piece_type, anim.color)

        elif anim.kind == "spawn":
            # Insert spawned piece
            board[anim.dst] = (anim.piece_type, anim.color)

        elif anim.kind == "capture":
            # already removed at start; nothing to do
            pass

        # optional callback
        if anim.on_complete:
            try:
                anim.on_complete()
            except Exception:
                pass

    # ------------------------------
    # Rendering helper
    # ------------------------------
    def get_animated_entries(self) -> List[Dict[str, Any]]:
        """
        Return a list of dictionaries representing transient animated render entries.
        Each dict has:
            - mesh: Mesh instance
            - texture: texture_id (int) or None
            - model: 4x4 numpy model matrix (float32)
            - alpha: float 0..1 (opacity)
        The caller should render these after rendering the static board/pieces.
        """
        entries = []
        if self.piece_renderer is None:
            return entries

        for anim in self._active:
            t = max(0.0, min(1.0, anim.time / anim.duration))
            eased = ease_in_out_cubic(t)

            if anim.kind == "move":
                sx, sy, sz = anim.start_pos
                ex, ey, ez = anim.end_pos
                cx = sx + (ex - sx) * eased
                cy = sy + (ey - sy) * eased
                cz = sz + (ez - sz) * eased
                # slight vertical arc for nicer motion
                arc = math.sin(math.pi * eased) * (0.06 * (self.tile_size))
                pos = (cx, cy + arc, cz)
                scale = 0.75  # match PieceRenderer default scale
                alpha = 1.0

            elif anim.kind == "capture":
                # capture: fade out and scale down
                pos = anim.pos
                scale = 0.75 * (1.0 - eased)
                alpha = 1.0 - eased

            elif anim.kind == "spawn":
                pos = anim.pos
                scale = 0.75 * (0.2 + 0.8 * eased)
                alpha = eased

            else:
                continue

            # find mesh + texture from piece_renderer.models
            pr = self.piece_renderer
            mesh = None
            tex = None
            try:
                mesh, tex = pr.models[anim.piece_type][anim.color]
            except Exception:
                # model missing — skip
                continue

            model = make_model_matrix(pos, scale)
            entries.append({
                "mesh": mesh,
                "texture": tex,
                "model": model,
                "alpha": float(alpha),
                "piece_type": anim.piece_type,
                "color": anim.color,
                "kind": anim.kind,
            })

        return entries

    # ------------------------------
    # Utility / Debug
    # ------------------------------
    def has_active(self) -> bool:
        return len(self._active) > 0
