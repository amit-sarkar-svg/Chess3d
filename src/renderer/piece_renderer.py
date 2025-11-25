"""
src/renderer/piece_renderer.py

Piece rendering system for Chess3D.

- Loads all chess piece models once
- Allows placing white/black pieces on 8x8 board
- Supports individual transforms per piece
- Highlighting selected piece
- Works in both perspective and orthographic modes
"""

import numpy as np
from OpenGL import GL

from .model_loader import load_obj
from .model_loader import create_primitive_piece


class PieceRenderer:
    def __init__(self, shader, asset_folder="assets/models/", tile_size=1.0):
        """
        shader       : Shader object (must support model/view/projection)
        asset_folder : path to directory containing all piece OBJ files
        tile_size    : board tile spacing (same as BoardRenderer)
        """

        self.shader = shader
        self.tile_size = tile_size
        self.asset_folder = asset_folder

        # Dictionary of models:
        # models["pawn"]["white"] -> (mesh, tex_id)
        self.models = {}
        self.scales = {}

        self._load_all_piece_models()

        # Selected piece (file, rank)
        self.selected_piece = None

        # Board dictionary: (file, rank) -> (piece_type, color)
        self.board_state = {}

    # ------------------------------------------------------------------------------------
    # Loading Models
    # ------------------------------------------------------------------------------------
    def _load_piece(self, name):
        """
        Loads both white and black variants of a chess piece model:
            pawn_white.obj
            pawn_black.obj
        """

        white_path = f"{self.asset_folder}/{name}_white.obj"
        black_path = f"{self.asset_folder}/{name}_black.obj"

        try:
            mesh_white, tex_white = load_obj(white_path)
        except Exception:
            mesh_white, tex_white = create_primitive_piece(name)
        try:
            mesh_black, tex_black = load_obj(black_path)
        except Exception:
            mesh_black, tex_black = create_primitive_piece(name)

        if getattr(mesh_white, "vertex_count", 0) == 0:
            mesh_white, tex_white = create_primitive_piece(name)
        if getattr(mesh_black, "vertex_count", 0) == 0:
            mesh_black, tex_black = create_primitive_piece(name)

        sx = mesh_white.bounds_max[0] - mesh_white.bounds_min[0]
        sz = mesh_white.bounds_max[2] - mesh_white.bounds_min[2]
        extent = max(1e-6, max(sx, sz))
        target = self.tile_size * 0.85
        self.scales[name] = float(target / extent)

        return {
            "white": (mesh_white, tex_white),
            "black": (mesh_black, tex_black)
        }

    def _load_all_piece_models(self):
        """
        Load piece models: king, queen, rook, bishop, knight, pawn.
        """

        piece_names = ["king", "queen", "rook", "bishop", "knight", "pawn"]

        for p in piece_names:
            try:
                self.models[p] = self._load_piece(p)
                print(f"[PieceRenderer] Loaded model: {p}")
            except Exception as e:
                print(f"[PieceRenderer] ERROR loading {p}: {e}")

    # ------------------------------------------------------------------------------------
    # Board Interaction
    # ------------------------------------------------------------------------------------
    def set_piece(self, file, rank, piece_type, color):
        """
        Place a piece on the board.
        piece_type = "pawn", "rook", "knight", "bishop", "queen", "king"
        color      = "white" or "black"
        """

        if piece_type not in self.models:
            raise ValueError(f"Unknown piece type: {piece_type}")

        if color not in ("white", "black"):
            raise ValueError("color must be 'white' or 'black'")

        self.board_state[(file, rank)] = (piece_type, color)

    def remove_piece(self, file, rank):
        """Remove a piece from a tile."""
        self.board_state.pop((file, rank), None)

    def set_selected_piece(self, file=None, rank=None):
        """Highlight the selected piece, or clear selection with None."""
        if file is None or rank is None:
            self.selected_piece = None
            return

        if (file, rank) in self.board_state:
            self.selected_piece = (file, rank)
        else:
            self.selected_piece = None

    # ------------------------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------------------------
    def draw(self, view, projection):
        """
        Draw all placed pieces on the board.
        """

        self.shader.use()
        self.shader.set_mat4("view", view)
        self.shader.set_mat4("projection", projection)

        self.shader.set_bool("enable_highlight", False)

        # Avoid culling issues with varied model windings
        GL.glDisable(GL.GL_CULL_FACE)
        for (file, rank), (piece_type, color) in self.board_state.items():

            mesh, tex = self.models[piece_type][color]

            # Model transform
            model = self._piece_transform(file, rank)
            self.shader.set_mat4("model", model)

            # Highlight if selected
            self.shader.set_bool("enable_highlight", self.selected_piece == (file, rank))

            # Texture bind
            GL.glActiveTexture(GL.GL_TEXTURE0)
            tex_id = 0 if self.models[piece_type][color][1] is None else int(self.models[piece_type][color][1])
            # Set texture usage/fallback color
            self.shader.set_bool("use_texture", tex_id != 0)
            if color == "white":
                self.shader.set_vec3("base_color", np.array([0.9, 0.9, 0.9], dtype=np.float32))
            else:
                self.shader.set_vec3("base_color", np.array([0.12, 0.12, 0.12], dtype=np.float32))
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            self.shader.set_int("texture0", 0)

            mesh.draw()

        # Unbind
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self.shader.stop()
        GL.glEnable(GL.GL_CULL_FACE)

    # ------------------------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------------------------
    def _piece_transform(self, file, rank):
        """
        Creates a model matrix for a piece standing on tile (file, rank).
        """

        x = (file - 3.5) * self.tile_size
        z = (rank - 3.5) * self.tile_size
        y = 0.02  # slight lift above board

        M = np.eye(4, dtype=np.float32)

        # Uniform scale for piece models
        S = self.scales.get("pawn", 0.65 * self.tile_size)
        key = self.board_state.get((file, rank))
        if key:
            ptype, _ = key
            S = self.scales.get(ptype, S)
        M[0, 0] = S
        M[1, 1] = S
        M[2, 2] = S

        # Translation (last column for GL column-major layout)
        M[0, 3] = x
        M[1, 3] = y
        M[2, 3] = z

        return M
