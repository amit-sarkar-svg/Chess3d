"""
src/renderer/board_renderer.py

Procedural Chessboard Renderer for Chess3D.

- Generates 64 tiles (8x8)
- Each tile uses either light or dark texture
- Uses Mesh class for GPU rendering
- Allows highlighting a selected tile
"""

import numpy as np
from OpenGL import GL

from .mesh import Mesh


class BoardRenderer:
    def __init__(self, shader, light_texture_id, dark_texture_id, tile_size=1.0, highlight_color=(1.0, 0.8, 0.1)):
        """
        shader = Shader object
        light_texture_id = OpenGL texture ID for light squares
        dark_texture_id  = OpenGL texture ID for dark squares
        tile_size = world space tile width/height
        highlight_color = RGB color tuple for highlight overlay
        """

        self.shader = shader
        self.light_tex = light_texture_id
        self.dark_tex = dark_texture_id
        self.tile_size = tile_size
        self.highlight_color = np.array(highlight_color, dtype=np.float32)

        # Selected tile coordinates (file, rank) or None
        self.selected_tile = None

        # Prepare meshes for 1 light tile and 1 dark tile
        self.light_mesh = self._create_tile_mesh()
        self.dark_mesh = self._create_tile_mesh()

    # ------------------------------------------------------------------------------------
    # Tile Geometry
    # ------------------------------------------------------------------------------------
    def _create_tile_mesh(self):
        """
        Create a single square tile mesh on the XZ plane.
        Centered at (0,0,0), 1x1 size → scaling is applied in model matrix.
        """

        # position(x,y,z) normal(x,y,z) uv(u,v)
        vertices = [
            # x, y, z,   nx, ny, nz,   u, v
            [-0.5, 0.0, -0.5,   0,1,0,   0,0],
            [ 0.5, 0.0, -0.5,   0,1,0,   1,0],
            [ 0.5, 0.0,  0.5,   0,1,0,   1,1],
            [-0.5, 0.0,  0.5,   0,1,0,   0,1],
        ]

        indices = [0, 1, 2,   2, 3, 0]

        return Mesh(np.array(vertices, dtype=np.float32),
                    np.array(indices, dtype=np.uint32))

    # ------------------------------------------------------------------------------------
    # Selection Logic
    # ------------------------------------------------------------------------------------
    def set_selected_tile(self, file=None, rank=None):
        """
        Set or clear the selected tile.
        Pass (file, rank) within 0..7, or None to clear selection.
        """
        if file is None or rank is None:
            self.selected_tile = None
            return

        if 0 <= file <= 7 and 0 <= rank <= 7:
            self.selected_tile = (file, rank)
        else:
            self.selected_tile = None

    # ------------------------------------------------------------------------------------
    # Draw Routine
    # ------------------------------------------------------------------------------------
    def draw(self, view, projection):
        """
        Draws all 64 tiles.
        `view` and `projection` are 4x4 numpy matrices.
        """

        self.shader.use()
        self.shader.set_mat4("view", view)
        self.shader.set_mat4("projection", projection)

        # Highlight color
        self.shader.set_vec3("highlight_color", self.highlight_color)
        self.shader.set_bool("enable_highlight", False)
        # Board always uses textures
        self.shader.set_bool("use_texture", True)

        # GL state
        GL.glActiveTexture(GL.GL_TEXTURE0)
        self.shader.set_int("texture0", 0)

        # Render all 64 tiles
        for file in range(8):
            for rank in range(8):

                # Determine tile type (light/dark)
                is_dark = (file + rank) % 2 == 1
                mesh = self.dark_mesh if is_dark else self.light_mesh
                tex = self.dark_tex if is_dark else self.light_tex

                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)

                # Create model matrix
                model = self._tile_transform(file, rank)
                self.shader.set_mat4("model", model)

                # Highlight if selected
                self.shader.set_bool("enable_highlight", self.selected_tile == (file, rank))

                mesh.draw()

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self.shader.stop()

    # ------------------------------------------------------------------------------------
    # Tile Transform
    # ------------------------------------------------------------------------------------
    def _tile_transform(self, file, rank):
        """
        Generate the model matrix for a tile at (file, rank).
        """
        x = (file - 3.5) * self.tile_size
        z = (rank - 3.5) * self.tile_size

        model = np.eye(4, dtype=np.float32)

        # Scale
        s = self.tile_size
        model[0, 0] = s
        model[2, 2] = s

        # Translate (last column for GL column-major layout)
        model[0, 3] = x
        model[2, 3] = z

        return model
