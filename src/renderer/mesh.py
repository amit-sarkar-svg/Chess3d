"""
src/renderer/mesh.py

Mesh abstraction for modern OpenGL (3.3+ Core Profile)
- Stores VAO, VBO, and optional EBO (index buffer)
- Supports positions, normals, and UVs
- Provides a clean draw() method
"""

from OpenGL import GL
import numpy as np
import ctypes


class Mesh:
    """
    Mesh object that uploads vertex data to GPU buffers and stores VAO/VBO/EBO.
    Expected vertex layout: position (3), normal (3), uv (2) = 8 floats per vertex

    Example:
        mesh = Mesh(vertices, indices)
        mesh.draw()
        mesh.delete()
    """

    def __init__(self, vertices: np.ndarray, indices: np.ndarray = None):
        """
        vertices: numpy array with shape (N, 8) — [px,py,pz, nx,ny,nz, u,v]
        indices : numpy array with shape (M,) or None
        """

        # Convert to float32 for OpenGL
        self.vertices = np.array(vertices, dtype=np.float32)

        self.indices = None
        if indices is not None:
            self.indices = np.array(indices, dtype=np.uint32)

        # Create GPU buffers
        self.VAO = GL.glGenVertexArrays(1)
        self.VBO = GL.glGenBuffers(1)
        self.EBO = GL.glGenBuffers(1) if self.indices is not None else None

        pos = self.vertices[:, 0:3] if self.vertices.size > 0 else np.zeros((0,3), dtype=np.float32)
        if pos.shape[0] > 0:
            self.bounds_min = pos.min(axis=0)
            self.bounds_max = pos.max(axis=0)
        else:
            self.bounds_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.bounds_max = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.vertex_count = self.vertices.shape[0]

        # Upload to GPU
        self._setup_mesh()

    # ---------------------------------------------------------------------
    # Mesh Setup
    # ---------------------------------------------------------------------
    def _setup_mesh(self):
        GL.glBindVertexArray(self.VAO)

        # ----- VBO -----
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.VBO)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self.vertices.nbytes,
            self.vertices,
            GL.GL_STATIC_DRAW
        )

        # ----- EBO (optional) -----
        if self.indices is not None:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.EBO)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                self.indices.nbytes,
                self.indices,
                GL.GL_STATIC_DRAW
            )

        # Each vertex = 8 floats (3 pos, 3 normal, 2 uv)
        stride = 8 * 4

        # Position attribute
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE,
            stride, ctypes.c_void_p(0)
        )

        # Normal attribute
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 3, GL.GL_FLOAT, GL.GL_FALSE,
            stride, ctypes.c_void_p(12)
        )

        # UV attribute
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(
            2, 2, GL.GL_FLOAT, GL.GL_FALSE,
            stride, ctypes.c_void_p(24)
        )

        GL.glBindVertexArray(0)

    # ---------------------------------------------------------------------
    # Draw Call
    # ---------------------------------------------------------------------
    def draw(self):
        """
        Draws the mesh using glDrawElements() if indexed,
        otherwise using glDrawArrays().
        """
        GL.glBindVertexArray(self.VAO)

        if self.indices is not None:
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                len(self.indices),
                GL.GL_UNSIGNED_INT,
                None
            )
        else:
            # vertices are flat array of N rows; draw count = number of vertices
            vertex_count = len(self.vertices)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, vertex_count)

        GL.glBindVertexArray(0)

    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    def delete(self):
        """
        Deletes GPU buffers. Call at shutdown.
        """
        GL.glDeleteVertexArrays(1, [self.VAO])
        GL.glDeleteBuffers(1, [self.VBO])

        if self.EBO:
            GL.glDeleteBuffers(1, [self.EBO])

        self.VAO = None
        self.VBO = None
        self.EBO = None
