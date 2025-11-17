"""
src/renderer/model_loader.py

OBJ + Texture loader for Chess3D.
Works with:
- OBJ files with materials (MTL + textures)
- OBJ files without materials (raw geometry)
- Multi-mesh OBJ scenes
- Texture fallback

Loads into Mesh class:
    Mesh(vertices [px,py,pz,nx,ny,nz,u,v], indices)

Fully compatible with PyWavefront and your lathed models.
"""

import os
import numpy as np
from OpenGL.GL import *
from PIL import Image
from pywavefront import Wavefront

from .mesh import Mesh


# -------------------------------------------------------------
# Texture Cache
# -------------------------------------------------------------
_texture_cache = {}


def load_texture(path: str) -> int:
    """Load a texture from disk → OpenGL texture ID."""
    global _texture_cache

    if path in _texture_cache:
        return _texture_cache[path]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Texture file not found: {path}")

    image = Image.open(path).convert("RGBA")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL origin fix
    img_data = image.tobytes()
    width, height = image.size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    glGenerateMipmap(GL_TEXTURE_2D)

    # texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glBindTexture(GL_TEXTURE_2D, 0)

    _texture_cache[path] = tex_id
    return tex_id


# -------------------------------------------------------------
# OBJ Loader (NEW — works for ALL OBJ formats)
# -------------------------------------------------------------
def load_obj(file_path: str, default_texture: str = None):
    """
    Loads any OBJ file: with MTL, without MTL, multi-mesh, raw geometry.

    Returns:
        Mesh instance
        texture_id (or None)
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"OBJ file not found: {file_path}")

    # Parse OBJ
    scene = Wavefront(
        file_path,
        create_materials=True,
        collect_faces=True,
        parse=True
    )

    vertices = []
    indices = []

    # Try to detect texture path
    texture_path = None

    # ---------------------------------------------------------
    # Extract geometry from meshes (safe for all OBJ types)
    # ---------------------------------------------------------
    for mesh in scene.mesh_list:
        raw = mesh.vertices      # flattened list: pos,normal,uv
        faces = mesh.faces       # list of index triplets

        for face in faces:
            for idx in face:
                base = idx * 8  # each vertex = 8 floats (pos3, normal3, uv2)

                px, py, pz = raw[base:base + 3]
                nx, ny, nz = raw[base + 3:base + 6]

                # Some models DO NOT have UV → handle gracefully
                if len(raw) >= base + 8:
                    u, v = raw[base + 6:base + 8]
                else:
                    u, v = 0.0, 0.0

                vertices.append([px, py, pz, nx, ny, nz, u, v])
                indices.append(len(indices))

    # Convert to numpy arrays
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    mesh = Mesh(vertices, indices)

    # ---------------------------------------------------------
    # Texture loading logic
    # ---------------------------------------------------------
    # If OBJ had materials, try to read diffuse texture
    try:
        for material in scene.materials.values():
            if hasattr(material, "texture") and material.texture:
                texture_path = material.texture.path
                break
    except:
        pass

    # If still none → fallback
    if not texture_path and default_texture:
        texture_path = default_texture

    if texture_path:
        try:
            tex_id = load_texture(texture_path)
        except Exception as e:
            print(f"[ModelLoader] Failed to load texture: {e}")
            tex_id = None
    else:
        tex_id = None

    return mesh, tex_id
