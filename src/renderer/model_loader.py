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
        # Prefer material-level vertices/faces if present (pywavefront >=1.x)
        materials = getattr(mesh, "materials", None)
        processed_any = False

        if materials:
            mats_iter = materials.values() if isinstance(materials, dict) else materials
            for mat in mats_iter:
                raw = getattr(mat, "vertices", None)
                faces = getattr(mat, "faces", None)
                if raw is None:
                    continue

                raw_len = len(raw)
                # If faces are present, use indexed extraction
                if faces:
                    processed_any = True
                    for face in faces:
                        for idx in face:
                            base = idx * 8
                            if raw_len < base + 6:
                                continue
                            px, py, pz = raw[base:base + 3]
                            nx, ny, nz = raw[base + 3:base + 6]
                            u, v = raw[base + 6:base + 8] if raw_len >= base + 8 else (0.0, 0.0)
                            vertices.append([px, py, pz, nx, ny, nz, u, v])
                            indices.append(len(indices))
                else:
                    # No faces → raw is already triangulated; append sequentially
                    processed_any = True
                    for base in range(0, raw_len, 8):
                        if raw_len < base + 6:
                            break
                        px, py, pz = raw[base:base + 3]
                        nx, ny, nz = raw[base + 3:base + 6]
                        u, v = raw[base + 6:base + 8] if raw_len >= base + 8 else (0.0, 0.0)
                        vertices.append([px, py, pz, nx, ny, nz, u, v])
        
        # Fallback to mesh-level vertices/faces (older pywavefront layouts)
        if not processed_any:
            raw = getattr(mesh, "vertices", None)
            faces = getattr(mesh, "faces", None)
            if raw is None:
                raise AttributeError("Mesh has no vertices/faces (material or mesh level)")

            raw_len = len(raw)
            if faces:
                for face in faces:
                    for idx in face:
                        base = idx * 8
                        if raw_len < base + 6:
                            continue
                        px, py, pz = raw[base:base + 3]
                        nx, ny, nz = raw[base + 3:base + 6]
                        u, v = raw[base + 6:base + 8] if raw_len >= base + 8 else (0.0, 0.0)
                        vertices.append([px, py, pz, nx, ny, nz, u, v])
                        indices.append(len(indices))
            else:
                # No faces → append sequentially
                for base in range(0, raw_len, 8):
                    if raw_len < base + 6:
                        break
                    px, py, pz = raw[base:base + 3]
                    nx, ny, nz = raw[base + 3:base + 6]
                    u, v = raw[base + 6:base + 8] if raw_len >= base + 8 else (0.0, 0.0)
                    vertices.append([px, py, pz, nx, ny, nz, u, v])

    if len(vertices) == 0:
        raw_positions = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                raw_positions.append((x, y, z))
        if len(raw_positions) >= 6:
            by_y = {}
            for x, y, z in raw_positions:
                ky = round(y, 5)
                by_y.setdefault(ky, []).append((x, y, z))
            rings = []
            for ky in sorted(by_y.keys()):
                ring = [(x, y, z) for (x, y, z) in by_y[ky] if abs(x) > 1e-6 or abs(z) > 1e-6]
                if len(ring) >= 3:
                    rings.append(ring)
            if len(rings) >= 2:
                n = max(len(r) for r in rings)
                rings_aligned = []
                for r in rings:
                    if len(r) == n:
                        rings_aligned.append(r)
                    else:
                        rr = r.copy()
                        while len(rr) < n:
                            rr.append(rr[len(rr) % len(r)])
                        rings_aligned.append(rr[:n])
                pos = []
                for r in rings_aligned:
                    pos.extend(r)
                vr = np.array(pos, dtype=np.float32)
                min_y = min(p[1] for p in pos)
                max_y = max(p[1] for p in pos)
                count_per_ring = n
                ring_count = len(rings_aligned)
                idx = []
                for ri in range(ring_count - 1):
                    for i in range(count_per_ring):
                        a = ri * count_per_ring + i
                        b = (ri + 1) * count_per_ring + i
                        c = (ri + 1) * count_per_ring + ((i + 1) % count_per_ring)
                        d = ri * count_per_ring + ((i + 1) % count_per_ring)
                        idx.extend([a, b, c, a, c, d])
                vn = np.zeros_like(vr)
                for t in range(0, len(idx), 3):
                    i0, i1, i2 = idx[t], idx[t + 1], idx[t + 2]
                    p0 = vr[i0]
                    p1 = vr[i1]
                    p2 = vr[i2]
                    nrm = np.cross(p1 - p0, p2 - p0)
                    l = np.linalg.norm(nrm)
                    if l > 1e-12:
                        nrm = nrm / l
                        vn[i0] += nrm
                        vn[i1] += nrm
                        vn[i2] += nrm
                for i in range(len(vn)):
                    l = np.linalg.norm(vn[i])
                    vn[i] = vn[i] / l if l > 1e-12 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
                uv = np.zeros((len(vr), 2), dtype=np.float32)
                for ri in range(ring_count):
                    for i in range(count_per_ring):
                        vi = ri * count_per_ring + i
                        u = i / float(count_per_ring)
                        v = 0.0 if max_y == min_y else (vr[vi][1] - min_y) / (max_y - min_y)
                        uv[vi] = [u, v]
                vertices = np.hstack([vr, vn, uv]).astype(np.float32)
                indices = np.array(idx, dtype=np.uint32)
    if len(vertices) == 0:
        vertices = np.array([[0.0, 0.0, 0.0, 0, 1, 0, 0, 0],
                             [0.5, 0.0, 0.0, 0, 1, 0, 1, 0],
                             [0.0, 0.5, 0.0, 0, 1, 0, 0, 1]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
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
                # "texture" may be a simple path or an object with a path
                tex_obj = material.texture
                if hasattr(tex_obj, "path"):
                    texture_path = tex_obj.path
                elif isinstance(tex_obj, str):
                    texture_path = tex_obj
                else:
                    # try common alternate attribute names
                    img_name = getattr(material, "image_name", None)
                    if isinstance(img_name, str):
                        texture_path = img_name
                if texture_path:
                    break
    except Exception:
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
def _make_cylinder(radius: float = 0.35, height: float = 1.2, segments: int = 32) -> Mesh:
    import math
    verts = []
    idx = []
    # side vertices
    for i in range(segments):
        ang = (i / segments) * 2.0 * math.pi
        x = math.cos(ang) * radius
        z = math.sin(ang) * radius
        nx, nz = math.cos(ang), math.sin(ang)
        # bottom and top
        verts.append([x, 0.0, z, nx, 0.0, nz, i / segments, 0.0])
        verts.append([x, height, z, nx, 0.0, nz, i / segments, 1.0])
    # side indices (two triangles per segment)
    for i in range(segments):
        i0 = (i * 2)
        i1 = ((i * 2) + 1)
        i2 = (((i + 1) % segments) * 2)
        i3 = (((i + 1) % segments) * 2 + 1)
        idx.extend([i0, i1, i3, i0, i3, i2])
    base_start = len(verts)
    # bottom center
    verts.append([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.5, 0.0])
    # bottom ring
    for i in range(segments):
        ang = (i / segments) * 2.0 * math.pi
        x = math.cos(ang) * radius
        z = math.sin(ang) * radius
        verts.append([x, 0.0, z, 0.0, -1.0, 0.0, (x / (2*radius)) + 0.5, (z / (2*radius)) + 0.5])
    for i in range(segments):
        c = base_start
        r = base_start + 1 + i
        r_next = base_start + 1 + ((i + 1) % segments)
        idx.extend([c, r_next, r])
    top_start = len(verts)
    # top center
    verts.append([0.0, height, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0])
    # top ring
    for i in range(segments):
        ang = (i / segments) * 2.0 * math.pi
        x = math.cos(ang) * radius
        z = math.sin(ang) * radius
        verts.append([x, height, z, 0.0, 1.0, 0.0, (x / (2*radius)) + 0.5, (z / (2*radius)) + 0.5])
    for i in range(segments):
        c = top_start
        r = top_start + 1 + i
        r_next = top_start + 1 + ((i + 1) % segments)
        idx.extend([c, r, r_next])
    return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))

def create_primitive_piece(name: str) -> tuple:
    """Return a simple Mesh, None texture for a given piece name."""
    height_map = {
        "pawn": 1.0,
        "rook": 1.1,
        "knight": 1.15,
        "bishop": 1.2,
        "queen": 1.3,
        "king": 1.35,
    }
    h = height_map.get(name, 1.0)
    mesh = _make_cylinder(radius=0.35, height=h, segments=32)
    return mesh, None
