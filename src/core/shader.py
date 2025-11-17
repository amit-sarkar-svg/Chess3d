"""
src/core/shader.py

Modern OpenGL Shader Program Loader for Chess3D.
- Loads GLSL shader files from disk
- Compiles & links programs with detailed error logs
- Caches uniform locations for faster access
- Provides helper methods for setting uniforms (int, float, vec, mat4)
"""

import os
from OpenGL.GL import *
import numpy as np


class Shader:
    def __init__(self, vertex_path: str, fragment_path: str):
        self.vertex_path = vertex_path
        self.fragment_path = fragment_path

        self.program_id = None
        self.uniform_cache = {}

        self._load_and_compile()

    # ------------------------------------------------------------
    # Core Shader Compilation
    # ------------------------------------------------------------
    def _load_and_compile(self):
        # Read GLSL files
        vertex_src = self._read_file(self.vertex_path)
        fragment_src = self._read_file(self.fragment_path)

        # Create, compile, attach, link
        vertex_shader = self._compile_shader(vertex_src, GL_VERTEX_SHADER, self.vertex_path)
        fragment_shader = self._compile_shader(fragment_src, GL_FRAGMENT_SHADER, self.fragment_path)

        # Create program & link
        self.program_id = glCreateProgram()
        glAttachShader(self.program_id, vertex_shader)
        glAttachShader(self.program_id, fragment_shader)
        glLinkProgram(self.program_id)

        # Check linking errors
        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            error_log = glGetProgramInfoLog(self.program_id).decode()
            raise RuntimeError(f"Shader Linking Failed:\n{error_log}")

        # Clean up shaders after linking
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def _read_file(self, path: str) -> str:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Shader file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _compile_shader(self, source: str, shader_type: int, filename: str) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        # Check compilation errors
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error_log = glGetShaderInfoLog(shader).decode()
            shader_type_str = "VERTEX" if shader_type == GL_VERTEX_SHADER else "FRAGMENT"
            raise RuntimeError(f"{shader_type_str} Shader Compilation Error in '{filename}':\n{error_log}")

        return shader

    # ------------------------------------------------------------
    # Using the Shader
    # ------------------------------------------------------------
    def use(self):
        glUseProgram(self.program_id)

    def stop(self):
        glUseProgram(0)

    # ------------------------------------------------------------
    # Uniform Helpers
    # ------------------------------------------------------------
    def _get_uniform_location(self, name: str) -> int:
        # Uniform caching for performance
        if name in self.uniform_cache:
            return self.uniform_cache[name]
        location = glGetUniformLocation(self.program_id, name)
        if location == -1:
            # Not fatal: sometimes uniforms are optimized out
            print(f"[Shader Warning] Uniform '{name}' not found or optimized out.")
        self.uniform_cache[name] = location
        return location

    # ----- Int / Float -----
    def set_bool(self, name: str, value: bool):
        glUniform1i(self._get_uniform_location(name), int(value))

    def set_int(self, name: str, value: int):
        glUniform1i(self._get_uniform_location(name), value)

    def set_float(self, name: str, value: float):
        glUniform1f(self._get_uniform_location(name), value)

    # ----- Vec2 / Vec3 / Vec4 -----
    def set_vec2(self, name: str, vec):
        glUniform2f(self._get_uniform_location(name), vec[0], vec[1])

    def set_vec3(self, name: str, vec):
        glUniform3f(self._get_uniform_location(name), vec[0], vec[1], vec[2])

    def set_vec4(self, name: str, vec):
        glUniform4f(self._get_uniform_location(name), vec[0], vec[1], vec[2], vec[3])

    # ----- Mat4 -----
    def set_mat4(self, name: str, mat4):
        """
        Set 4x4 matrix uniform (numpy array) as float32.
        """
        glUniformMatrix4fv(
            self._get_uniform_location(name),
            1,
            GL_FALSE,
            np.array(mat4, dtype=np.float32)
        )

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    def delete(self):
        """
        Delete the shader program from GPU memory.
        Call at program exit.
        """
        if self.program_id:
            glDeleteProgram(self.program_id)
            self.program_id = None
