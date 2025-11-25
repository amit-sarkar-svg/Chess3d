"""
src/core/window.py

Window + GLFW wrapper for Chess3D Engine.

Provides:
- Easy window creation
- GLFW initialization
- OpenGL context setup
- Viewport handling (auto-resize)
- Time tracking (delta time)
- Integration with Input and Timer systems
"""

import glfw
from OpenGL import GL
from .timer import Timer


class Window:
    def __init__(self, width=1280, height=720, title="Chess3D"):
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        self.timer = Timer()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def init(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # For MacOS
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        GL.glViewport(0, 0, self.width, self.height)

        # Enable vsync
        glfw.swap_interval(1)

        # Setup resize callback
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_resize_callback)

        # OpenGL settings
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)
        GL.glClearColor(0.1, 0.1, 0.12, 1.0)

        print("[Window] Initialized successfully")

    # -------------------------------------------------------------------------
    # Callback forwarding
    # (Input callbacks will be attached externally from main.py)
    # -------------------------------------------------------------------------
    def set_key_callback(self, fn):
        glfw.set_key_callback(self.window, fn)

    def set_cursor_pos_callback(self, fn):
        glfw.set_cursor_pos_callback(self.window, fn)

    def set_mouse_button_callback(self, fn):
        glfw.set_mouse_button_callback(self.window, fn)

    def set_scroll_callback(self, fn):
        glfw.set_scroll_callback(self.window, fn)

    # -------------------------------------------------------------------------
    # Resize handler
    # -------------------------------------------------------------------------
    def _framebuffer_resize_callback(self, window, width, height):
        self.width = width
        self.height = height
        GL.glViewport(0, 0, width, height)

    # -------------------------------------------------------------------------
    # Time update
    # -------------------------------------------------------------------------
    def update_time(self):
        self.timer.update()
        self.delta_time = self.timer.delta_time

    # -------------------------------------------------------------------------
    # Event / buffer handling
    # -------------------------------------------------------------------------
    def swap_buffers(self):
        glfw.swap_buffers(self.window)

    def poll_events(self):
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def terminate(self):
        glfw.terminate()
        print("[Window] Terminated")
