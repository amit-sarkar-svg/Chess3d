"""
src/core/input.py

Unified input system for Chess3D.
Wraps GLFW input into a clean, engine-level interface.

Features:
- Key press / release tracking
- Mouse button state
- Mouse cursor position + movement delta
- Scroll wheel input
- Resettable per-frame delta
"""

import glfw


class Input:
    _keys_down = set()
    _mouse_buttons = set()

    _mouse_x = 0.0
    _mouse_y = 0.0
    _mouse_dx = 0.0
    _mouse_dy = 0.0
    _mouse_first = True

    _scroll_y = 0.0   # scroll delta per frame

    # -------------------------------------------------------------------------
    # GLFW CALLBACKS (Window class should forward these)
    # -------------------------------------------------------------------------
    @staticmethod
    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            Input._keys_down.add(key)
        elif action == glfw.RELEASE:
            Input._keys_down.discard(key)

    @staticmethod
    def mouse_button_callback(window, button, action, mods):
        if action == glfw.PRESS:
            Input._mouse_buttons.add(button)
        elif action == glfw.RELEASE:
            Input._mouse_buttons.discard(button)

    @staticmethod
    def cursor_pos_callback(window, xpos, ypos):
        if Input._mouse_first:
            Input._mouse_x = xpos
            Input._mouse_y = ypos
            Input._mouse_first = False
            return

        Input._mouse_dx = xpos - Input._mouse_x
        Input._mouse_dy = ypos - Input._mouse_y

        Input._mouse_x = xpos
        Input._mouse_y = ypos

    @staticmethod
    def scroll_callback(window, xoffset, yoffset):
        Input._scroll_y += yoffset  # accumulate scroll for this frame

    # -------------------------------------------------------------------------
    # QUERY FUNCTIONS
    # -------------------------------------------------------------------------
    @staticmethod
    def is_key_down(key) -> bool:
        return key in Input._keys_down

    @staticmethod
    def is_mouse_down(button) -> bool:
        return button in Input._mouse_buttons

    @staticmethod
    def get_mouse_pos():
        return Input._mouse_x, Input._mouse_y

    @staticmethod
    def get_mouse_delta():
        return Input._mouse_dx, Input._mouse_dy

    @staticmethod
    def get_scroll_y():
        return Input._scroll_y

    # -------------------------------------------------------------------------
    # PER-FRAME RESET
    # -------------------------------------------------------------------------
    @staticmethod
    def end_frame_reset():
        """
        Call this ONCE per frame at end of main loop.
        Clears scroll and mouse delta for next frame.
        """
        Input._mouse_dx = 0.0
        Input._mouse_dy = 0.0
        Input._scroll_y = 0.0
