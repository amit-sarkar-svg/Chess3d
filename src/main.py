"""
main.py — Chess3D Engine (Final Version with Picking + Animation)

Features:
- Window + GLFW
- Perspective <-> Orthographic camera switching
- Picking system (tile + piece selection)
- Piece animations: move, capture, spawn
- BoardRenderer + PieceRenderer
- Clean and modular structure
"""

import numpy as np
from OpenGL import GL
import glfw

# Core engine modules
from core.window import Window
from core.camera import OrbitCamera, CameraManager
from core.shader import Shader

# Renderers
from renderer.board_renderer import BoardRenderer
from renderer.piece_renderer import PieceRenderer

# Game systems
from game.picking import pick_at_screen
from game.animation import AnimationManager


# -----------------------------------------------------------------------------
# GLOBAL INPUT STATE
# -----------------------------------------------------------------------------
mouse_last_x = 0
mouse_last_y = 0
mouse_first = True
mouse_button = None     # "left", "right", "middle", None

keys_down = set()

camera = None
camera_mgr = None
win = None
board = None
pieces = None
animation = None


# -----------------------------------------------------------------------------
# INPUT CALLBACKS
# -----------------------------------------------------------------------------
def key_callback(window, key, scancode, action, mods):
    global keys_down
    if action == glfw.PRESS:
        keys_down.add(key)
    elif action == glfw.RELEASE:
        keys_down.discard(key)


def mouse_button_callback(window, button, action, mods):
    global mouse_button

    if action == glfw.PRESS:
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse_button = "left"
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse_button = "right"
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            mouse_button = "middle"

    elif action == glfw.RELEASE:
        mouse_button = None

    # ------------------------------------------
    # LEFT CLICK PICKING
    # ------------------------------------------
    if action == glfw.PRESS and button == glfw.MOUSE_BUTTON_LEFT:

        x, y = glfw.get_cursor_pos(window)

        aspect = win.width / win.height
        projection = camera.get_projection_matrix(aspect)
        view = camera.get_view_matrix()

        result = pick_at_screen(
            x=x, y=y,
            width=win.width, height=win.height,
            projection=projection, view=view,
            tile_size=board.tile_size,
            pieces=pieces.board_state.items(),
            piece_sphere_radius=0.35
        )

        # Prioritize piece selection
        if result["piece"] is not None:
            (f, r, ptype, color), _ = result["piece"]
            pieces.set_selected_piece(f, r)
            board.set_selected_tile(f, r)

            print(f"[Pick] Piece selected: {ptype} {color} at ({f}, {r})")

        elif result["tile"] is not None:
            f, r = result["tile"]
            board.set_selected_tile(f, r)
            pieces.set_selected_piece(f, r)

            print(f"[Pick] Tile selected: {f}, {r}")

        else:
            board.set_selected_tile(None)
            pieces.set_selected_piece(None)
            print("[Pick] Outside board")


def cursor_pos_callback(window, xpos, ypos):
    global mouse_last_x, mouse_last_y, mouse_first

    if mouse_first:
        mouse_last_x = xpos
        mouse_last_y = ypos
        mouse_first = False

    dx = xpos - mouse_last_x
    dy = ypos - mouse_last_y

    mouse_last_x = xpos
    mouse_last_y = ypos

    # Orbit or pan camera
    if mouse_button:
        camera.process_mouse_move(dx, dy, button=mouse_button)


def scroll_callback(window, xoffset, yoffset):
    camera.process_scroll(yoffset)


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------
def main():
    global camera, camera_mgr, win, board, pieces, animation

    # -----------------------------
    # WINDOW
    # -----------------------------
    win = Window(1280, 720, "Chess3D Engine — Picking + Animation")
    win.init()

    win.set_key_callback(key_callback)
    win.set_mouse_button_callback(mouse_button_callback)
    win.set_cursor_pos_callback(cursor_pos_callback)
    win.set_scroll_callback(scroll_callback)

    # -----------------------------
    # SHADER
    # -----------------------------
    shader = Shader(
        "assets/shaders/vertex_shader.glsl",
        "assets/shaders/fragment_shader.glsl"
    )

    # -----------------------------
    # CAMERA + MANAGER
    # -----------------------------
    camera = OrbitCamera(
        mode="perspective",
        eye=np.array([7.0, 10.0, 7.0], dtype=np.float32),
        target=np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    camera_mgr = CameraManager(camera)

    # -----------------------------
    # BOARD
    # -----------------------------
    from renderer.model_loader import load_texture
    light_tex = load_texture("assets/textures/light.jpg")
    dark_tex = load_texture("assets/textures/dark.jpg")

    board = BoardRenderer(shader, light_tex, dark_tex)

    # -----------------------------
    # PIECES
    # -----------------------------
    pieces = PieceRenderer(shader)

    # Simple initial setup
    for f in range(8):
        pieces.set_piece(f, 1, "pawn", "white")
        pieces.set_piece(f, 6, "pawn", "black")

    pieces.set_piece(4, 0, "king", "white")
    pieces.set_piece(4, 7, "king", "black")
    pieces.set_piece(3, 0, "queen", "white")
    pieces.set_piece(3, 7, "queen", "black")

    # -----------------------------
    # ANIMATION SYSTEM
    # -----------------------------
    animation = AnimationManager(piece_renderer=pieces, tile_size=board.tile_size)

    # Example animation start (REMOVE once tested)
    # animation.move_piece((4,1), (4,3), 0.8)
    # animation.capture_piece((3,7), 0.6)
    # animation.spawn_piece((0,2), "rook", "white", 0.6)

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    while not win.should_close():

        win.update_time()
        dt = win.delta_time

        # Camera mode switching
        if glfw.KEY_1 in keys_down:
            camera_mgr.to_mode("perspective", duration=0.7)
        if glfw.KEY_2 in keys_down:
            camera_mgr.to_mode("orthographic", duration=0.7)

        # Update camera & animations
        camera_mgr.update(dt)
        animation.update(dt)

        # Clear
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Matrices
        view = camera.get_view_matrix()
        projection = camera.get_projection_matrix(win.width / win.height)

        # Draw board and static pieces
        board.draw(view, projection)
        pieces.draw(view, projection)

        # Draw animated pieces OVER the static board/pieces
        entries = animation.get_animated_entries()
        for e in entries:
            shader.use()
            shader.set_mat4("model", e["model"])
            shader.set_float("alpha", e["alpha"])

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, e["texture"])
            shader.set_int("texture0", 0)

            e["mesh"].draw()

            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            shader.stop()

        # End frame
        win.poll_events()
        win.swap_buffers()

    win.terminate()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
