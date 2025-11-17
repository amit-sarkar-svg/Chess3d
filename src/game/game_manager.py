"""
src/game/game_manager.py

High-level game manager for Chess3D.

Responsibilities:
- Synchronize 3D board with chess_logic backend
- Handle tile/piece selection from picking system
- Generate legal moves
- Validate moves
- Trigger animations (move, capture, promotion)
- Update 3D piece renderer after logic updates
"""

from __future__ import annotations
from typing import Optional, Tuple

from .chess_logic import Board, Move
from renderer.piece_renderer import PieceRenderer
from renderer.board_renderer import BoardRenderer
from game.animation import AnimationManager


Square = Tuple[int, int]  # (file, rank)


class GameManager:
    def __init__(self,
                 board_logic: Board,
                 piece_renderer: PieceRenderer,
                 board_renderer: BoardRenderer,
                 animation_manager: AnimationManager):

        self.logic: Board = board_logic
        self.pieces: PieceRenderer = piece_renderer
        self.board: BoardRenderer = board_renderer
        self.anim: AnimationManager = animation_manager

        # UI selection
        self.selected_square: Optional[Square] = None
        self.legal_moves_cache = []  # List[Move]

        # Sync model renderer with logic board
        self._sync_full_board()

    # -------------------------------------------------------------------------
    # Board → Renderer Sync
    # -------------------------------------------------------------------------
    def _sync_full_board(self):
        """Update all pieces in 3D renderer to match logic board."""
        self.pieces.board_state.clear()

        for r in range(8):
            for c in range(8):
                ch = self.logic.board[r][c]
                if ch == '.':
                    continue

                piece_type = self._map_piece_type(ch)
                color = "white" if ch.isupper() else "black"

                file = c
                rank = 7 - r

                self.pieces.set_piece(file, rank, piece_type, color)

    def _sync_single_move(self, move: Move):
        """Update only moved and/or captured piece in renderer."""
        sr, sc = move.src
        dr, dc = move.dst

        file_src = sc
        rank_src = 7 - sr
        file_dst = dc
        rank_dst = 7 - dr

        piece_type = self._map_piece_type(move.piece)
        color = "white" if move.piece.isupper() else "black"

        # Remove at source
        self.pieces.remove_piece(file_src, rank_src)

        # If capture
        if move.capture and not move.is_en_passant:
            self.pieces.remove_piece(file_dst, rank_dst)

        # En-passant capture
        if move.is_en_passant:
            cap_file = dc
            cap_rank = 7 - sr
            self.pieces.remove_piece(cap_file, cap_rank)

        # Add moved piece (post-promotion if needed)
        if move.promotion:
            piece_type = self._map_piece_type(move.promotion)

        self.pieces.set_piece(file_dst, rank_dst, piece_type, color)

    def _map_piece_type(self, ch: str) -> str:
        """Map logic engine char → model name."""
        return {
            'P': 'pawn', 'p': 'pawn',
            'N': 'knight', 'n': 'knight',
            'B': 'bishop', 'b': 'bishop',
            'R': 'rook', 'r': 'rook',
            'Q': 'queen', 'q': 'queen',
            'K': 'king',  'k': 'king'
        }[ch]

    # -------------------------------------------------------------------------
    # Input Handling
    # -------------------------------------------------------------------------
    def handle_tile_click(self, file: int, rank: int):
        """Called from main.py when a tile is left-clicked."""
        board_r = 7 - rank
        board_c = file
        clicked_square = (board_r, board_c)

        piece = self.logic.piece_at(clicked_square)

        # --------------------------
        # 1) If nothing selected yet
        # --------------------------
        if self.selected_square is None:
            if piece is None:
                return  # clicked empty tile

            # Must pick your own color
            if self.logic.white_to_move and piece.islower():
                return
            if not self.logic.white_to_move and piece.isupper():
                return

            # Select piece
            self.selected_square = clicked_square
            self.board.set_selected_tile(file, rank)
            self.pieces.set_selected_piece(file, rank)

            # Compute moves for this piece
            self.legal_moves_cache = [
                m for m in self.logic.generate_legal_moves()
                if m.src == clicked_square
            ]

            return

        # ------------------------------------------------------
        # 2) If a piece was selected and you clicked another tile
        # ------------------------------------------------------
        src = self.selected_square

        # Is this a legal move?
        found_move = None
        for mv in self.legal_moves_cache:
            if mv.dst == clicked_square:
                found_move = mv
                break

        if found_move:
            self._play_move(found_move)
            return

        # Else: clicked invalid square → re-select piece if same color
        if piece and (
            (self.logic.white_to_move and piece.isupper()) or
            (not self.logic.white_to_move and piece.islower())
        ):
            # re-select
            self.selected_square = clicked_square
            self.board.set_selected_tile(file, rank)
            self.pieces.set_selected_piece(file, rank)

            self.legal_moves_cache = [
                m for m in self.logic.generate_legal_moves()
                if m.src == clicked_square
            ]
        else:
            # clear selection
            self.clear_selection()

    # -------------------------------------------------------------------------
    # Move Execution + Animation
    # -------------------------------------------------------------------------
    def _play_move(self, move: Move):
        """Called only for legal moves."""
        self.clear_selection()

        # Trigger animations BEFORE logic updates (so we know exact positions)
        self._trigger_animation(move)

        # Apply logic move
        self.logic.make_move(move)

        # Sync 3D board → match new logic board
        self._sync_single_move(move)

    def _trigger_animation(self, move: Move):
        sr, sc = move.src
        dr, dc = move.dst

        file_src = sc
        rank_src = 7 - sr
        file_dst = dc
        rank_dst = 7 - dr

        # Capture animation first
        if move.capture:
            # If en passant, captured piece is behind dst
            if move.is_en_passant:
                cap_file = dc
                cap_rank = 7 - sr
                self.anim.capture_piece((cap_file, cap_rank), duration=0.4)
            else:
                self.anim.capture_piece((file_dst, rank_dst), duration=0.5)

        # Movement animation
        self.anim.move_piece(
            (file_src, rank_src),
            (file_dst, rank_dst),
            duration=0.6
        )

        # Promotion animation (spawn)
        if move.promotion:
            promoted_type = self._map_piece_type(move.promotion)
            color = "white" if move.piece.isupper() else "black"
            self.anim.spawn_piece((file_dst, rank_dst), promoted_type, color, duration=0.4)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def clear_selection(self):
        self.selected_square = None
        self.board.set_selected_tile(None)
        self.pieces.set_selected_piece(None)
        self.legal_moves_cache = []

    # -------------------------------------------------------------------------
    # Debug / info
    # -------------------------------------------------------------------------
    def print_state(self):
        print("--- Game Manager State ---")
        print("Selected:", self.selected_square)
        print("White to move:", self.logic.white_to_move)
        print("Legal moves:", [m.uci() for m in self.legal_moves_cache])
        print("--------------------------")
