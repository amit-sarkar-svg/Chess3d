"""
src/game/chess_logic.py

A production-ready chess logic engine (move generation + legality checks).

Features:
- 8x8 board (row 0 = rank 8, row 7 = rank 1) — use board[row][col]
- Piece encoding: 'P','N','B','R','Q','K' for white; lowercase for black
- Castling rights tracked (KQkq)
- En-passant target square stored as (r,c) or None
- Halfmove clock & fullmove number
- Move application + undo (preserves state)
- Legal move generation (filters out moves that leave king in check)
- Check / checkmate / stalemate detection

API:
    board = Board()
    moves = board.generate_legal_moves()          # list of Move
    mv = moves[0]
    board.make_move(mv)                           # applies the move
    board.undo_move()                             # undo last move
    board.is_checkmate(), board.is_stalemate()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import copy

# Types
Square = Tuple[int, int]  # (row, col) where 0 <= row,col <=7; row0 = rank8


# --------------------------------------------
# Move object
# --------------------------------------------
@dataclass
class Move:
    src: Square
    dst: Square
    piece: str
    capture: Optional[str] = None
    promotion: Optional[str] = None  # 'q','r','b','n' (lowercase for black promotions)
    is_castle: bool = False
    is_en_passant: bool = False

    def uci(self) -> str:
        """Return simple UCI-style string (e2e4, e7e8q)."""
        def sq2(sq):
            r, c = sq
            file = chr(ord('a') + c)
            rank = str(8 - r)
            return file + rank
        s = sq2(self.src) + sq2(self.dst)
        if self.promotion:
            s += self.promotion.lower()
        return s

    def __str__(self):
        return self.uci()


# --------------------------------------------
# Board class
# --------------------------------------------
class Board:
    def __init__(self):
        self.reset()

    def reset(self):
        """Set starting position."""
        self.board = [
            list("rnbqkbnr"),
            list("pppppppp"),
            list("........"),
            list("........"),
            list("........"),
            list("........"),
            list("PPPPPPPP"),
            list("RNBQKBNR"),
        ]
        # Castling rights string: subset of 'KQkq'
        self.castling = "KQkq"
        # en-passant target square (row,col) after a double pawn move, else None
        self.ep_target: Optional[Square] = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.white_to_move = True

        # Move stack for undo: list of dicts capturing previous state
        self._stack: List[dict] = []

    # ----------------------
    # Helpers
    # ----------------------
    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def piece_at(self, sq: Square) -> Optional[str]:
        r, c = sq
        ch = self.board[r][c]
        return None if ch == '.' else ch

    def is_white_piece(self, ch: str) -> bool:
        return ch.isupper()

    def is_black_piece(self, ch: str) -> bool:
        return ch.islower()

    def ally_color(self):
        return 'white' if self.white_to_move else 'black'

    def enemy_color(self):
        return 'black' if self.white_to_move else 'white'

    # ----------------------
    # Move generation (legal)
    # ----------------------
    def generate_legal_moves(self) -> List[Move]:
        """Generate all legal moves for the side to move."""
        moves = self._generate_pseudo_legal_moves()
        legal = []
        for m in moves:
            self._make_move_internal(m)
            king_in_check = self._is_king_in_check(not self.white_to_move)
            self._undo_move_internal()
            if not king_in_check:
                legal.append(m)
        return legal

    # ----------------------
    # Pseudo-legal moves (may leave king in check)
    # ----------------------
    def _generate_pseudo_legal_moves(self) -> List[Move]:
        moves: List[Move] = []
        for r in range(8):
            for c in range(8):
                ch = self.board[r][c]
                if ch == '.':
                    continue
                if self.white_to_move and not ch.isupper():
                    continue
                if not self.white_to_move and not ch.islower():
                    continue
                piece = ch.upper()
                if piece == 'P':
                    moves.extend(self._pawn_moves((r, c), ch))
                elif piece == 'N':
                    moves.extend(self._knight_moves((r, c), ch))
                elif piece == 'B':
                    moves.extend(self._sliding_moves((r, c), ch, directions=[(-1,-1),(-1,1),(1,-1),(1,1)]))
                elif piece == 'R':
                    moves.extend(self._sliding_moves((r, c), ch, directions=[(-1,0),(1,0),(0,-1),(0,1)]))
                elif piece == 'Q':
                    moves.extend(self._sliding_moves((r, c), ch, directions=[(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]))
                elif piece == 'K':
                    moves.extend(self._king_moves((r, c), ch))
        return moves

    # ----------------------
    # Pawn moves (including en-passant & promotion)
    # ----------------------
    def _pawn_moves(self, sq: Square, ch: str) -> List[Move]:
        r, c = sq
        moves: List[Move] = []
        dir_forward = -1 if ch.isupper() else 1  # white moves up (decreasing row)
        start_row = 6 if ch.isupper() else 1
        enemy_check = (lambda ch2: ch2 != '.' and (ch2.islower() if ch.isupper() else ch2.isupper()))

        # Single step
        nr = r + dir_forward
        if self.in_bounds(nr, c) and self.board[nr][c] == '.':
            # promotion check
            if nr == 0 or nr == 7:
                for promo in ['Q','R','B','N']:
                    promo_ch = promo if ch.isupper() else promo.lower()
                    moves.append(Move((r,c),(nr,c),ch,promotion=promo_ch))
            else:
                moves.append(Move((r,c),(nr,c),ch))
            # Double step
            if r == start_row:
                nr2 = r + 2*dir_forward
                if self.in_bounds(nr2, c) and self.board[nr2][c] == '.':
                    moves.append(Move((r,c),(nr2,c),ch))
        # Captures
        for dc in (-1, 1):
            nc = c + dc
            nr = r + dir_forward
            if not self.in_bounds(nr, nc):
                continue
            target = self.board[nr][nc]
            if enemy_check(target):
                if nr == 0 or nr == 7:
                    for promo in ['Q','R','B','N']:
                        promo_ch = promo if ch.isupper() else promo.lower()
                        moves.append(Move((r,c),(nr,nc),ch,capture=target,promotion=promo_ch))
                else:
                    moves.append(Move((r,c),(nr,nc),ch,capture=target))
        # En-passant
        if self.ep_target:
            ep_r, ep_c = self.ep_target
            if ep_r == r + dir_forward and abs(ep_c - c) == 1:
                # The pawn that moved two squares is behind ep square
                moves.append(Move((r,c),(ep_r,ep_c),ch,capture=self.board[r][ep_c],is_en_passant=True))
        return moves

    # ----------------------
    # Knight moves
    # ----------------------
    def _knight_moves(self, sq: Square, ch: str) -> List[Move]:
        r,c = sq
        moves = []
        deltas = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
        for dr,dc in deltas:
            nr, nc = r+dr, c+dc
            if not self.in_bounds(nr,nc):
                continue
            target = self.board[nr][nc]
            if target == '.':
                moves.append(Move((r,c),(nr,nc),ch))
            else:
                if (ch.isupper() and target.islower()) or (ch.islower() and target.isupper()):
                    moves.append(Move((r,c),(nr,nc),ch,capture=target))
        return moves

    # ----------------------
    # Sliding pieces (bishop/rook/queen)
    # ----------------------
    def _sliding_moves(self, sq: Square, ch: str, directions: List[Tuple[int,int]]) -> List[Move]:
        r,c = sq
        moves = []
        for dr,dc in directions:
            nr, nc = r+dr, c+dc
            while self.in_bounds(nr,nc):
                target = self.board[nr][nc]
                if target == '.':
                    moves.append(Move((r,c),(nr,nc),ch))
                else:
                    if (ch.isupper() and target.islower()) or (ch.islower() and target.isupper()):
                        moves.append(Move((r,c),(nr,nc),ch,capture=target))
                    break
                nr += dr; nc += dc
        return moves

    # ----------------------
    # King moves including castling
    # ----------------------
    def _king_moves(self, sq: Square, ch: str) -> List[Move]:
        r,c = sq
        moves = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if not self.in_bounds(nr,nc): continue
                target = self.board[nr][nc]
                if target == '.':
                    moves.append(Move((r,c),(nr,nc),ch))
                else:
                    if (ch.isupper() and target.islower()) or (ch.islower() and target.isupper()):
                        moves.append(Move((r,c),(nr,nc),ch,capture=target))
        # Castling
        if ch == 'K' and self.white_to_move:
            # white king at e1 -> row 7, col 4
            if (7,4) == (r,c):
                # king side
                if 'K' in self.castling:
                    if self.board[7][5]=='.' and self.board[7][6]=='.':
                        # can't castle if in check or squares attacked
                        if not self._is_king_in_check(True) and not self._is_square_attacked((7,5),False) and not self._is_square_attacked((7,6),False):
                            moves.append(Move((r,c),(7,6),ch,is_castle=True))
                # queen side
                if 'Q' in self.castling:
                    if self.board[7][1]=='.' and self.board[7][2]=='.' and self.board[7][3]=='.':
                        if not self._is_king_in_check(True) and not self._is_square_attacked((7,3),False) and not self._is_square_attacked((7,2),False):
                            moves.append(Move((r,c),(7,2),ch,is_castle=True))
        if ch == 'k' and not self.white_to_move:
            if (0,4) == (r,c):
                if 'k' in self.castling:
                    if self.board[0][5]=='.' and self.board[0][6]=='.':
                        if not self._is_king_in_check(False) and not self._is_square_attacked((0,5),True) and not self._is_square_attacked((0,6),True):
                            moves.append(Move((r,c),(0,6),ch,is_castle=True))
                if 'q' in self.castling:
                    if self.board[0][1]=='.' and self.board[0][2]=='.' and self.board[0][3]=='.':
                        if not self._is_king_in_check(False) and not self._is_square_attacked((0,3),True) and not self._is_square_attacked((0,2),True):
                            moves.append(Move((r,c),(0,2),ch,is_castle=True))
        return moves

    # ----------------------
    # Make / Undo moves (public)
    # ----------------------
    def make_move(self, move: Move) -> None:
        """Apply a legal move (assumes legal). Records state for undo."""
        self._make_move_internal(move)
        # After applying, update clocks and turn
        self.white_to_move = not self.white_to_move
        if self.white_to_move:
            self.fullmove_number += 1
        # halfmove clock handling
        if move.piece.upper() == 'P' or move.capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

    def undo_move(self) -> None:
        """Undo last move."""
        self._undo_move_internal()
        # restore turn and clocks from stack pop restored state

    # ----------------------
    # Internal apply/undo (stack-based)
    # ----------------------
    def _make_move_internal(self, move: Move):
        """Apply move and push previous state onto stack (used for searching/legal testing)."""
        # Save snapshot for undo
        snap = {
            'board_before': None,  # we'll store only changed squares to be lightweight
            'castling': self.castling,
            'ep_target': self.ep_target,
            'halfmove_clock': self.halfmove_clock,
            'fullmove_number': self.fullmove_number,
            'white_to_move': self.white_to_move,
            'move': move
        }

        sr, sc = move.src
        dr, dc = move.dst
        piece = self.board[sr][sc]
        target = self.board[dr][dc]

        # We'll store the minimal changes: src, dst, and (for en-passant) captured pawn square
        snap['board_before'] = [(sr,sc,piece),(dr,dc,target)]
        # Handle en-passant captured pawn (the pawn being captured is behind dst)
        ep_captured = None
        if move.is_en_passant:
            # captured pawn is on same file as dst, but on source row
            cap_r = sr
            cap_c = dc
            ep_captured = (cap_r, cap_c, self.board[cap_r][cap_c])
            snap['board_before'].append(ep_captured)
            # remove the pawn
            self.board[cap_r][cap_c] = '.'
        # Execute move
        self.board[sr][sc] = '.'
        # Promotion
        if move.promotion:
            self.board[dr][dc] = move.promotion
        else:
            self.board[dr][dc] = piece

        # Castling: move rook as well
        if move.is_castle:
            # White king-side: king e1->g1 (7,4)->(7,6) => rook h1(7,7)->f1(7,5)
            if piece == 'K' and (sr,sc)==(7,4) and (dr,dc)==(7,6):
                self.board[7][7], self.board[7][5] = '.', 'R'
                snap['board_before'].append((7,7,'R')); snap['board_before'].append((7,5,'.'))
            # White queen-side: e1->c1 rook a1->d1
            if piece == 'K' and (sr,sc)==(7,4) and (dr,dc)==(7,2):
                self.board[7][0], self.board[7][3] = '.', 'R'
                snap['board_before'].append((7,0,'R')); snap['board_before'].append((7,3,'.'))
            # Black king-side
            if piece == 'k' and (sr,sc)==(0,4) and (dr,dc)==(0,6):
                self.board[0][7], self.board[0][5] = '.', 'r'
                snap['board_before'].append((0,7,'r')); snap['board_before'].append((0,5,'.'))
            # Black queen-side
            if piece == 'k' and (sr,sc)==(0,4) and (dr,dc)==(0,2):
                self.board[0][0], self.board[0][3] = '.', 'r'
                snap['board_before'].append((0,0,'r')); snap['board_before'].append((0,3,'.'))

        # Update castling rights if king or rook moved or rook captured
        def remove_castling_rights_for_square(sq):
            nonlocal snap
            mapping = {
                (7,4): 'KQ',  # white king moved => remove K and Q
                (0,4): 'kq',  # black king moved
                (7,7): 'K',   # white rook h1 affects K
                (7,0): 'Q',   # white rook a1 affects Q
                (0,7): 'k',
                (0,0): 'q'
            }
            if sq in mapping:
                for ch in mapping[sq]:
                    if ch in self.castling:
                        self.castling = self.castling.replace(ch, '')

        # remove rights for moved pieces' squares
        remove_castling_rights_for_square((sr,sc))
        remove_castling_rights_for_square((dr,dc))
        # Also if we captured a rook on its original square, remove corresponding right
        if move.capture:
            # captured piece target square was (dr,dc)
            remove_castling_rights_for_square((dr,dc))

        # En-passant target update: if pawn moved two squares, set ep target
        self.ep_target = None
        if piece.upper() == 'P' and abs(dr - sr) == 2:
            # target square is the square behind the pawn's landing square
            ep_r = (sr + dr) // 2
            self.ep_target = (ep_r, sc)

        # Save snap and push
        self._stack.append(snap)

    def _undo_move_internal(self):
        """Undo last _make_move_internal operation restoring state from stack."""
        if not self._stack:
            raise RuntimeError("No moves to undo")
        snap = self._stack.pop()
        move = snap['move']
        # Restore fields
        self.castling = snap['castling']
        self.ep_target = snap['ep_target']
        self.halfmove_clock = snap['halfmove_clock']
        self.fullmove_number = snap['fullmove_number']
        self.white_to_move = snap['white_to_move']
        # Restore board squares saved
        # We saved a list of tuples (r,c,piece)
        for r,c,piece in snap['board_before']:
            self.board[r][c] = piece if piece is not None else '.'

    # ----------------------
    # Attack detection
    # ----------------------
    def _is_square_attacked(self, sq: Square, by_white: bool) -> bool:
        """Return True if square is attacked by side `by_white`."""
        r,c = sq
        # Pawn attacks
        if by_white:
            pawn_dirs = [(-1,-1),(-1,1)]
            for dr,dc in pawn_dirs:
                rr, cc = r - dr, c - dc  # reverse since pawns attack from their perspective
                # Instead we search from attacker squares: white pawns at (r+1, c+-1)
            # simpler: check white pawns located at r+1,c+-1
            for dr,dc in [(1,-1),(1,1)]:
                rr,cc = r+dr,c+dc
                if self.in_bounds(rr,cc) and self.board[rr][cc] == 'P':
                    return True
        else:
            for dr,dc in [(-1,-1),(-1,1)]:
                rr,cc = r+dr,c+dc
                if self.in_bounds(rr,cc) and self.board[rr][cc] == 'p':
                    return True

        # Knights
        for dr,dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
            rr,cc = r+dr,c+dc
            if not self.in_bounds(rr,cc): continue
            ch = self.board[rr][cc]
            if by_white and ch == 'N': return True
            if not by_white and ch == 'n': return True

        # Sliding pieces
        # bishops / queens (diagonals)
        for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr,cc = r+dr,c+dc
            while self.in_bounds(rr,cc):
                ch = self.board[rr][cc]
                if ch != '.':
                    if by_white and ch in ('B','Q'): return True
                    if not by_white and ch in ('b','q'): return True
                    break
                rr += dr; cc += dc

        # rooks / queens (orthogonal)
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr,cc = r+dr,c+dc
            while self.in_bounds(rr,cc):
                ch = self.board[rr][cc]
                if ch != '.':
                    if by_white and ch in ('R','Q'): return True
                    if not by_white and ch in ('r','q'): return True
                    break
                rr += dr; cc += dc

        # King adjacency
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                rr,cc = r+dr,c+dc
                if not self.in_bounds(rr,cc): continue
                ch = self.board[rr][cc]
                if by_white and ch == 'K': return True
                if not by_white and ch == 'k': return True

        return False

    def _is_king_in_check(self, white_king: bool) -> bool:
        """Return True if the specified side's king is in check."""
        # find king position
        king_char = 'K' if white_king else 'k'
        pos = None
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == king_char:
                    pos = (r,c); break
            if pos: break
        if not pos:
            # no king (shouldn't happen) -> consider in check
            return True
        return self._is_square_attacked(pos, by_white=not white_king)

    # ----------------------
    # Game state queries
    # ----------------------
    def is_check(self) -> bool:
        return self._is_king_in_check(self.white_to_move)

    def is_checkmate(self) -> bool:
        if not self.is_check():
            return False
        return len(self.generate_legal_moves()) == 0

    def is_stalemate(self) -> bool:
        if self.is_check():
            return False
        return len(self.generate_legal_moves()) == 0

    # ----------------------
    # Utility / debug
    # ----------------------
    def print_board(self):
        for row in self.board:
            print(''.join(row))
        print(f"Turn: {'white' if self.white_to_move else 'black'}, castling: {self.castling}, ep: {self.ep_target}, hm:{self.halfmove_clock}, fm:{self.fullmove_number}")

    def fen(self) -> str:
        """Return a simple FEN (partial) representation for debug (not fully spec-compliant)."""
        rows = []
        for r in self.board:
            empty = 0
            srow = ''
            for ch in r:
                if ch == '.':
                    empty += 1
                else:
                    if empty:
                        srow += str(empty); empty = 0
                    srow += ch
            if empty:
                srow += str(empty)
            rows.append(srow)
        side = 'w' if self.white_to_move else 'b'
        cast = self.castling if self.castling else '-'
        ep = '-' if not self.ep_target else (chr(ord('a')+self.ep_target[1]) + str(8-self.ep_target[0]))
        return ' '.join([ '/'.join(rows), side, cast, ep, str(self.halfmove_clock), str(self.fullmove_number) ])
