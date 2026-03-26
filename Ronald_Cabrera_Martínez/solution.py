import math
import random
import time
from collections import deque
from typing import Optional

from board import HexBoard
from player import Player

PATTERN_SELF = "P"
PATTERN_OPP = "O"
PATTERN_EMPTY = "E"

LOCAL_PATTERN_PRIORS: dict[str, list[tuple[float, tuple[tuple[tuple[int, int], str], ...]]]] = {
    "strong_bridge": [
        (5.0, (((-1, 0), PATTERN_SELF), ((0, 1), PATTERN_SELF))),
        (5.0, (((0, 1), PATTERN_SELF), ((1, 0), PATTERN_SELF))),
        (5.0, (((-1, -1), PATTERN_SELF), ((1, 0), PATTERN_SELF))),
        (5.0, (((0, -1), PATTERN_SELF), ((1, 1), PATTERN_SELF))),
        (5.0, (((-1, 1), PATTERN_SELF), ((0, -1), PATTERN_SELF))),
        (5.0, (((-1, 0), PATTERN_SELF), ((1, -1), PATTERN_SELF))),
    ],
    "weak_bridge": [
        (2.5, (((-1, 0), PATTERN_SELF), ((0, 1), PATTERN_EMPTY))),
        (2.5, (((0, 1), PATTERN_SELF), ((1, 0), PATTERN_EMPTY))),
        (2.5, (((-1, -1), PATTERN_SELF), ((1, 0), PATTERN_EMPTY))),
        (2.5, (((0, -1), PATTERN_SELF), ((1, 1), PATTERN_EMPTY))),
        (2.5, (((-1, 1), PATTERN_SELF), ((0, -1), PATTERN_EMPTY))),
        (2.5, (((-1, 0), PATTERN_SELF), ((1, -1), PATTERN_EMPTY))),
    ],
    "cut": [
        (3.5, (((-1, 0), PATTERN_OPP), ((0, 1), PATTERN_OPP))),
        (3.5, (((0, 1), PATTERN_OPP), ((1, 0), PATTERN_OPP))),
        (3.5, (((-1, -1), PATTERN_OPP), ((1, 0), PATTERN_OPP))),
        (3.5, (((0, -1), PATTERN_OPP), ((1, 1), PATTERN_OPP))),
    ],
    "critical_connection": [
        (4.5, (((-1, 0), PATTERN_SELF), ((1, 0), PATTERN_SELF))),
        (4.5, (((0, -1), PATTERN_SELF), ((0, 1), PATTERN_SELF))),
        (4.5, (((-1, -1), PATTERN_SELF), ((1, 1), PATTERN_SELF))),
    ],
    "safe_jump": [
        (3.0, (((-2, 0), PATTERN_SELF), ((-1, 0), PATTERN_EMPTY))),
        (3.0, (((0, 2), PATTERN_SELF), ((0, 1), PATTERN_EMPTY))),
        (3.0, (((2, 0), PATTERN_SELF), ((1, 0), PATTERN_EMPTY))),
        (3.0, (((0, -2), PATTERN_SELF), ((0, -1), PATTERN_EMPTY))),
        (3.0, (((-2, -1), PATTERN_SELF), ((-1, -1), PATTERN_EMPTY))),
        (3.0, (((2, 1), PATTERN_SELF), ((1, 1), PATTERN_EMPTY))),
    ],
}

class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent: list[int] = list(range(size))
        self.rank: list[int] = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)


def get_legal_moves(board: HexBoard) -> list[tuple[int, int]]:
    matrix = board.board
    legal_moves: list[tuple[int, int]] = []
    for row_index, row in enumerate(matrix):
        for col_index, cell in enumerate(row):
            if cell == 0:
                legal_moves.append((row_index, col_index))
    return legal_moves


def opponent(player_id: int) -> int:
    return 2 if player_id == 1 else 1


def get_hex_neighbors(row: int, col: int, board_size: int) -> list[tuple[int, int]]:
    if row % 2 == 0:
        deltas = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    else:
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

    neighbors: list[tuple[int, int]] = []
    for delta_row, delta_col in deltas:
        neighbor_row = row + delta_row
        neighbor_col = col + delta_col
        if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size:
            neighbors.append((neighbor_row, neighbor_col))
    return neighbors


def are_neighbors(a: tuple[int, int], b: tuple[int, int], board_size: int) -> bool:
    return b in get_hex_neighbors(a[0], a[1], board_size)


def matrix_legal_moves(board_matrix: list[list[int]]) -> list[tuple[int, int]]:
    moves: list[tuple[int, int]] = []
    for row_index, row in enumerate(board_matrix):
        for col_index, value in enumerate(row):
            if value == 0:
                moves.append((row_index, col_index))
    return moves


def check_connection_matrix(board_matrix: list[list[int]], player_id: int) -> bool:
    board_size = len(board_matrix)
    if board_size == 0:
        return False

    queue: deque[tuple[int, int]] = deque()
    visited: set[tuple[int, int]] = set()

    if player_id == 1:
        for row in range(board_size):
            if board_matrix[row][0] == player_id:
                queue.append((row, 0))
                visited.add((row, 0))
        target_col = board_size - 1
        while queue:
            row, col = queue.popleft()
            if col == target_col:
                return True
            for n_row, n_col in get_hex_neighbors(row, col, board_size):
                if (n_row, n_col) not in visited and board_matrix[n_row][n_col] == player_id:
                    visited.add((n_row, n_col))
                    queue.append((n_row, n_col))
        return False

    for col in range(board_size):
        if board_matrix[0][col] == player_id:
            queue.append((0, col))
            visited.add((0, col))
    target_row = board_size - 1
    while queue:
        row, col = queue.popleft()
        if row == target_row:
            return True
        for n_row, n_col in get_hex_neighbors(row, col, board_size):
            if (n_row, n_col) not in visited and board_matrix[n_row][n_col] == player_id:
                visited.add((n_row, n_col))
                queue.append((n_row, n_col))
    return False


def two_distance_cost(board_matrix: list[list[int]], player_id: int) -> int:
    board_size = len(board_matrix)
    if board_size == 0:
        return 10**9

    inf_cost = 10**9
    dist = [[inf_cost for _ in range(board_size)] for _ in range(board_size)]
    dq: deque[tuple[int, int]] = deque()
    enemy = opponent(player_id)

    if player_id == 1:
        for row in range(board_size):
            value = board_matrix[row][0]
            if value == enemy:
                continue
            start_cost = 0 if value == player_id else 1
            dist[row][0] = start_cost
            if start_cost == 0:
                dq.appendleft((row, 0))
            else:
                dq.append((row, 0))

        while dq:
            row, col = dq.popleft()
            current = dist[row][col]
            for n_row, n_col in get_hex_neighbors(row, col, board_size):
                value = board_matrix[n_row][n_col]
                if value == enemy:
                    continue
                edge_cost = 0 if value == player_id else 1
                new_cost = current + edge_cost
                if new_cost < dist[n_row][n_col]:
                    dist[n_row][n_col] = new_cost
                    if edge_cost == 0:
                        dq.appendleft((n_row, n_col))
                    else:
                        dq.append((n_row, n_col))

        best = min(dist[row][board_size - 1] for row in range(board_size))
        return best

    for col in range(board_size):
        value = board_matrix[0][col]
        if value == enemy:
            continue
        start_cost = 0 if value == player_id else 1
        dist[0][col] = start_cost
        if start_cost == 0:
            dq.appendleft((0, col))
        else:
            dq.append((0, col))

    while dq:
        row, col = dq.popleft()
        current = dist[row][col]
        for n_row, n_col in get_hex_neighbors(row, col, board_size):
            value = board_matrix[n_row][n_col]
            if value == enemy:
                continue
            edge_cost = 0 if value == player_id else 1
            new_cost = current + edge_cost
            if new_cost < dist[n_row][n_col]:
                dist[n_row][n_col] = new_cost
                if edge_cost == 0:
                    dq.appendleft((n_row, n_col))
                else:
                    dq.append((n_row, n_col))

    best = min(dist[board_size - 1][col] for col in range(board_size))
    return best


def focused_moves(board_matrix: list[list[int]]) -> list[tuple[int, int]]:
    board_size = len(board_matrix)
    active: set[tuple[int, int]] = set()
    has_piece = False

    for row in range(board_size):
        for col in range(board_size):
            if board_matrix[row][col] != 0:
                has_piece = True
                for n_row, n_col in get_hex_neighbors(row, col, board_size):
                    if board_matrix[n_row][n_col] == 0:
                        active.add((n_row, n_col))

    if not has_piece:
        return matrix_legal_moves(board_matrix)
    if active:
        return list(active)
    return matrix_legal_moves(board_matrix)


def get_relative_cell_state(
    board_matrix: list[list[int]],
    row: int,
    col: int,
    delta_row: int,
    delta_col: int,
    player: int,
    enemy_player: int,
) -> Optional[str]:
    target_row = row + delta_row
    target_col = col + delta_col
    board_size = len(board_matrix)

    if not (0 <= target_row < board_size and 0 <= target_col < board_size):
        return None

    value = board_matrix[target_row][target_col]
    if value == 0:
        return PATTERN_EMPTY
    if value == player:
        return PATTERN_SELF
    if value == enemy_player:
        return PATTERN_OPP
    return None


def match_local_pattern(
    board_matrix: list[list[int]],
    row: int,
    col: int,
    player: int,
    enemy_player: int,
    pattern: tuple[tuple[tuple[int, int], str], ...],
) -> bool:
    for (delta_row, delta_col), expected_state in pattern:
        observed_state = get_relative_cell_state(
            board_matrix,
            row,
            col,
            delta_row,
            delta_col,
            player,
            enemy_player,
        )
        if observed_state != expected_state:
            return False
    return True


def evaluate_local_pattern_priors(board: HexBoard, move: tuple[int, int], player: int) -> float:
    row, col = move
    board_matrix = board.board
    enemy_player = opponent(player)
    pattern_score = 0.0
    attack_matches = 0
    defense_matches = 0

    for pattern_name, weighted_patterns in LOCAL_PATTERN_PRIORS.items():
        for weight, pattern in weighted_patterns:
            if match_local_pattern(board_matrix, row, col, player, enemy_player, pattern):
                pattern_score += weight
                if pattern_name in {"strong_bridge", "weak_bridge", "critical_connection", "safe_jump"}:
                    attack_matches += 1
                if pattern_name == "cut":
                    defense_matches += 1

    if attack_matches >= 2:
        pattern_score += 6.0
    if defense_matches >= 2:
        pattern_score += 3.0

    return pattern_score


def random_move(board: HexBoard) -> tuple[int, int]:
    legal_moves = get_legal_moves(board)
    if not legal_moves:
        raise ValueError("No legal moves available")
    return random.choice(legal_moves)


def get_phase_weights(board: HexBoard) -> tuple[float, float, float, float, float, float, float]:
    board_matrix = board.board
    board_size = getattr(board, "size", len(board_matrix))
    total_cells = board_size * board_size
    empty_cells = sum(1 for row in board_matrix for cell in row if cell == 0)
    played_ratio = (total_cells - empty_cells) / total_cells if total_cells > 0 else 0.0

    if played_ratio < 0.2:
        return 0.35, 2.5, 1.2, 1.1, 0.8, 1.3, 0.8
    if played_ratio < 0.7:
        return 0.2, 3.2, 1.8, 1.0, 1.0, 1.1, 1.0
    return 0.1, 3.8, 2.4, 0.9, 1.2, 0.9, 1.2


def get_immediate_winning_moves_matrix(
    board_matrix: list[list[int]],
    legal_moves: list[tuple[int, int]],
    player_id: int,
) -> list[tuple[int, int]]:
    winning_moves: list[tuple[int, int]] = []
    for row, col in legal_moves:
        if board_matrix[row][col] != 0:
            continue
        board_matrix[row][col] = player_id
        if check_connection_matrix(board_matrix, player_id):
            winning_moves.append((row, col))
        board_matrix[row][col] = 0
    return winning_moves


def find_immediate_winning_move(
    board: HexBoard, legal_moves: list[tuple[int, int]], player_id: int
) -> Optional[tuple[int, int]]:
    board_matrix = [row[:] for row in board.board]
    for row, col in legal_moves:
        if board_matrix[row][col] != 0:
            continue
        board_matrix[row][col] = player_id
        if check_connection_matrix(board_matrix, player_id):
            board_matrix[row][col] = 0
            return (row, col)
        board_matrix[row][col] = 0
    return None


def get_immediate_winning_moves(
    board: HexBoard, legal_moves: list[tuple[int, int]], player_id: int
) -> list[tuple[int, int]]:
    board_matrix = [row[:] for row in board.board]
    return get_immediate_winning_moves_matrix(board_matrix, legal_moves, player_id)


def find_urgent_block_move(
    board: HexBoard, legal_moves: list[tuple[int, int]], player_id: int
) -> Optional[tuple[int, int]]:
    enemy_player = opponent(player_id)
    board_matrix = [row[:] for row in board.board]
    enemy_winning_moves = get_immediate_winning_moves_matrix(board_matrix, legal_moves, enemy_player)

    if not enemy_winning_moves:
        return None

    if len(enemy_winning_moves) == 1:
        return enemy_winning_moves[0]

    enemy_winning_set = set(enemy_winning_moves)
    candidates = [move for move in legal_moves if move in enemy_winning_set]
    if not candidates:
        return None

    best_move: Optional[tuple[int, int]] = None
    lowest_threat_count = float("inf")

    for block_move in candidates:
        if board_matrix[block_move[0]][block_move[1]] != 0:
            continue
        board_matrix[block_move[0]][block_move[1]] = player_id
        remaining_legal_moves = [move for move in legal_moves if move != block_move]
        remaining_enemy_wins = get_immediate_winning_moves_matrix(
            board_matrix,
            remaining_legal_moves,
            enemy_player,
        )
        threat_count = len(remaining_enemy_wins)
        board_matrix[block_move[0]][block_move[1]] = 0

        if threat_count < lowest_threat_count:
            lowest_threat_count = threat_count
            best_move = block_move
            if threat_count == 0:
                break

    return best_move


def find_forcing_two_ply_move(
    board: HexBoard, legal_moves: list[tuple[int, int]], player_id: int, max_candidates: int = 8
) -> Optional[tuple[int, int]]:
    if not legal_moves:
        return None

    board_matrix = [row[:] for row in board.board]
    ordered_candidates = order_moves(board, legal_moves, player_id)[:max_candidates]
    enemy_player = opponent(player_id)

    for candidate_move in ordered_candidates:
        c_row, c_col = candidate_move
        if board_matrix[c_row][c_col] != 0:
            continue
        board_matrix[c_row][c_col] = player_id

        if check_connection_matrix(board_matrix, player_id):
            board_matrix[c_row][c_col] = 0
            return candidate_move

        enemy_replies = [move for move in legal_moves if move != candidate_move]
        forced = True

        for enemy_move in enemy_replies:
            e_row, e_col = enemy_move
            if board_matrix[e_row][e_col] != 0:
                continue
            board_matrix[e_row][e_col] = enemy_player
            our_replies = [move for move in enemy_replies if move != enemy_move]
            our_immediate_wins = get_immediate_winning_moves_matrix(board_matrix, our_replies, player_id)
            board_matrix[e_row][e_col] = 0
            if not our_immediate_wins:
                forced = False
                break

        board_matrix[c_row][c_col] = 0
        if forced:
            return candidate_move

    return None


def score_move(board: HexBoard, move: tuple[int, int], player: int) -> float:
    row, col = move
    board_matrix = board.board
    board_size = getattr(board, "size", len(board_matrix))
    center = (board_size - 1) / 2.0

    distance_to_center_sq = (row - center) * (row - center) + (col - center) * (col - center)
    center_score = -distance_to_center_sq

    player_adjacent = 0
    opponent_adjacent = 0
    enemy_player = opponent(player)
    neighbors = get_hex_neighbors(row, col, board_size)
    player_neighbor_cells: list[tuple[int, int]] = []
    enemy_neighbor_cells: list[tuple[int, int]] = []

    for neighbor_row, neighbor_col in neighbors:
        neighbor_value = board_matrix[neighbor_row][neighbor_col]
        if neighbor_value == player:
            player_adjacent += 1
            player_neighbor_cells.append((neighbor_row, neighbor_col))
        elif neighbor_value == enemy_player:
            opponent_adjacent += 1
            enemy_neighbor_cells.append((neighbor_row, neighbor_col))

    friendly_connection_bonus = 4.0 if player_adjacent >= 2 else 0.0
    blocking_bonus = 2.0 if opponent_adjacent >= 2 else 0.0
    bridge_bonus = 0.0
    block_bridge_bonus = 0.0
    pattern_prior_score = evaluate_local_pattern_priors(board, move, player)
    (
        center_weight,
        friendly_weight,
        blocking_weight,
        friendly_connection_weight,
        blocking_bonus_weight,
        bridge_weight,
        block_bridge_weight,
    ) = get_phase_weights(board)

    for index_a in range(len(player_neighbor_cells)):
        for index_b in range(index_a + 1, len(player_neighbor_cells)):
            if not are_neighbors(player_neighbor_cells[index_a], player_neighbor_cells[index_b], board_size):
                bridge_bonus = 5.0
                break
        if bridge_bonus > 0.0:
            break

    for index_a in range(len(enemy_neighbor_cells)):
        for index_b in range(index_a + 1, len(enemy_neighbor_cells)):
            if not are_neighbors(enemy_neighbor_cells[index_a], enemy_neighbor_cells[index_b], board_size):
                block_bridge_bonus = 2.5
                break
        if block_bridge_bonus > 0.0:
            break

    return (
        (friendly_weight * player_adjacent)
        + (blocking_weight * opponent_adjacent)
        + (center_weight * center_score)
        + (friendly_connection_weight * friendly_connection_bonus)
        + (blocking_bonus_weight * blocking_bonus)
        + (bridge_weight * bridge_bonus)
        + (block_bridge_weight * block_bridge_bonus)
        + pattern_prior_score
    )


def order_moves(board: HexBoard, moves: list[tuple[int, int]], player: int) -> list[tuple[int, int]]:
    return sorted(moves, key=lambda move: score_move(board, move, player), reverse=True)


def heuristic_rollout_move(
    board: HexBoard, player: int, legal_moves: list[tuple[int, int]]
) -> tuple[int, int]:
    if not legal_moves:
        raise ValueError("No legal moves available")

    best_score: Optional[float] = None
    best_moves: list[tuple[int, int]] = []
    all_scores_equal = True
    first_score: Optional[float] = None

    for row, col in legal_moves:
        score = score_move(board, (row, col), player)

        if first_score is None:
            first_score = score
        elif score != first_score:
            all_scores_equal = False

        if best_score is None or score > best_score:
            best_score = score
            best_moves = [(row, col)]
        elif score == best_score:
            best_moves.append((row, col))

    if all_scores_equal:
        return random.choice(legal_moves)

    return random.choice(best_moves)


class Node:
    def __init__(
        self,
        board: HexBoard,
        player: int,
        parent: Optional["Node"] = None,
        move: Optional[tuple[int, int]] = None,
    ) -> None:
        self.board: HexBoard = board
        self.player: int = player
        self.parent: Optional[Node] = parent
        self.move: Optional[tuple[int, int]] = move
        self.children: list[Node] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.rave_wins: float = 0.0
        self.rave_visits: int = 0
        self.untried_moves: list[tuple[int, int]] = order_moves(board, get_legal_moves(board), player)
        self.untried_moves.reverse()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.board.check_connection(1) or self.board.check_connection(2)

    def add_child(
        self, move: tuple[int, int], next_player: int, remove_from_untried: bool = True
    ) -> "Node":
        new_board = self.board.clone()
        new_board.place_piece(move[0], move[1], self.player)
        child = Node(board=new_board, player=next_player, parent=self, move=move)

        if remove_from_untried and self.untried_moves:
            if self.untried_moves[-1] == move:
                self.untried_moves.pop()
            elif move in self.untried_moves:
                self.untried_moves.remove(move)

        self.children.append(child)
        return child

    def update(self, winner: int, root_player: int) -> None:
        self.visits += 1
        if winner == root_player:
            self.wins += 1

    def update_rave(self, winner: int, root_player: int) -> None:
        self.rave_visits += 1
        if winner == root_player:
            self.rave_wins += 1


class MCTS:
    def __init__(self, iterations: int = 1000) -> None:
        self.iterations = iterations
        self.root_player: int = 1
        self.transposition_table: dict[
            int,
            tuple[int, float, int, float],
        ] = {}
        self.zobrist_size: int = -1
        self.zobrist_table: list[list[list[int]]] = []
        self.zobrist_player_key: dict[int, int] = {}
        self.tactical_tt: dict[
            tuple[int, int],
            tuple[int, int, str, Optional[tuple[int, int]]],
        ] = {}
        self.killer_moves: dict[int, list[tuple[int, int]]] = {}
        self.history_scores: dict[tuple[int, int, int], int] = {}
        self.two_distance_cache: dict[tuple[int, int], int] = {}

    def init_zobrist(self, board_size: int) -> None:
        if self.zobrist_size == board_size and self.zobrist_table:
            return
        rng = random.Random(112358 + board_size)
        self.zobrist_size = board_size
        self.zobrist_table = [
            [[rng.getrandbits(64) for _ in range(3)] for _ in range(board_size)]
            for _ in range(board_size)
        ]
        self.zobrist_player_key = {1: rng.getrandbits(64), 2: rng.getrandbits(64)}

    def compute_zobrist_hash(self, board_matrix: list[list[int]], player_to_move: int) -> int:
        z_hash = self.zobrist_player_key.get(player_to_move, 0)
        board_size = len(board_matrix)
        for row in range(board_size):
            for col in range(board_size):
                value = board_matrix[row][col]
                if value != 0:
                    z_hash ^= self.zobrist_table[row][col][value]
        return z_hash

    def get_two_distance_cached(self, board_matrix: list[list[int]], player_id: int, position_hash: int) -> int:
        cache_key = (position_hash, player_id)
        cached = self.two_distance_cache.get(cache_key)
        if cached is not None:
            return cached

        value = two_distance_cost(board_matrix, player_id)
        self.two_distance_cache[cache_key] = value
        return value

    def tactical_eval(self, board_matrix: list[list[int]], player_to_move: int, z_hash: int) -> int:
        if check_connection_matrix(board_matrix, self.root_player):
            return 100000
        if check_connection_matrix(board_matrix, opponent(self.root_player)):
            return -100000

        position_hash = z_hash ^ self.zobrist_player_key[player_to_move]
        self_dist = self.get_two_distance_cached(board_matrix, self.root_player, position_hash)
        opp_dist = self.get_two_distance_cached(board_matrix, opponent(self.root_player), position_hash)
        base = (opp_dist - self_dist) * 100
        return base if player_to_move == self.root_player else -base

    def order_tactical_moves(
        self,
        board_matrix: list[list[int]],
        moves: list[tuple[int, int]],
        player_to_move: int,
        depth: int,
        tt_move: Optional[tuple[int, int]] = None,
    ) -> list[tuple[int, int]]:
        board_size = len(board_matrix)
        center = (board_size - 1) / 2.0
        killer_set = set(self.killer_moves.get(depth, []))

        scored: list[tuple[int, tuple[int, int]]] = []
        for row, col in moves:
            history = self.history_scores.get((player_to_move, row, col), 0)
            tt_bonus = 50000 if tt_move is not None and (row, col) == tt_move else 0
            killer_bonus = 20000 if (row, col) in killer_set else 0
            center_bonus = int(100 - ((row - center) * (row - center) + (col - center) * (col - center)) * 10)
            scored.append((tt_bonus + killer_bonus + history + center_bonus, (row, col)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def update_killer(self, depth: int, move: tuple[int, int]) -> None:
        killers = self.killer_moves.setdefault(depth, [])
        if move in killers:
            killers.remove(move)
        killers.insert(0, move)
        if len(killers) > 2:
            killers.pop()

    def negamax(
        self,
        board_matrix: list[list[int]],
        player_to_move: int,
        depth: int,
        alpha: int,
        beta: int,
        z_hash: int,
        ply: int,
        deadline: float,
    ) -> int:
        if time.time() >= deadline:
            return self.tactical_eval(board_matrix, player_to_move, z_hash)

        if check_connection_matrix(board_matrix, self.root_player):
            return 100000 - ply
        if check_connection_matrix(board_matrix, opponent(self.root_player)):
            return -100000 + ply
        if depth == 0:
            return self.tactical_eval(board_matrix, player_to_move, z_hash)

        tt_key = (z_hash, player_to_move)
        tt_entry = self.tactical_tt.get(tt_key)
        tt_move: Optional[tuple[int, int]] = None
        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == "EXACT":
                    return tt_score
                if tt_flag == "LOWER":
                    alpha = max(alpha, tt_score)
                elif tt_flag == "UPPER":
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        alpha_orig = alpha
        best_score = -10**9
        legal = focused_moves(board_matrix)
        if not legal:
            return self.tactical_eval(board_matrix, player_to_move, z_hash)

        ordered_moves = self.order_tactical_moves(
            board_matrix,
            legal,
            player_to_move,
            depth,
            tt_move=tt_move,
        )
        best_move_found = ordered_moves[0]
        for move_index, (row, col) in enumerate(ordered_moves):
            board_matrix[row][col] = player_to_move
            child_hash = z_hash ^ self.zobrist_table[row][col][player_to_move] ^ self.zobrist_player_key[player_to_move] ^ self.zobrist_player_key[opponent(player_to_move)]

            is_killer = (row, col) in self.killer_moves.get(depth, [])
            is_forcing = check_connection_matrix(board_matrix, player_to_move)
            reduced_depth = depth - 1
            if move_index >= 3 and depth >= 3 and not is_killer and not is_forcing:
                reduced_depth = depth - 2

            if move_index == 0:
                score = -self.negamax(
                    board_matrix,
                    opponent(player_to_move),
                    reduced_depth,
                    -beta,
                    -alpha,
                    child_hash,
                    ply + 1,
                    deadline,
                )
            else:
                score = -self.negamax(
                    board_matrix,
                    opponent(player_to_move),
                    reduced_depth,
                    -(alpha + 1),
                    -alpha,
                    child_hash,
                    ply + 1,
                    deadline,
                )

                if score > alpha and score < beta:
                    score = -self.negamax(
                        board_matrix,
                        opponent(player_to_move),
                        depth - 1,
                        -beta,
                        -alpha,
                        child_hash,
                        ply + 1,
                        deadline,
                    )

            board_matrix[row][col] = 0

            if score > best_score:
                best_score = score
                best_move_found = (row, col)
            if score > alpha:
                alpha = score

            if alpha >= beta:
                self.update_killer(depth, (row, col))
                key = (player_to_move, row, col)
                self.history_scores[key] = self.history_scores.get(key, 0) + depth * depth
                break

        if best_score <= alpha_orig:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        else:
            flag = "EXACT"

        prev = self.tactical_tt.get(tt_key)
        if prev is None or depth >= prev[0]:
            self.tactical_tt[tt_key] = (depth, best_score, flag, best_move_found)

        return best_score

    def tactical_best_move(
        self,
        board: HexBoard,
        player: int,
        deadline: float,
        max_depth: int = 3,
    ) -> Optional[tuple[int, int]]:
        board_matrix = [row[:] for row in board.board]
        legal = focused_moves(board_matrix)
        if not legal:
            return None

        z_hash = self.compute_zobrist_hash(board_matrix, player)
        best_move = legal[0]
        best_score = -10**9
        aspiration = 120

        for depth in range(1, max_depth + 1):
            if time.time() >= deadline:
                break

            root_tt_entry = self.tactical_tt.get((z_hash, player))
            root_tt_move = root_tt_entry[3] if root_tt_entry is not None else None
            ordered = self.order_tactical_moves(board_matrix, legal, player, depth, tt_move=root_tt_move)
            alpha = best_score - aspiration if best_score > -10**8 else -10**9
            beta = best_score + aspiration if best_score > -10**8 else 10**9
            local_best_move = ordered[0]
            local_best_score = -10**9

            for row, col in ordered:
                if time.time() >= deadline:
                    break
                board_matrix[row][col] = player
                child_hash = z_hash ^ self.zobrist_table[row][col][player] ^ self.zobrist_player_key[player] ^ self.zobrist_player_key[opponent(player)]
                score = -self.negamax(
                    board_matrix,
                    opponent(player),
                    depth - 1,
                    -beta,
                    -alpha,
                    child_hash,
                    1,
                    deadline,
                )
                board_matrix[row][col] = 0

                if score > local_best_score:
                    local_best_score = score
                    local_best_move = (row, col)
                if score > alpha:
                    alpha = score

            if local_best_score > best_score:
                best_score = local_best_score
                best_move = local_best_move

            aspiration = min(1000, int(aspiration * 1.5))

        return best_move

    def widening_limit(self, visits: int) -> int:
        return max(1, int(math.sqrt(visits + 1)))

    def uct_score(self, parent_visits: int, parent_board: HexBoard, child: Node) -> float:
        table_key = self.compute_zobrist_hash(child.board.board, child.player)
        table_visits, table_wins, table_rave_visits, table_rave_wins = self.transposition_table.get(
            table_key,
            (0, 0.0, 0, 0.0),
        )

        agg_visits = child.visits + table_visits
        agg_wins = child.wins + table_wins
        agg_rave_visits = child.rave_visits + table_rave_visits
        agg_rave_wins = child.rave_wins + table_rave_wins

        if agg_visits == 0:
            return float("inf")

        normal_value = agg_wins / agg_visits
        exploration = 1.4 * math.sqrt(math.log(parent_visits) / agg_visits)

        if agg_rave_visits == 0:
            blended_value = normal_value
        else:
            beta = agg_rave_visits / (agg_visits + agg_rave_visits + 300.0)
            rave_value = agg_rave_wins / agg_rave_visits
            blended_value = (1.0 - beta) * normal_value + beta * rave_value

        bias = 0.0 if child.move is None else score_move(parent_board, child.move, self.root_player)
        progressive_bias = bias / (agg_visits + 10)
        return blended_value + exploration + progressive_bias

    def search(self, board: HexBoard, player: int) -> tuple[int, int]:
        self.root_player = player
        self.transposition_table = {}
        self.tactical_tt = {}
        self.killer_moves = {}
        self.history_scores = {}
        self.two_distance_cache = {}
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            raise ValueError("No legal moves available")

        board_size = getattr(board, "size", len(board.board))
        self.init_zobrist(board_size)
        total_cells = board_size * board_size

        winning_move = find_immediate_winning_move(board, legal_moves, player)
        if winning_move is not None:
            return winning_move

        block_move = find_urgent_block_move(board, legal_moves, player)
        if block_move is not None:
            return block_move

        forcing_cutoff = max(10, min(24, total_cells // 12))
        forcing_candidates = max(6, min(10, board_size // 2 + 3))
        if len(legal_moves) <= forcing_cutoff:
            forcing_move = find_forcing_two_ply_move(
                board,
                legal_moves,
                player,
                max_candidates=forcing_candidates,
            )
            if forcing_move is not None:
                return forcing_move

        number_of_moves_played = total_cells - len(legal_moves)

        if legal_moves and number_of_moves_played <= 1:
            center = (board_size - 1) / 2.0
            return min(
                legal_moves,
                key=lambda move: (move[0] - center) * (move[0] - center)
                + (move[1] - center) * (move[1] - center),
            )

        root = Node(board=board.clone(), player=player)
        start_time = time.time()
        deadline = start_time + 4.45

        tactical_budget = min(deadline - time.time(), 0.35)
        if tactical_budget > 0.03 and len(legal_moves) <= max(14, board_size * 2):
            tactical_move = self.tactical_best_move(board, player, time.time() + tactical_budget, max_depth=3)
            if tactical_move is not None:
                return tactical_move

        while time.time() < deadline:
            node = self.selection(root)
            node = self.expansion(node)
            winner, played_moves = self.simulation(node, deadline)
            self.backpropagation(node, winner, played_moves)

        if not root.children:
            return random_move(board)

        best_child = max(root.children, key=lambda child: child.visits)
        if best_child.move is None:
            return random_move(board)
        return best_child.move

    def selection(self, node: Node) -> Node:
        while not node.is_terminal() and node.children:
            if node.untried_moves and len(node.children) < self.widening_limit(node.visits):
                break

            if not node.children:
                break

            parent_visits = max(1, node.visits)
            node = max(node.children, key=lambda child: self.uct_score(parent_visits, node.board, child))
        return node

    def expansion(self, node: Node) -> Node:
        if node.is_terminal() or not node.untried_moves:
            return node

        if len(node.children) >= self.widening_limit(node.visits):
            return node

        move = node.untried_moves.pop()
        next_player = opponent(node.player)
        return node.add_child(move=move, next_player=next_player, remove_from_untried=False)

    def simulation(self, node: Node, deadline: float) -> tuple[int, list[tuple[int, int]]]:
        rollout_board = node.board.clone()
        board_matrix = rollout_board.board
        board_size = getattr(rollout_board, "size", len(board_matrix))
        rollout_cutoff = max(6, min(12, board_size // 2))
        current_player = node.player
        played_moves: list[tuple[int, int]] = []

        total_cells = board_size * board_size
        virtual_a = total_cells
        virtual_b = total_cells + 1
        uf_player_1 = UnionFind(total_cells + 2)
        uf_player_2 = UnionFind(total_cells + 2)

        def cell_index(row: int, col: int) -> int:
            return row * board_size + col

        def union_for_piece(row: int, col: int, player_id: int) -> None:
            idx = cell_index(row, col)
            uf = uf_player_1 if player_id == 1 else uf_player_2

            if player_id == 1:
                if col == 0:
                    uf.union(idx, virtual_a)
                if col == board_size - 1:
                    uf.union(idx, virtual_b)
            else:
                if row == 0:
                    uf.union(idx, virtual_a)
                if row == board_size - 1:
                    uf.union(idx, virtual_b)

            for neighbor_row, neighbor_col in get_hex_neighbors(row, col, board_size):
                if board_matrix[neighbor_row][neighbor_col] == player_id:
                    uf.union(idx, cell_index(neighbor_row, neighbor_col))

        for row in range(board_size):
            for col in range(board_size):
                piece = board_matrix[row][col]
                if piece == 1:
                    union_for_piece(row, col, 1)
                elif piece == 2:
                    union_for_piece(row, col, 2)

        if uf_player_1.connected(virtual_a, virtual_b):
            return 1, played_moves
        if uf_player_2.connected(virtual_a, virtual_b):
            return 2, played_moves

        legal_moves = get_legal_moves(rollout_board)
        while True:
            if time.time() >= deadline:
                if uf_player_1.connected(virtual_a, virtual_b):
                    return 1, played_moves
                if uf_player_2.connected(virtual_a, virtual_b):
                    return 2, played_moves
                if not legal_moves:
                    return opponent(current_player), played_moves

                root_best = max(score_move(rollout_board, move, self.root_player) for move in legal_moves)
                opp_player = opponent(self.root_player)
                opp_best = max(score_move(rollout_board, move, opp_player) for move in legal_moves)
                winner_estimate = self.root_player if root_best >= opp_best else opp_player
                return winner_estimate, played_moves

            if not legal_moves:
                return opponent(current_player), played_moves

            if len(legal_moves) <= rollout_cutoff:
                if uf_player_1.connected(virtual_a, virtual_b):
                    return 1, played_moves
                if uf_player_2.connected(virtual_a, virtual_b):
                    return 2, played_moves

            if random.random() < 0.8:
                move = random.choice(legal_moves)
            else:
                move = heuristic_rollout_move(rollout_board, current_player, legal_moves)
            played_moves.append(move)
            rollout_board.place_piece(move[0], move[1], current_player)
            union_for_piece(move[0], move[1], current_player)

            legal_moves.remove(move)

            if current_player == 1:
                if uf_player_1.connected(virtual_a, virtual_b):
                    return 1, played_moves
            else:
                if uf_player_2.connected(virtual_a, virtual_b):
                    return 2, played_moves

            current_player = opponent(current_player)

    def backpropagation(self, node: Node, winner: int, played_moves: list[tuple[int, int]]) -> None:
        played_moves_set = set(played_moves)
        current: Optional[Node] = node
        while current is not None:
            current.update(winner=winner, root_player=self.root_player)
            table_key = self.compute_zobrist_hash(current.board.board, current.player)
            table_visits, table_wins, table_rave_visits, table_rave_wins = self.transposition_table.get(
                table_key,
                (0, 0.0, 0, 0.0),
            )
            table_visits += 1
            if winner == self.root_player:
                table_wins += 1.0

            if current.move is not None and current.move in played_moves_set:
                current.update_rave(winner=winner, root_player=self.root_player)
                table_rave_visits += 1
                if winner == self.root_player:
                    table_rave_wins += 1.0

            self.transposition_table[table_key] = (
                table_visits,
                table_wins,
                table_rave_visits,
                table_rave_wins,
            )
            current = current.parent


class SmartPlayer(Player):
    def __init__(self, player_id: int, iterations: int = 1000) -> None:
        self.player_id = player_id
        self.mcts = MCTS(iterations=iterations)

    def play(self, board: HexBoard) -> tuple[int, int]:
        return self.mcts.search(board, self.player_id)
