from __future__ import annotations

import math
import random
import time
from typing import Optional


class Player:
    pass


class HexBoard:
    board: list[list[int]]


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


def random_move(board: HexBoard) -> tuple[int, int]:
    legal_moves = get_legal_moves(board)
    if not legal_moves:
        raise ValueError("No legal moves available")
    return random.choice(legal_moves)


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
    neighbor_deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    bridge_pairs = [
        ((-1, 0), (0, 1)),
        ((-1, 1), (1, 0)),
        ((0, -1), (1, -1)),
        ((-1, 1), (0, -1)),
        ((-1, 0), (1, -1)),
        ((0, 1), (1, 0)),
    ]

    for delta_row, delta_col in neighbor_deltas:
        neighbor_row = row + delta_row
        neighbor_col = col + delta_col
        if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size:
            neighbor_value = board_matrix[neighbor_row][neighbor_col]
            if neighbor_value == player:
                player_adjacent += 1
            elif neighbor_value == enemy_player:
                opponent_adjacent += 1

    friendly_connection_bonus = 4.0 if player_adjacent >= 2 else 0.0
    blocking_bonus = 2.0 if opponent_adjacent >= 2 else 0.0
    bridge_bonus = 0.0
    block_bridge_bonus = 0.0

    for (delta_row_a, delta_col_a), (delta_row_b, delta_col_b) in bridge_pairs:
        row_a = row + delta_row_a
        col_a = col + delta_col_a
        row_b = row + delta_row_b
        col_b = col + delta_col_b

        if not (0 <= row_a < board_size and 0 <= col_a < board_size):
            continue
        if not (0 <= row_b < board_size and 0 <= col_b < board_size):
            continue

        value_a = board_matrix[row_a][col_a]
        value_b = board_matrix[row_b][col_b]

        if value_a == player and value_b == player:
            bridge_bonus = 5.0
        if value_a == enemy_player and value_b == enemy_player:
            block_bridge_bonus = 2.5

        if bridge_bonus > 0.0 and block_bridge_bonus > 0.0:
            break

    return (
        (3.0 * player_adjacent)
        + (1.5 * opponent_adjacent)
        + (0.2 * center_score)
        + friendly_connection_bonus
        + blocking_bonus
        + bridge_bonus
        + block_bridge_bonus
    )


def order_moves(board: HexBoard, moves: list[tuple[int, int]], player: int) -> list[tuple[int, int]]:
    return sorted(moves, key=lambda move: score_move(board, move, player), reverse=True)


def heuristic_rollout_move(
    board: HexBoard, player: int, legal_moves: list[tuple[int, int]]
) -> tuple[int, int]:
    if not legal_moves:
        raise ValueError("No legal moves available")

    board_matrix = board.board
    board_size = getattr(board, "size", len(board_matrix))
    center = (board_size - 1) / 2.0
    neighbor_deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    best_score: Optional[tuple[int, float]] = None
    best_moves: list[tuple[int, int]] = []
    all_scores_equal = True
    first_score: Optional[tuple[int, float]] = None

    for row, col in legal_moves:
        adjacent_count = 0
        for delta_row, delta_col in neighbor_deltas:
            neighbor_row = row + delta_row
            neighbor_col = col + delta_col
            if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size:
                if board_matrix[neighbor_row][neighbor_col] == player:
                    adjacent_count += 1

        distance_to_center_sq = (row - center) * (row - center) + (col - center) * (col - center)
        score = (adjacent_count, -distance_to_center_sq)

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

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.board.check_connection(1) or self.board.check_connection(2)

    def add_child(self, move: tuple[int, int], next_player: int) -> "Node":
        new_board = self.board.clone()
        new_board.place_piece(move[0], move[1], self.player)
        child = Node(board=new_board, player=next_player, parent=self, move=move)

        try:
            self.untried_moves.remove(move)
        except ValueError:
            pass

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

    def uct_score(self, parent_visits: int, parent_board: HexBoard, child: Node) -> float:
        if child.visits == 0:
            return float("inf")

        normal_value = child.wins / child.visits
        exploration = 1.4 * math.sqrt(math.log(parent_visits) / child.visits)

        if child.rave_visits == 0:
            blended_value = normal_value
        else:
            beta = child.rave_visits / (child.visits + child.rave_visits + 1e-6)
            rave_value = child.rave_wins / child.rave_visits
            blended_value = (1.0 - beta) * normal_value + beta * rave_value

        bias = 0.0 if child.move is None else score_move(parent_board, child.move, self.root_player)
        progressive_bias = bias / (child.visits + 1)
        return blended_value + exploration + progressive_bias

    def search(self, board: HexBoard, player: int) -> tuple[int, int]:
        self.root_player = player
        legal_moves = get_legal_moves(board)
        board_size = getattr(board, "size", len(board.board))
        total_cells = board_size * board_size
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

        while time.time() - start_time < 4.5:
            node = self.selection(root)
            node = self.expansion(node)
            winner, played_moves = self.simulation(node)
            self.backpropagation(node, winner, played_moves)

        if not root.children:
            return random_move(board)

        best_child = max(root.children, key=lambda child: child.visits if child.visits > 0 else -1)
        if best_child.move is None:
            return random_move(board)
        return best_child.move

    def selection(self, node: Node) -> Node:
        while not node.is_terminal() and node.is_fully_expanded() and node.children:
            parent_visits = max(1, node.visits)
            node = max(node.children, key=lambda child: self.uct_score(parent_visits, node.board, child))
        return node

    def expansion(self, node: Node) -> Node:
        if node.is_terminal() or not node.untried_moves:
            return node

        move = node.untried_moves[0]
        next_player = opponent(node.player)
        return node.add_child(move=move, next_player=next_player)

    def simulation(self, node: Node) -> tuple[int, list[tuple[int, int]]]:
        rollout_board = node.board.clone()
        board_matrix = rollout_board.board
        board_size = getattr(rollout_board, "size", len(board_matrix))
        current_player = node.player
        played_moves: list[tuple[int, int]] = []
        neighbor_deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

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

            for delta_row, delta_col in neighbor_deltas:
                neighbor_row = row + delta_row
                neighbor_col = col + delta_col
                if 0 <= neighbor_row < board_size and 0 <= neighbor_col < board_size:
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
            if not legal_moves:
                return opponent(current_player), played_moves

            if len(legal_moves) <= 6:
                if rollout_board.check_connection(1):
                    return 1, played_moves
                if rollout_board.check_connection(2):
                    return 2, played_moves

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
            if current.move is not None and current.move in played_moves_set:
                current.update_rave(winner=winner, root_player=self.root_player)
            current = current.parent


class SmartPlayer(Player):
    def __init__(self, player_id: int, iterations: int = 1000) -> None:
        self.player_id = player_id
        self.mcts = MCTS(iterations=iterations)

    def play(self, board: HexBoard) -> tuple[int, int]:
        return self.mcts.search(board, self.player_id)
