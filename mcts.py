import random
import math
import copy
import pickle
import os

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # The game state (board)
        self.parent = parent  # Parent node
        self.move = move  # The move that led to this state
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node
        self.untried_moves = self.get_available_moves()  # Moves not yet tried

    def get_available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    moves.append((i, j))
        return moves

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb1())

    def add_child(self, move):
        child_state = copy.deepcopy(self.state)
        child_state[move[0]][move[1]] = 2 if self.state[move[0]][move[1]] == 1 else 1
        child = Node(child_state, self, move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def get_state_key(self):
        # Convert board state to a tuple of tuples for hashing
        return tuple(tuple(row) for row in self.state)

class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.knowledge_base = {}  # Dictionary to store learned knowledge
        self.model_path = "tictactoe_model.pkl"

    def load_knowledge(self):
        """Load the trained model if it exists"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.knowledge_base = pickle.load(f)
            print(f"Loaded knowledge base with {len(self.knowledge_base)} states")

    def save_knowledge(self):
        """Save the current knowledge base"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        print(f"Saved knowledge base with {len(self.knowledge_base)} states")

    def get_best_move(self, board):
        root = Node(board)
        state_key = root.get_state_key()
        
        # If we have knowledge about this state, use it to initialize the root node
        if state_key in self.knowledge_base:
            root.visits = self.knowledge_base[state_key]['visits']
            root.wins = self.knowledge_base[state_key]['wins']
        
        for _ in range(self.iterations):
            node = root
            # Selection
            while not node.untried_moves and node.children:
                node = node.select_child()
            
            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)
            
            # Simulation
            result = self.simulate_random_play(node.state)
            
            # Backpropagation
            while node:
                node.update(result)
                node = node.parent
        
        # Update knowledge base with root node's statistics
        self.knowledge_base[state_key] = {
            'visits': root.visits,
            'wins': root.wins
        }
        
        # Return the move with the most visits
        return max(root.children, key=lambda c: c.visits).move

    def simulate_random_play(self, board):
        temp_board = copy.deepcopy(board)
        current_player = 2 if any(1 in row for row in board) else 1
        
        while True:
            # Check for win
            if self.check_win(temp_board, 1):
                return 1
            if self.check_win(temp_board, 2):
                return 0
            
            # Check for draw
            if all(all(cell != 0 for cell in row) for row in temp_board):
                return 0.5
            
            # Make random move
            available_moves = [(i, j) for i in range(3) for j in range(3) if temp_board[i][j] == 0]
            if not available_moves:
                return 0.5
            
            move = random.choice(available_moves)
            temp_board[move[0]][move[1]] = current_player
            current_player = 3 - current_player

    def check_win(self, board, player):
        # Check rows
        for row in board:
            if all(cell == player for cell in row):
                return True
        
        # Check columns
        for col in range(3):
            if all(board[row][col] == player for row in range(3)):
                return True
        
        # Check diagonals
        if all(board[i][i] == player for i in range(3)):
            return True
        if all(board[i][2-i] == player for i in range(3)):
            return True
        
        return False 