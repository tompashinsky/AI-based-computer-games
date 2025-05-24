import random
import math
import copy
import pickle
import os

class Node:
    def __init__(self, state, parent=None, move=None, strategy=0.8):
        self.state = state  # The game state (board)
        self.parent = parent  # Parent node
        self.move = move  # The move that led to this state
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node
        self.losses = 0  # Number of losses from this node
        self.untried_moves = self.get_available_moves()  # Moves not yet tried
        self.strategy = strategy  # Strategy parameter for UCB1

    def get_available_moves(self):
        """Get available moves, prioritized by domain knowledge"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    moves.append((i, j))
        
        # Sort moves based on domain knowledge
        #return self.prioritize_moves(moves) - Original code
        return moves

    def prioritize_moves(self, moves):
        """Prioritize moves based on Tic Tac Toe strategy"""
        if not moves:
            return moves

        current_player = 2 if sum(row.count(1) for row in self.state) > sum(row.count(2) for row in self.state) else 1
        prioritized_moves = []
        
        # Check for winning moves
        winning_moves = self.find_winning_moves(current_player, moves)
        if winning_moves:
            return winning_moves

        # Check for blocking opponent's winning moves
        opponent = 3 - current_player
        blocking_moves = self.find_winning_moves(opponent, moves)
        if blocking_moves:
            return blocking_moves

        # Prioritize center if available
        if (1, 1) in moves:
            prioritized_moves.append((1, 1))
            moves.remove((1, 1))

        # Prioritize corners
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for corner in corners:
            if corner in moves:
                prioritized_moves.append(corner)
                moves.remove(corner)

        # Add remaining moves
        prioritized_moves.extend(moves)
        return prioritized_moves

    def find_winning_moves(self, player, available_moves):
        """Find moves that would result in an immediate win"""
        winning_moves = []
        for move in available_moves:
            # Try the move
            temp_state = copy.deepcopy(self.state)
            temp_state[move[0]][move[1]] = player
            
            # Check if it's a winning move
            if self.check_win(temp_state, player):
                winning_moves.append(move)
        
        return winning_moves

    def check_win(self, board, player):
        # Check rows and columns
        for i in range(3):
            if all(board[i][j] == player for j in range(3)) or \
               all(board[j][i] == player for j in range(3)):
                return True
        
        # Check diagonals
        if all(board[i][i] == player for i in range(3)) or \
           all(board[i][2-i] == player for i in range(3)):
            return True
        
        return False

    def get_position_weight(self, move):
        """Calculate position weight based on strategic value"""
        if not move:
            return 0
            
        # Center position has highest weight
        if move == (1, 1):
            return 3.0  # Significantly increased from 2.0
            
        # Corners have second highest weight
        if move in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return 1.5  # Increased from 0.8
            
        # Edges have lowest weight
        return 0.5  # Increased from 0.4

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        
        # Modified UCB1 with domain knowledge weight
        win_rate = (self.wins - self.losses) / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        # Add domain knowledge bonus with increased weight
        position_weight = self.get_position_weight(self.move) if self.move else 0
        
        return win_rate + exploration_term + self.strategy * position_weight

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb1())

    def add_child(self, move):
        child_state = copy.deepcopy(self.state)
        child_state[move[0]][move[1]] = 2 if sum(row.count(1) for row in self.state) > sum(row.count(2) for row in self.state) else 1
        child = Node(child_state, self, move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result, is_loss=False):
        self.visits += 1
        if is_loss:
            self.losses += 1
        else:
            self.wins += result

    def get_state_key(self):
        return tuple(tuple(row) for row in self.state)

class MCTS:
    def __init__(self, iterations=1000, strategy=0.8):
        self.iterations = iterations
        self.knowledge_base = {}  # Dictionary to store learned knowledge
        self.model_path = "tictactoe_model.pkl"
        self.last_game_moves = []  # Track moves from the last game
        self.last_game_result = None  # Track the result of the last game
        self.strategy = strategy  # Strategy parameter for UCB1

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

    def record_move(self, state, move):
        """Record a move made during the game"""
        self.last_game_moves.append((state, move))

    def learn_from_loss(self):
        """Learn from the last game if it was a loss"""
        if self.last_game_result == 0:  # If the last game was a loss
            # Penalize the moves that led to the loss
            for state, move in self.last_game_moves:
                state_key = tuple(tuple(row) for row in state)
                if state_key in self.knowledge_base:
                    # Increase the loss count for these states
                    self.knowledge_base[state_key]['losses'] = self.knowledge_base[state_key].get('losses', 0) + 1
                    # Decrease the win rate to make these moves less likely to be chosen
                    self.knowledge_base[state_key]['wins'] = max(0, self.knowledge_base[state_key].get('wins', 0) - 1)
                    
                    # Extra penalty for not playing center when available
                    if move != (1, 1) and state[1][1] == 0:
                        self.knowledge_base[state_key]['wins'] = max(0, self.knowledge_base[state_key].get('wins', 0) - 1.0)
                    
                    # Extra penalty for not playing corners when center is taken
                    if move not in [(0, 0), (0, 2), (2, 0), (2, 2)] and state[1][1] == 1:
                        self.knowledge_base[state_key]['wins'] = max(0, self.knowledge_base[state_key].get('wins', 0) - 0.5)
            self.save_knowledge()

    def get_best_move(self, board):
        root = Node(board, strategy=self.strategy)
        state_key = root.get_state_key()
        
        # If we have knowledge about this state, use it to initialize the root node
        if state_key in self.knowledge_base:
            root.visits = self.knowledge_base[state_key]['visits']
            root.wins = self.knowledge_base[state_key]['wins']
            root.losses = self.knowledge_base[state_key].get('losses', 0)
        
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
            'wins': root.wins,
            'losses': root.losses
        }
        
        # Record this move for learning from losses
        best_move = max(root.children, key=lambda c: c.visits).move
        self.record_move(board, best_move)
        
        return best_move

    def simulate_random_play(self, board):
        """Simulate a game with smart playout strategy"""
        temp_board = copy.deepcopy(board)
        current_player = 2 if sum(row.count(1) for row in board) > sum(row.count(2) for row in board) else 1
        
        while True:
            # Check for win
            if self.check_win(temp_board, 1):
                return 1
            if self.check_win(temp_board, 2):
                return 0
            
            # Check for draw
            if all(all(cell != 0 for cell in row) for row in temp_board):
                return 0.5
            
            # Create a temporary node to use its move prioritization
            temp_node = Node(temp_board)
            available_moves = temp_node.get_available_moves()
            
            if not available_moves:
                return 0.5
            
            # Use weighted random choice based on move priority
            weights = []
            for move in available_moves:
                if move == (1, 1):  # Center
                    weights.append(0.6)  # Increased from 0.5
                elif move in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Corners
                    weights.append(0.3)
                else:  # Edges
                    weights.append(0.1)  # Decreased from 0.2
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            
            # Make weighted random move
            move = random.choices(available_moves, weights=weights, k=1)[0]
            temp_board[move[0]][move[1]] = current_player
            current_player = 3 - current_player

    def check_win(self, board, player):
        # Check rows and columns
        for i in range(3):
            if all(board[i][j] == player for j in range(3)) or \
               all(board[j][i] == player for j in range(3)):
                return True
        
        # Check diagonals
        if all(board[i][i] == player for i in range(3)) or \
           all(board[i][2-i] == player for i in range(3)):
            return True
        
        return False

    def record_game_result(self, result):
        """Record the result of the last game"""
        self.last_game_result = result
        if result == 0:  # If it was a loss
            self.learn_from_loss()
        elif result == 0.5:  # If it was a draw
            self.learn_from_draw()
        self.last_game_moves = []  # Reset moves for next game

    def learn_from_draw(self):
        """Learn from a drawn game"""
        if self.last_game_result == 0.5:  # If the last game was a draw
            # Update knowledge base with neutral feedback
            for state, move in self.last_game_moves:
                state_key = tuple(tuple(row) for row in state)
                if state_key in self.knowledge_base:
                    # Increase visits but don't change win/loss ratio
                    self.knowledge_base[state_key]['visits'] += 1
                    # Slightly increase wins to encourage these moves
                    # (since they led to a draw rather than a loss)
                    self.knowledge_base[state_key]['wins'] += 0.5
            self.save_knowledge() 