import numpy as np
import matplotlib.pyplot as plt
from mcts import MCTS
import random
import copy
import os

class RandomOpponent:
    def get_move(self, board):
        available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
        return random.choice(available_moves) if available_moves else None

class RationalOpponent:
    def get_move(self, board):
        # Check for winning move
        move = self.find_winning_move(board, 1)
        if move:
            return move
            
        # Check for blocking opponent's winning move
        move = self.find_winning_move(board, 2)
        if move:
            return move
            
        # Take center if available
        if board[1][1] == 0:
            return (1, 1)
            
        # Take corner if available
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        available_corners = [corner for corner in corners if board[corner[0]][corner[1]] == 0]
        if available_corners:
            return random.choice(available_corners)
            
        # Take any available edge
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        available_edges = [edge for edge in edges if board[edge[0]][edge[1]] == 0]
        if available_edges:
            return random.choice(available_edges)
            
        return None

    def find_winning_move(self, board, player):
        # Check rows and columns
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    # Check row
                    if sum(board[i]) == 2 * player:
                        return (i, j)
                    # Check column
                    if sum(board[k][j] for k in range(3)) == 2 * player:
                        return (i, j)
        
        # Check diagonals
        if board[1][1] == 0:
            # Main diagonal
            if board[0][0] == board[2][2] == player:
                return (1, 1)
            # Anti-diagonal
            if board[0][2] == board[2][0] == player:
                return (1, 1)
        
        return None

def check_winner(board):
    # Check rows and columns
    for i in range(3):
        if all(board[i][j] == 1 for j in range(3)) or all(board[j][i] == 1 for j in range(3)):
            return 1
        if all(board[i][j] == 2 for j in range(3)) or all(board[j][i] == 2 for j in range(3)):
            return 2
    
    # Check diagonals
    if all(board[i][i] == 1 for i in range(3)) or all(board[i][2-i] == 1 for i in range(3)):
        return 1
    if all(board[i][i] == 2 for i in range(3)) or all(board[i][2-i] == 2 for i in range(3)):
        return 2
    
    # Check for draw
    if all(all(cell != 0 for cell in row) for row in board):
        return 0
    
    return None

def play_game(ai, opponent, ai_first=True):
    board = [[0 for _ in range(3)] for _ in range(3)]
    current_player = 1 if ai_first else 2
    
    while True:
        if current_player == 1:
            if ai_first:
                move = ai.get_best_move(board)
            else:
                move = opponent.get_move(board)
        else:
            if ai_first:
                move = opponent.get_move(board)
            else:
                move = ai.get_best_move(board)
        
        if move is None:
            return 0  # Draw
        
        board[move[0]][move[1]] = current_player
        winner = check_winner(board)
        
        if winner is not None:
            if winner == 0:
                return 0  # Draw
            return 1 if (winner == 1 and ai_first) or (winner == 2 and not ai_first) else 0
        
        current_player = 3 - current_player

def train_ai_self_play(ai, num_games):
    """Train AI against itself for specified number of games"""
    for _ in range(num_games):
        board = [[0 for _ in range(3)] for _ in range(3)]
        current_player = 1
        
        while True:
            move = ai.get_best_move(board)
            if move is None:
                break
                
            board[move[0]][move[1]] = current_player
            winner = check_winner(board)
            
            if winner is not None:
                # Record game result (1 for win, 0 for loss/draw)
                result = 1 if winner == current_player else 0
                ai.record_game_result(result)
                break
                
            current_player = 3 - current_player

def test_ai_performance(ai, opponent, num_games):
    """Test AI against an opponent for specified number of games"""
    wins = 0
    for _ in range(num_games):
        ai_first = random.choice([True, False])
        result = play_game(ai, opponent, ai_first)
        wins += result
    return wins / num_games

def analyze_ai_performance(strategy_param, num_games, games_per_interval=100, test_games=1000):
    # Create models directory if it doesn't exist
    os.makedirs('analysis_models', exist_ok=True)
    
    # Initialize AI with strategy parameter
    model_path = f'analysis_models/tictactoe_model_strategy_{strategy_param:.1f}.pkl'
    ai = MCTS(iterations=1500, strategy=strategy_param)
    ai.model_path = model_path
    
    # Initialize opponents
    random_opponent = RandomOpponent()
    rational_opponent = RationalOpponent()
    
    # Track results
    random_results = []
    rational_results = []
    game_counts = []
    
    # Train and test for each interval
    total_training_games = 0
    while total_training_games <= num_games:
        if total_training_games == 0:
            total_training_games = games_per_interval
            continue
            
        print(f"\nTraining AI with strategy {strategy_param:.1f} for {total_training_games} games...")
        # Train AI against itself for the total accumulated games
        ai.knowledge_base = {}  # Ensure we start with empty knowledge
        train_ai_self_play(ai, total_training_games)
        
        # Test against both opponents
        print(f"Testing against random opponent...")
        random_win_rate = test_ai_performance(ai, random_opponent, test_games)
        print(f"Testing against rational opponent...")
        rational_win_rate = test_ai_performance(ai, rational_opponent, test_games)
        
        random_results.append(random_win_rate)
        rational_results.append(rational_win_rate)
        game_counts.append(total_training_games)
        
        print(f"Strategy {strategy_param:.1f}: Total Training Games {total_training_games}")
        print(f"Random Opponent Win Rate: {random_win_rate:.2f}")
        print(f"Rational Opponent Win Rate: {rational_win_rate:.2f}")
        
        # Save the model after each training interval
        ai.save_knowledge()
        
        # Increment total training games
        total_training_games += games_per_interval
    
    return game_counts, random_results, rational_results

def main():
    # Parameters
    strategy_params = np.arange(0, 1.1, 0.1)
    num_games = 1000
    games_per_interval = 100
    test_games = 1000
    
    # Create plots directory
    os.makedirs('analysis_plots', exist_ok=True)
    
    # Analyze each strategy parameter
    for strategy in strategy_params:
        print(f"\nAnalyzing strategy parameter: {strategy:.1f}")
        game_counts, random_results, rational_results = analyze_ai_performance(
            strategy, num_games, games_per_interval, test_games
        )
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(game_counts, random_results, 'b-', label='vs Random Opponent')
        plt.plot(game_counts, rational_results, 'r-', label='vs Rational Opponent')
        plt.title(f'AI Win Rate vs Training Games (Strategy Parameter = {strategy:.1f})')
        plt.xlabel('Number of Training Games')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.savefig(f'analysis_plots/strategy_{strategy:.1f}_analysis.png')
        plt.close()

if __name__ == "__main__":
    main() 