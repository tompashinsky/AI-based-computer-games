import numpy as np
import matplotlib.pyplot as plt
from mcts import MCTS
import random
import copy
import os
import argparse

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
            return 0.5  # Draw
        
        board[move[0]][move[1]] = current_player
        winner = check_winner(board)
        
        if winner is not None:
            if winner == 0:
                return 0.5  # Draw
            # Return 1 for win, 0 for loss, 0.5 for draw
            if (winner == 1 and ai_first) or (winner == 2 and not ai_first):
                return 1.0  # AI wins
            return 0.0  # AI loses
        
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
    draws = 0
    for _ in range(num_games):
        ai_first = random.choice([True, False])
        result = play_game(ai, opponent, ai_first)
        if result == 1.0:
            wins += 1
        elif result == 0.5:
            draws += 1
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    print(f"Wins: {wins}, Draws: {draws}, Losses: {num_games - wins - draws}")
    return win_rate, draw_rate

def test_iterations(training_games=1000, test_games=1000):
    # Create plots directory
    os.makedirs('iteration_analysis_plots', exist_ok=True)
    
    # Test iterations from 1 to 100 in steps of 10
    iterations = range(1, 101, 10)
    
    # Initialize opponents
    random_opponent = RandomOpponent()
    rational_opponent = RationalOpponent()
    
    # Track results
    random_results = []
    rational_results = []
    
    for iteration_count in iterations:
        print(f"\nTesting with {iteration_count} iterations...")
        
        # Initialize AI
        ai = MCTS(iterations=iteration_count)
        
        # Train AI against itself
        print(f"\nTraining for {training_games} games...")
        ai.knowledge_base = {}  # Start with empty knowledge
        train_ai_self_play(ai, training_games)
        
        # Test against both opponents with 1000 games each
        print(f"\nTesting against random opponent (1000 games)...")
        random_win_rate, random_draw_rate = test_ai_performance(ai, random_opponent, test_games)
        print(f"\nTesting against rational opponent (1000 games)...")
        rational_win_rate, rational_draw_rate = test_ai_performance(ai, rational_opponent, test_games)
        
        random_results.append(random_win_rate)
        rational_results.append(rational_win_rate)
        
        print(f"\nIterations {iteration_count}: Training Games {training_games}")
        print(f"Random Opponent Win Rate: {random_win_rate:.2f}, Draw Rate: {random_draw_rate:.2f}")
        print(f"Rational Opponent Win Rate: {rational_win_rate:.2f}, Draw Rate: {rational_draw_rate:.2f}")
    
    # Create a single plot comparing win rates against iteration counts
    plt.figure(figsize=(12, 8))
    
    plt.plot(iterations, random_results, 'b-o', label='vs Random')
    plt.plot(iterations, rational_results, 'r-o', label='vs Rational')
    
    plt.title(f'Win Rates vs MCTS Iterations (Trained on {training_games} games)')
    plt.xlabel('Number of MCTS Iterations')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    
    plt.savefig('iteration_analysis_plots/win_rates_vs_iterations.png')
    plt.close()

if __name__ == "__main__":
    test_iterations() 