from mcts import MCTS
import copy
import time

def play_game(ai1, ai2):
    """Play a single game between two AI players"""
    board = [[0] * 3 for _ in range(3)]
    current_player = 1
    
    while True:
        # Get current AI's move
        current_ai = ai1 if current_player == 1 else ai2
        move = current_ai.get_best_move(board)
        
        # Make the move
        board[move[0]][move[1]] = current_player
        
        # Check for win
        if check_win(board, current_player):
            return current_player
        
        # Check for draw
        if is_board_full(board):
            return 0
        
        # Switch players
        current_player = 3 - current_player

def check_win(board, player):
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

def is_board_full(board):
    return all(all(cell != 0 for cell in row) for row in board)

def train_ai(num_games=1000):
    """Train the AI by having it play against itself"""
    print("Starting AI training...")
    start_time = time.time()
    
    # Create two AI instances (they share the same knowledge base)
    ai = MCTS(iterations=80, strategy=0.1)
    
    # Load existing knowledge if available
    ai.load_knowledge()
    
    # Training loop
    for game in range(num_games):
        if (game + 1) % 100 == 0:
            print(f"Completed {game + 1} games...")
        
        # Play a game
        result = play_game(ai, ai)
        
        # Save knowledge periodically
        if (game + 1) % 100 == 0:
            ai.save_knowledge()
    
    # Final save
    ai.save_knowledge()
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    print(f"Final knowledge base size: {len(ai.knowledge_base)} states")

if __name__ == "__main__":
    train_ai(num_games=500)  # Train for 1000 games 