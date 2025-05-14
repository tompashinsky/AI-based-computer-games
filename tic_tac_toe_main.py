import pygame
import sys
import pygame.gfxdraw  # Import gfxdraw for anti-aliasing
from mcts import MCTS  # Import the MCTS AI

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 850
LINE_WIDTH = 15
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")
screen.fill(WHITE)

# Initialize MCTS AI
ai = MCTS(iterations=1000)
ai.load_knowledge()  # Load the trained model

# Board
board = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]

# Draw lines
def draw_lines():
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, BLACK, (0, row * SQUARE_SIZE + 50), (WIDTH, row * SQUARE_SIZE + 50), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, BLACK, (col * SQUARE_SIZE, 50), (col * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

# Draw X
def draw_x(row, col):
    start_desc = (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE + 50)  # Adjusted for extra space
    end_desc = (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE + 50)  # Adjusted
    start_asc = (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE + 50)  # Adjusted
    end_asc = (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE + 50)  # Adjusted
    pygame.draw.line(screen, BLACK, start_desc, end_desc, CROSS_WIDTH)
    pygame.draw.line(screen, BLACK, start_asc, end_asc, CROSS_WIDTH)

# Draw O
def draw_o(row, col):
    center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2 + 50)  # Adjusted for extra space
    pygame.draw.circle(screen, BLACK, center, CIRCLE_RADIUS, CIRCLE_WIDTH)

# Mark square
def mark_square(row, col, player):
    board[row][col] = player

# Check if square is available
def is_square_available(row, col):
    return board[row][col] == 0

# Check for win
def check_win(player):
    # Check rows
    for row in range(BOARD_ROWS):
        if all([board[row][col] == player for col in range(BOARD_COLS)]):
            return True
    # Check columns
    for col in range(BOARD_COLS):
        if all([board[row][col] == player for row in range(BOARD_ROWS)]):
            return True
    # Check diagonals
    if all([board[i][i] == player for i in range(BOARD_ROWS)]) or all([board[i][BOARD_ROWS - i - 1] == player for i in range(BOARD_ROWS)]):
        
        return True
    return False


# Function to display the current player's turn
def display_turn(player):
    # Clear the top area
    pygame.draw.rect(screen, WHITE, (0, 0, WIDTH, 50))
    
    # Display the player's turn
    font = pygame.font.Font(None, 50)
    if player == 1:
        text = font.render("Player X's Turn", True, BLACK)
    else:
        text = font.render("Player O's Turn", True, BLACK)
    text_rect = text.get_rect(center=(WIDTH // 2, 25))
    screen.blit(text, text_rect)

# Function to display scores
def display_scores():
    global player_x_score
    global player_o_score
    font = pygame.font.Font(None, 40)
    score_text = f"Player X: {player_x_score}  |  Player O: {player_o_score}"
    text = font.render(score_text, True, BLACK)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - 30))  # Display at the bottom of the screen
    pygame.draw.rect(screen, WHITE, (0, HEIGHT - 50, WIDTH, 50))  # Clear the score area
    screen.blit(text, text_rect)

# Display winner message
def display_winner_message(player, result):
    global player_x_score
    global player_o_score
    # Update scores
    if result == 1:  # If there's a winner
        if player == 1:
            player_x_score += 1
        else:
            player_o_score += 1
    
    # Record the game result for AI learning
    if player == 2:  # If AI was playing
        ai.record_game_result(1 if result == 1 else 0)
    
    # Draw a grey background
    background_rect = pygame.Rect(0, HEIGHT // 3, WIDTH, HEIGHT // 3)
    pygame.draw.rect(screen, (200, 200, 200), background_rect)

    msg = ""
    if player == 1:
        msg = "X"
    else:
        msg = "O"

    # Display the winner message
    font = pygame.font.Font(None, 74)  # Create a font object
    if result == 0:
        text = font.render("It's a draw!", True, RED)  # Render the draw message
    else:
        text = font.render(f"Player {msg} wins!", True, RED)  # Render the winner message
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))  # Center the text above the button
    screen.blit(text, text_rect)  # Draw the text on the screen

    # Draw the restart button
    button_font = pygame.font.Font(None, 50)
    button_text = button_font.render("Restart", True, BLACK)
    button_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 20, 200, 60)  # Button dimensions
    pygame.draw.rect(screen, WHITE, button_rect)  # Button background
    pygame.draw.rect(screen, BLACK, button_rect, 3)  # Button border
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)

    display_scores()  # Display scores after the winner message

    pygame.display.update()  # Update the display


    # Wait for user interaction
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                if button_rect.collidepoint(mouseX, mouseY):  # Check if the button is clicked
                    restart_game()
                    return  # Exit the function and return control to the main game loop
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Clear the winner message by redrawing the board
                    screen.fill(WHITE)  # Clear the screen
                    draw_lines()  # Redraw the grid lines
                    for row in range(BOARD_ROWS):  # Redraw the current board state
                        for col in range(BOARD_COLS):
                            if board[row][col] == 1:
                                draw_x(row, col)
                            elif board[row][col] == 2:
                                draw_o(row, col)
                    display_scores() # Redraw the scores
                    pygame.display.update()  # Update the display
                    return
                    return

# Check if board is full
def is_board_full():
    return all([board[row][col] != 0 for row in range(BOARD_ROWS) for col in range(BOARD_COLS)])

# Restart game
def restart_game():
    global game_over
    global player
    screen.fill(WHITE)
    draw_lines()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board[row][col] = 0
    player = 1
    game_over = False
    return

# Main loop
draw_lines()
global player
global game_over
player = 1  # Human player is X (1)
game_over = False
global player_x_score
global player_o_score
player_x_score = 0
player_o_score = 0
display_scores()  # Display initial scores

while True:
    display_turn(player)  # Display the current player's turn
    display_scores()  # Display scores
    
    # AI's turn (player 2)
    if player == 2 and not game_over:
        # Get AI move
        ai_move = ai.get_best_move(board)
        row, col = ai_move
        
        # Make the move
        mark_square(row, col, player)
        draw_o(row, col)
        
        # Check for win or draw
        if check_win(player):
            game_over = True
            display_winner_message(player, 1)
        elif is_board_full():
            game_over = True
            display_winner_message(player, 0)
        else:
            player = 3 - player  # Switch to human player
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over and player == 1:  # Only human player can click
            mouseX, mouseY = event.pos
            clicked_row = (mouseY - 50) // SQUARE_SIZE
            clicked_col = mouseX // SQUARE_SIZE

            if 50 <= mouseY <= HEIGHT and is_square_available(clicked_row, clicked_col):
                mark_square(clicked_row, clicked_col, player)
                draw_x(clicked_row, clicked_col)

                if check_win(player):
                    game_over = True
                    display_winner_message(player, 1)
                elif is_board_full():
                    game_over = True
                    display_winner_message(player, 0)
                else:
                    player = 3 - player  # Switch to AI player

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart_game()

    pygame.display.update()