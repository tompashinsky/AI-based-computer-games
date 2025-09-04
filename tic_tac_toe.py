import pygame
import sys
import pygame.gfxdraw
from mcts import MCTS

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 850
LINE_WIDTH = 5
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = 600 // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIGHT_BLUE = (0, 220, 250)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")
screen.fill(WHITE)

# Load graphics
wooden_background = pygame.image.load("wood_background.png").convert()
wooden_background = pygame.transform.scale(wooden_background, (WIDTH, HEIGHT))

# Load sound effects
victory_sound = pygame.mixer.Sound("goodresult-82807.mp3")
lose_sound = pygame.mixer.Sound("losing-horn-313723.mp3")

# Initialize MCTS AI
ai = MCTS(iterations=1500)
ai.load_knowledge()

# Board and scores
board = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
player_x_score = 0
player_o_score = 0


# ------------------- GRAPHICAL FUNCTIONS -------------------
def draw_board():
    board_rect = pygame.Rect(100, 100, 600, 600)

    # Shadow
    pygame.draw.rect(screen, (0, 0, 10), board_rect.move(5, 5), border_radius=20)

    # Main board (lighter wood)
    pygame.draw.rect(screen, (245, 245, 170), board_rect, border_radius=20)

    # Grid lines
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, BLACK,
                         (100, 100 + row * SQUARE_SIZE),
                         (700, 100 + row * SQUARE_SIZE), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, BLACK,
                         (100 + col * SQUARE_SIZE, 100),
                         (100 + col * SQUARE_SIZE, 700), LINE_WIDTH)


def draw_x(row, col):
    start_x = col * SQUARE_SIZE + 100
    start_y = row * SQUARE_SIZE + 100
    end_x = start_x + SQUARE_SIZE
    end_y = start_y + SQUARE_SIZE

    # Dark red X
    pygame.draw.line(screen, (180, 0, 0),
                     (start_x + 35, start_y + 35),
                     (end_x - 35, end_y - 35), CROSS_WIDTH)
    pygame.draw.line(screen, (180, 0, 0),
                     (start_x + 35, end_y - 35),
                     (end_x - 35, start_y + 35), CROSS_WIDTH)


def draw_o(row, col):
    center = (col * SQUARE_SIZE + SQUARE_SIZE // 2 + 100,
              row * SQUARE_SIZE + SQUARE_SIZE // 2 + 100)
    # Blue O
    pygame.draw.circle(screen, (100, 120, 255), (center[0], center[1]), CIRCLE_RADIUS, CIRCLE_WIDTH)


# ---------------------------------------------------------------


def player_choice_screen():
    screen.fill(WHITE)
    font = pygame.font.Font(None, 60)
    prompt = font.render("Choose your symbol:", True, BLACK)
    screen.blit(prompt, prompt.get_rect(center=(WIDTH // 2, HEIGHT // 3)))

    button_font = pygame.font.Font(None, 50)
    x_button = pygame.Rect(WIDTH // 2 - 160, HEIGHT // 2, 100, 60)
    o_button = pygame.Rect(WIDTH // 2 + 60, HEIGHT // 2, 100, 60)

    pygame.draw.rect(screen, WHITE, x_button)
    pygame.draw.rect(screen, BLACK, x_button, 3)
    pygame.draw.rect(screen, WHITE, o_button)
    pygame.draw.rect(screen, BLACK, o_button, 3)

    x_text = button_font.render("X", True, BLACK)
    o_text = button_font.render("O", True, BLACK)

    screen.blit(x_text, x_text.get_rect(center=x_button.center))
    screen.blit(o_text, o_text.get_rect(center=o_button.center))

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                if x_button.collidepoint(mouseX, mouseY):
                    return 1  # Human plays X
                elif o_button.collidepoint(mouseX, mouseY):
                    return 2  # Human plays O


def mark_square(row, col, player):
    board[row][col] = player


def is_square_available(row, col):
    return board[row][col] == 0


def check_win(player):
    for row in range(BOARD_ROWS):
        if all([board[row][col] == player for col in range(BOARD_COLS)]):
            return True
    for col in range(BOARD_COLS):
        if all([board[row][col] == player for row in range(BOARD_ROWS)]):
            return True
    if all([board[i][i] == player for i in range(BOARD_ROWS)]) or all(
            [board[i][BOARD_ROWS - i - 1] == player for i in range(BOARD_ROWS)]):
        return True
    return False


def display_turn(player, human_symbol):
    turn_rect = pygame.Rect(30, 35, 270, 40)
    pygame.draw.rect(screen, (0, 175, 0), turn_rect, border_radius=10)
    font = pygame.font.Font(None, 40)
    current_player = "Human" if player == 1 else "AI"
    symbol = "X" if (player == 1 and human_symbol == 1) or (player == 2 and human_symbol == 2) else "O"
    text = font.render(f"{current_player}'s Turn ({symbol})", True, WHITE)
    text_rect = text.get_rect(center=turn_rect.center)
    screen.blit(text, text_rect)


def display_scores():
    global player_x_score, player_o_score
    font = pygame.font.Font(None, 40)
    line1 = "Total score"
    line2 = f"Player X : {player_x_score}  |  Player O : {player_o_score}"
    text1 = font.render(line1, True, BLACK)
    text2 = font.render(line2, True, BLACK)
    text1_rect = text1.get_rect(center=(WIDTH // 2, HEIGHT - 100))
    text2_rect = text2.get_rect(center=(WIDTH // 2, HEIGHT - 60))
    pygame.draw.rect(screen, WHITE, (0, HEIGHT - 120, WIDTH, 80))
    screen.blit(text1, text1_rect)
    screen.blit(text2, text2_rect)


def display_winner_message(player, result, human_symbol):
    global player_x_score, player_o_score

    # Update score
    if result == 1:
        if (player == 1 and human_symbol == 1) or (player == 2 and human_symbol == 2):
            player_x_score += 1
        else:
            player_o_score += 1

    # Record the game result for AI learning
    if player == 2:  # If AI was playing
        ai.record_game_result(1 if result == 1 else 0)

    # --- Stylish overlay ---
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)   # semi-transparent
    overlay.fill((0, 0, 0))  # dark background
    screen.blit(overlay, (0, 0))

    # Winner / draw text
    font = pygame.font.Font(None, 90)
    if result == 0:
        text = font.render("It's a Draw!", True, (255, 215, 0))  # Gold
    else:
        winner = "Human" if player == 1 else "AI"
        symbol = "X" if (player == 1 and human_symbol == 1) or (player == 2 and human_symbol == 2) else "O"
        text = font.render(f"{winner} ({symbol}) Wins!", True, (0, 150, 255))  # Blue
        if winner == 'Human':
            victory_sound.play()
        else:
            lose_sound.play()
    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100)))

    # Show scores
    score_font = pygame.font.Font(None, 60)
    score_text = score_font.render(f"Score: X = {player_x_score} | O = {player_o_score}", True, (255, 255, 255))
    screen.blit(score_text, score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30)))

    # Button rectangles
    restart_rect = pygame.Rect(WIDTH // 2 - 220, HEIGHT // 2 + 60, 180, 60)
    quit_rect = pygame.Rect(WIDTH // 2 + 40, HEIGHT // 2 + 60, 180, 60)

    button_font = pygame.font.Font(None, 50)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Restart button (hover effect)
        if restart_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (255, 255, 255), restart_rect, border_radius=12)
            pygame.draw.rect(screen, (200, 200, 200), restart_rect, 3, border_radius=12)
            restart_text = button_font.render("Restart", True, (0, 0, 0))
        else:
            pygame.draw.rect(screen, (50, 50, 50), restart_rect, border_radius=12)
            pygame.draw.rect(screen, (200, 200, 200), restart_rect, 3, border_radius=12)
            restart_text = button_font.render("Restart", True, (255, 255, 255))
        screen.blit(restart_text, restart_text.get_rect(center=restart_rect.center))

        # Quit button (hover effect)
        if quit_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (255, 255, 255), quit_rect, border_radius=12)
            pygame.draw.rect(screen, (200, 200, 200), quit_rect, 3, border_radius=12)
            quit_text = button_font.render("Quit", True, (0, 0, 0))
        else:
            pygame.draw.rect(screen, (50, 50, 50), quit_rect, border_radius=12)
            pygame.draw.rect(screen, (200, 200, 200), quit_rect, 3, border_radius=12)
            quit_text = button_font.render("Quit", True, (255, 255, 255))
        screen.blit(quit_text, quit_text.get_rect(center=quit_rect.center))

        pygame.display.update()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                if restart_rect.collidepoint(mouseX, mouseY):
                    restart_game()
                    return
                elif quit_rect.collidepoint(mouseX, mouseY):
                    return "back_to_menu"



def is_board_full():
    return all([board[row][col] != 0 for row in range(BOARD_ROWS) for col in range(BOARD_COLS)])


def restart_game():
    global game_over, player, human_symbol
    screen.blit(wooden_background, (0, 0))
    draw_board()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board[row][col] = 0
    player = 1 if human_symbol == 1 else 2  # If human chose X, they start; if O, AI starts
    game_over = False


def draw_back_arrow():
    arrow_x = WIDTH - 250
    arrow_y = 35
    arrow_height = 40
    arrow_width = 220
    triangle_width = 30

    arrow_points = [
        (arrow_x, arrow_y),
        (arrow_x + arrow_width - triangle_width, arrow_y),
        (arrow_x + arrow_width, arrow_y + arrow_height // 2),
        (arrow_x + arrow_width - triangle_width, arrow_y + arrow_height),
        (arrow_x, arrow_y + arrow_height)
    ]

    pygame.draw.polygon(screen, (0, 0, 150), arrow_points)
    pygame.draw.polygon(screen, BLACK, arrow_points, 3)

    font = pygame.font.Font(None, 28)
    text = font.render("Back to Home Page", True, WHITE)
    text_rect = text.get_rect(center=(arrow_x + arrow_width / 2.1, arrow_y + arrow_height // 2))
    screen.blit(text, text_rect)

    return pygame.Rect(arrow_x, arrow_y, arrow_width, arrow_height)


def run_tic_tac_toe():
    global player, game_over, player_x_score, player_o_score, human_symbol

    human_symbol = player_choice_screen()  # 1 for X, 2 for O

    screen.blit(wooden_background, (0, 0))
    draw_board()
    player = 1 if human_symbol == 1 else 2
    game_over = False
    player_x_score = 0
    player_o_score = 0
    display_scores()

    while True:
        screen.blit(wooden_background, (0, 0))
        draw_board()

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if board[row][col] == 1:
                    if human_symbol == 1:
                        draw_x(row, col)
                    else:
                        draw_o(row, col)
                elif board[row][col] == 2:
                    if human_symbol == 1:
                        draw_o(row, col)
                    else:
                        draw_x(row, col)

        display_turn(player, human_symbol)
        display_scores()
        arrow_rect = draw_back_arrow()

        if player == 2 and not game_over:
            ai_move = ai.get_best_move(board)
            row, col = ai_move
            pygame.display.update()
            pygame.time.delay(750)
            mark_square(row, col, player)
            if human_symbol == 1:
                draw_o(row, col)
            else:
                draw_x(row, col)

            if check_win(player):
                game_over = True
                result = display_winner_message(player, 1, human_symbol)
                if result == "back_to_menu":
                    restart_game()
                    return "back_to_menu"
            elif is_board_full():
                game_over = True
                result = display_winner_message(player, 0, human_symbol)
                if result == "back_to_menu":
                    restart_game()
                    return "back_to_menu"
            else:
                player = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                if arrow_rect.collidepoint(mouseX, mouseY):
                    restart_game()
                    return "back_to_menu"

                if not game_over and player == 1:
                    clicked_row = (mouseY - 100) // SQUARE_SIZE
                    clicked_col = (mouseX - 100) // SQUARE_SIZE

                    if 0 <= clicked_row < BOARD_ROWS and is_square_available(clicked_row, clicked_col):
                        mark_square(clicked_row, clicked_col, player)
                        if human_symbol == 1:
                            draw_x(clicked_row, clicked_col)
                        else:
                            draw_o(clicked_row, clicked_col)

                        if check_win(player):
                            game_over = True
                            result = display_winner_message(player, 1, human_symbol)
                            if result == "back_to_menu":
                                restart_game()
                                return "back_to_menu"
                        elif is_board_full():
                            game_over = True
                            result = display_winner_message(player, 0, human_symbol)
                            if result == "back_to_menu":
                                restart_game()
                                return "back_to_menu"
                        else:
                            player = 2

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                restart_game()

        pygame.display.update()


if __name__ == "__main__":
    run_tic_tac_toe()
