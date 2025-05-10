import pygame
import sys

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

# Board and scores
board = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
player_x_score = 0
player_o_score = 0


def draw_lines():
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, BLACK, (0, row * SQUARE_SIZE + 50), (WIDTH, row * SQUARE_SIZE + 50), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, BLACK, (col * SQUARE_SIZE, 50), (col * SQUARE_SIZE, HEIGHT), LINE_WIDTH)


def draw_x(row, col):
    start_desc = (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE + 50)
    end_desc = (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE + 50)
    start_asc = (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE + 50)
    end_asc = (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE + 50)
    pygame.draw.line(screen, BLACK, start_desc, end_desc, CROSS_WIDTH)
    pygame.draw.line(screen, BLACK, start_asc, end_asc, CROSS_WIDTH)


def draw_o(row, col):
    center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2 + 50)
    pygame.draw.circle(screen, BLACK, center, CIRCLE_RADIUS, CIRCLE_WIDTH)


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


def display_turn(player):
    pygame.draw.rect(screen, WHITE, (0, 0, WIDTH, 50))
    font = pygame.font.Font(None, 50)
    text = font.render(f"Player {'X' if player == 1 else 'O'}'s Turn", True, BLACK)
    screen.blit(text, text.get_rect(center=(WIDTH // 2, 25)))


def display_scores():
    global player_x_score
    global player_o_score
    font = pygame.font.Font(None, 40)
    score_text = f"Player X: {player_x_score}  |  Player O: {player_o_score}"
    text = font.render(score_text, True, BLACK)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT - 30))
    pygame.draw.rect(screen, WHITE, (0, HEIGHT - 50, WIDTH, 50))
    screen.blit(text, text_rect)


def display_winner_message(player, result):
    global player_x_score
    global player_o_score
    if result == 1:
        if player == 1:
            player_x_score += 1
        else:
            player_o_score += 1

    pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(0, HEIGHT // 3, WIDTH, HEIGHT // 3))

    font = pygame.font.Font(None, 74)
    if result == 0:
        text = font.render("It's a draw!", True, RED)
    else:
        msg = "X" if player == 1 else "O"
        text = font.render(f"Player {msg} wins!", True, RED)
    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30)))

    # Restart Button
    button_font = pygame.font.Font(None, 50)
    restart_text = button_font.render("Restart", True, BLACK)
    restart_rect = pygame.Rect(WIDTH // 2 - 220, HEIGHT // 2 + 20, 180, 60)
    pygame.draw.rect(screen, WHITE, restart_rect)
    pygame.draw.rect(screen, BLACK, restart_rect, 3)
    screen.blit(restart_text, restart_text.get_rect(center=restart_rect.center))

    # Quit Button
    quit_text = button_font.render("Quit", True, BLACK)
    quit_rect = pygame.Rect(WIDTH // 2 + 40, HEIGHT // 2 + 20, 180, 60)
    pygame.draw.rect(screen, WHITE, quit_rect)
    pygame.draw.rect(screen, BLACK, quit_rect, 3)
    screen.blit(quit_text, quit_text.get_rect(center=quit_rect.center))

    display_scores()
    pygame.display.update()

    while True:
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
                    return "back_to_menu"  # 🔁 triggers home screen
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

def is_board_full():
    return all([board[row][col] != 0 for row in range(BOARD_ROWS) for col in range(BOARD_COLS)])


def restart_game():
    global game_over, player
    screen.fill(WHITE)
    draw_lines()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board[row][col] = 0
    player = 1
    game_over = False


def draw_back_arrow():
    arrow_x = WIDTH - 250
    arrow_y = 5
    arrow_height = 40
    arrow_width = 220
    triangle_width = 30

    arrow_points = [
        (arrow_x + triangle_width, arrow_y),
        (arrow_x + arrow_width, arrow_y),
        (arrow_x + arrow_width, arrow_y + arrow_height),
        (arrow_x + triangle_width, arrow_y + arrow_height),
        (arrow_x, arrow_y + arrow_height // 2)
    ]

    pygame.draw.polygon(screen, (180, 180, 180), arrow_points)
    pygame.draw.polygon(screen, BLACK, arrow_points, 3)

    font = pygame.font.Font(None, 32)
    text = font.render("Back to Home Page", True, BLACK)
    text_rect = text.get_rect(center=(arrow_x + arrow_width // 2 + 10, arrow_y + arrow_height // 2))
    screen.blit(text, text_rect)

    return pygame.Rect(arrow_x, arrow_y, arrow_width, arrow_height)


def run_tic_tac_toe():
    global player, game_over, player_x_score, player_o_score
    screen.fill(WHITE)
    draw_lines()
    player = 1
    game_over = False
    player_x_score = 0
    player_o_score = 0
    display_scores()

    while True:
        screen.fill(WHITE)
        draw_lines()

        # Redraw Xs and Os
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if board[row][col] == 1:
                    draw_x(row, col)
                elif board[row][col] == 2:
                    draw_o(row, col)

        display_turn(player)
        display_scores()
        arrow_rect = draw_back_arrow()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                if arrow_rect.collidepoint(mouseX, mouseY):
                    restart_game()
                    return "back_to_menu"

                if not game_over:
                    clicked_row = (mouseY - 50) // SQUARE_SIZE
                    clicked_col = mouseX // SQUARE_SIZE

                    if 0 <= clicked_row < BOARD_ROWS and is_square_available(clicked_row, clicked_col):
                        mark_square(clicked_row, clicked_col, player)
                        if player == 1:
                            draw_x(clicked_row, clicked_col)
                        else:
                            draw_o(clicked_row, clicked_col)

                        if check_win(player):
                            game_over = True
                            result = display_winner_message(player, 1)
                            if result == "back_to_menu":
                                restart_game()
                                return "back_to_menu"
                        elif is_board_full():
                            game_over = True
                            result = display_winner_message(player, 0)
                            if result == "back_to_menu":
                                restart_game()
                                return "back_to_menu"
                        else:
                            player = 3 - player

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                restart_game()

        pygame.display.update()
