import pygame
import random

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 850
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
FPS = 60
GAME_DURATION = 180  # 3 minutes

# Set up screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
background = pygame.image.load("field_image.png").convert()
background = pygame.transform.scale(background, (WIDTH, HEIGHT))
pygame.display.set_caption("Head Soccer")
clock = pygame.time.Clock()

# Load graphics
ball_img = pygame.image.load("football.png").convert_alpha()
ball_img = pygame.transform.scale(ball_img, (30, 30))

left_goal_img = pygame.image.load("left_goal.png").convert_alpha()
left_goal_img = pygame.transform.scale(left_goal_img, (80, 200))
right_goal_img = pygame.image.load("right_goal.png").convert_alpha()
right_goal_img = pygame.transform.scale(right_goal_img, (80, 200))

left_player_img = pygame.image.load("left_player.png").convert_alpha()
left_player_img = pygame.transform.scale(left_player_img, (50, 90))
right_player_img = pygame.image.load("right_player.png").convert_alpha()
right_player_img = pygame.transform.scale(right_player_img, (50, 90))

left_player_kick_img = pygame.image.load("left_player_kick.png").convert_alpha()
left_player_kick_img = pygame.transform.scale(left_player_kick_img, (50, 90))
right_player_kick_img = pygame.image.load("right_player_kick.png").convert_alpha()
right_player_kick_img = pygame.transform.scale(right_player_kick_img, (50, 90))

# Player Class
class Player(pygame.sprite.Sprite):
    def __init__(self, x, idle_path, kick_path):
        super().__init__()
        self.image_idle = pygame.transform.scale(pygame.image.load(idle_path).convert_alpha(), (50, 90))
        self.image_kick = pygame.transform.scale(pygame.image.load(kick_path).convert_alpha(), (50, 90))
        self.image = self.image_idle
        self.rect = self.image.get_rect()
        self.rect.midbottom = (x, HEIGHT - 20)
        self.vel_y = 0
        self.on_ground = True
        self.kick_timer = 0

    def update(self, keys, left_key, right_key, jump_key, kick_key=None):
        if keys[left_key]:
            self.rect.x -= 5
        if keys[right_key]:
            self.rect.x += 5
        if keys[jump_key] and self.on_ground:
            self.vel_y = -12
            self.on_ground = False
        if kick_key and keys[kick_key]:
            self.kick_timer = 5  # show kick image for a few frames

        self.vel_y += 0.5
        self.rect.y += self.vel_y

        # Boundaries
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH

        # Ground collision
        if self.rect.bottom >= HEIGHT - 20:
            self.rect.bottom = HEIGHT - 20
            self.vel_y = 0
            self.on_ground = True

        # Handle kick image
        if self.kick_timer > 0:
            self.image = self.image_kick
            self.kick_timer -= 1
        else:
            self.image = self.image_idle


# Ball Class
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = ball_img
        self.rect = self.image.get_rect(center=(WIDTH//2, HEIGHT//2))
        self.vel = pygame.Vector2(random.choice([-4, 4]), -5)

    # Update the location of the ball on the screen
    def update(self):
        self.vel.y += 0.5  # gravity
        self.rect.x += int(self.vel.x)
        self.rect.y += int(self.vel.y)

        if self.rect.left < 0 or self.rect.right > WIDTH:
            self.vel.x *= -1
            self.rect.left = max(0, self.rect.left)
            self.rect.right = min(WIDTH, self.rect.right)

        if self.rect.top < 0:
            self.rect.top = 0
            self.vel.y *= -1

        if self.rect.bottom >= HEIGHT - 20:
            self.rect.bottom = HEIGHT - 20
            self.vel.y *= -0.8

# Draw the "Back to Home Page" arrow on the top right of the screen
def draw_back_arrow():
    arrow_x = WIDTH - 250
    arrow_y = 30
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

def restart_game():
    return "restart"

# Print "Goal!" every time a player gets a point
def show_goal_message():
    font = pygame.font.Font(None, 100)
    text = font.render("GOAL!", True, YELLOW)
    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
    pygame.display.flip()
    pygame.time.delay(1500)

def show_game_over_screen(score1, score2):
    font = pygame.font.Font(None, 74)
    winner = "Draw"
    if score1 > score2:
        winner = "Player 1 Wins!"
    elif score2 > score1:
        winner = "Player 2 Wins!"
    pygame.draw.rect(screen, (200, 200, 200), pygame.Rect(0, HEIGHT // 3, WIDTH, HEIGHT // 3))
    text = font.render(winner, True, (255, 0, 0))
    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50)))

    button_font = pygame.font.Font(None, 50)
    restart_rect = pygame.Rect(WIDTH // 2 - 200, HEIGHT // 2 + 20, 180, 60)
    quit_rect = pygame.Rect(WIDTH // 2 + 20, HEIGHT // 2 + 20, 180, 60)

    pygame.draw.rect(screen, WHITE, restart_rect)
    pygame.draw.rect(screen, BLACK, restart_rect, 3)
    pygame.draw.rect(screen, WHITE, quit_rect)
    pygame.draw.rect(screen, BLACK, quit_rect, 3)

    screen.blit(button_font.render("Restart", True, BLACK), restart_rect.move(30, 13))
    screen.blit(button_font.render("Quit", True, BLACK), quit_rect.move(52, 13))

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "back_to_menu"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if restart_rect.collidepoint(event.pos):
                    return "restart"
                elif quit_rect.collidepoint(event.pos):
                    return "back_to_menu"

def run_head_soccer():
    while True:
        player1 = Player(150, "left_player.png", "left_player_kick.png")
        player2 = Player(650, "right_player.png", "right_player_kick.png")
        ball = Ball()
        all_sprites = pygame.sprite.Group(player1, player2, ball)

        score1 = 0
        score2 = 0
        start_ticks = pygame.time.get_ticks()

        running = True
        while running:
            clock.tick(FPS)
            seconds = (pygame.time.get_ticks() - start_ticks) // 1000
            remaining = max(0, GAME_DURATION - seconds)
            minutes = remaining // 60
            secs = remaining % 60

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "back_to_menu"
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if draw_back_arrow().collidepoint(event.pos):
                        return "back_to_menu"

            keys = pygame.key.get_pressed()
            left_kicking = keys[pygame.K_k]
            right_kicking = keys[pygame.K_KP_PLUS]
            player1.update(keys, pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_k)
            player2.update(keys, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_KP_PLUS)
            ball.update()

            # Draw field and goals
            screen.blit(background, (0, 0))
            pygame.draw.rect(screen, WHITE, (0, HEIGHT - 20, WIDTH, 20))

            # Goal positions
            goal_width = left_goal_img.get_width()
            goal_height = left_goal_img.get_height()
            goal_y = HEIGHT - 20 - goal_height

            # Goal rectangles for collision
            left_goal_rect = pygame.Rect(0, goal_y, goal_width, goal_height)
            right_goal_rect = pygame.Rect(WIDTH - goal_width, goal_y, goal_width, goal_height)

            # Define crossbars as solid horizontal bars at the top of each goal
            crossbar_height = 10
            left_crossbar = pygame.Rect(0, goal_y, goal_width, crossbar_height)
            right_crossbar = pygame.Rect(WIDTH - goal_width, goal_y, goal_width, crossbar_height)

            # Collision with crossbars
            if ball.rect.colliderect(left_crossbar) or ball.rect.colliderect(right_crossbar):
                ball.rect.top = goal_y - ball.rect.height
                ball.vel.y *= -0.8  # bounce off with some energy loss

            # Draw goal images
            screen.blit(left_goal_img, (0, goal_y))
            screen.blit(right_goal_img, (WIDTH - goal_width, goal_y))

            # GOAL DETECTION - must be BEFORE player collision
            if ball.rect.colliderect(left_goal_rect):
                score2 += 1
                show_goal_message()
                ball.rect.center = (WIDTH // 2, HEIGHT // 2)
                ball.vel = pygame.Vector2(random.choice([-4, 4]), -5)
                player1.rect.midbottom = (150, HEIGHT - 20)
                player2.rect.midbottom = (650, HEIGHT - 20)

            elif ball.rect.colliderect(right_goal_rect):
                score1 += 1
                show_goal_message()
                ball.rect.center = (WIDTH // 2, HEIGHT // 2)
                ball.vel = pygame.Vector2(random.choice([-4, 4]), -5)
                player1.rect.midbottom = (150, HEIGHT - 20)
                player2.rect.midbottom = (650, HEIGHT - 20)

            # Collision with player 1
            if ball.rect.colliderect(player1.rect):
                if ball.rect.centerx < player1.rect.centerx:
                    ball.vel.x = -abs(ball.vel.x)  # bounce left
                else:
                    ball.vel.x = abs(ball.vel.x)  # bounce right
                ball.vel.y = -7 if not left_kicking else -10

            # Collision with player 2
            elif ball.rect.colliderect(player2.rect):
                if ball.rect.centerx < player2.rect.centerx:
                    ball.vel.x = -abs(ball.vel.x)  # bounce left
                else:
                    ball.vel.x = abs(ball.vel.x)  # bounce right
                ball.vel.y = -7 if not right_kicking else -10

            all_sprites.draw(screen)
            arrow_rect = draw_back_arrow()

            font = pygame.font.Font(None, 40)
            timer_text = font.render(f"Time Left: {minutes:02}:{secs:02}", True, WHITE)
            score_text = font.render(f"{score1} - {score2}", True, WHITE)
            screen.blit(timer_text, (30, 30))
            screen.blit(score_text, (WIDTH//2 - 30, 30))

            pygame.display.flip()

            if remaining <= 0:
                running = False

        result = show_game_over_screen(score1, score2)
        if result != "restart":
            return result

# Main execution
if __name__ == "__main__":
    run_head_soccer()
