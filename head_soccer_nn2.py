import pygame
import random
import torch
import torch.nn as nn
import numpy as np

# Initialize pygame
pygame.init()
pygame.mixer.init()

WIDTH = 800
HEIGHT = 850
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
FPS = 60
GAME_DURATION = 180

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Head Soccer with NN AI")
clock = pygame.time.Clock()

# Load graphics
background = pygame.image.load("field_image.png").convert()
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

ball_img = pygame.image.load("football.png").convert_alpha()
ball_img = pygame.transform.scale(ball_img, (30, 30))

left_goal_img = pygame.image.load("left_goal.png").convert_alpha()
left_goal_img = pygame.transform.scale(left_goal_img, (80, 200))
right_goal_img = pygame.image.load("right_goal.png").convert_alpha()
right_goal_img = pygame.transform.scale(right_goal_img, (80, 200))

left_player_img = pygame.image.load("left_player.png").convert_alpha()
left_player_img = pygame.transform.scale(left_player_img, (50, 90))
left_player_kick_img = pygame.image.load("left_player_kick.png").convert_alpha()
left_player_kick_img = pygame.transform.scale(left_player_kick_img, (50, 90))

right_player_img = pygame.image.load("right_player.png").convert_alpha()
right_player_img = pygame.transform.scale(right_player_img, (50, 90))
right_player_kick_img = pygame.image.load("right_player_kick.png").convert_alpha()
right_player_kick_img = pygame.transform.scale(right_player_kick_img, (50, 90))

goal_img = pygame.image.load("left_goal.png").convert_alpha()
goal_img = pygame.transform.scale(goal_img, (80, 200))

goal_sound = pygame.mixer.Sound("goal_sound.mp3")
whistle_sound = pygame.mixer.Sound("whistle_sound.mp3")
ending_whistle_sound = pygame.mixer.Sound("zapsplat_sport_whistle_soccer_full_time_002_76140.mp3")
kick_sound = pygame.mixer.Sound("kick_sound.mp3")

ACTIONS = ['left', 'right', 'jump', 'kick', 'stay']


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTIONS))
        )

    def forward(self, x):
        return self.fc(x)


class Player(pygame.sprite.Sprite):
    def __init__(self, x, idle_img, kick_img):
        super().__init__()
        self.image_idle = idle_img
        self.image_kick = kick_img
        self.image = self.image_idle
        self.rect = self.image.get_rect()
        self.rect.midbottom = (x, HEIGHT - 20)
        self.vel_y = 0
        self.on_ground = True
        self.kick_timer = 0

    def apply_action(self, action):
        if action == 'left':
            self.rect.x -= 5
        elif action == 'right':
            self.rect.x += 5
        elif action == 'jump' and self.on_ground:
            self.vel_y = -12
            self.on_ground = False
        elif action == 'kick':
            self.kick_timer = 5
        elif action == 'stay':
            self.rect.x += 0

        self.vel_y += 0.5
        self.rect.y += self.vel_y

        if self.rect.left < 40: self.rect.left = 40
        if self.rect.right > WIDTH - 40: self.rect.right = WIDTH - 40
        if self.rect.bottom >= HEIGHT - 20:
            self.rect.bottom = HEIGHT - 20
            self.vel_y = 0
            self.on_ground = True

        if self.kick_timer > 0:
            self.image = self.image_kick
            self.kick_timer -= 1
        else:
            self.image = self.image_idle

    def update_keyboard(self, keys, left, right, jump, kick):
        action = None
        if keys[left]:
            self.rect.x -= 5
        if keys[right]:
            self.rect.x += 5
        if keys[jump] and self.on_ground:
            self.vel_y = -12
            self.on_ground = False
        if keys[kick]:
            self.kick_timer = 5

        self.vel_y += 0.5
        self.rect.y += self.vel_y

        if self.rect.left < 40: self.rect.left = 40
        if self.rect.right > WIDTH - 40: self.rect.right = WIDTH - 40
        if self.rect.bottom >= HEIGHT - 20:
            self.rect.bottom = HEIGHT - 20
            self.vel_y = 0
            self.on_ground = True

        if self.kick_timer > 0:
            self.image = self.image_kick
            self.kick_timer -= 1
        else:
            self.image = self.image_idle


class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = ball_img
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.vel = pygame.Vector2(random.choice([-4, 4]), -5)

    def update(self):
        self.vel.y += 0.5
        self.rect.x += int(self.vel.x)
        self.rect.y += int(self.vel.y)

        if self.rect.left < 0 or self.rect.right > WIDTH:
            self.vel.x *= -1
        if self.rect.top < 0:
            self.rect.top = 0
            self.vel.y *= -1
        if self.rect.bottom >= HEIGHT - 20:
            self.rect.bottom = HEIGHT - 20
            self.vel.y *= -0.85


def get_nn_action(model, player, ball, opponent):
    state = np.array([
        # --- AI Player position ---
        (player.rect.centerx - 40) / (WIDTH - 80),
        player.rect.centery / HEIGHT,

        # --- Ball position and velocity ---
        (ball.rect.centerx - 40) / (WIDTH - 80),
        ball.rect.centery / HEIGHT,
        ball.vel.x / 10,
        ball.vel.y / 10,

        # --- Opponent position ---
        (opponent.rect.centerx - 40) / (WIDTH - 80),
        opponent.rect.centery / HEIGHT,  # added vertical position

        # --- Ball relative to AI ---
        (ball.rect.centerx - player.rect.centerx) / (WIDTH - 80),  # ahead/behind
        (player.rect.centery - ball.rect.centery) / HEIGHT,  # vertical distance
        abs(ball.rect.centerx - WIDTH - 80) / (WIDTH - 80),  # distance to AI's goal

        # --- Ball movement flags ---
        1.0 if ball.vel.x > 0 else 0.0,  # is ball moving toward AI goal
        1.0 if player.on_ground else 0.0,  # AI on ground
        1.0 if ball.rect.centerx > WIDTH - 200 and ball.vel.x > 0 else 0.0,  # danger zone

        # --- Opponent relative info ---
        (opponent.rect.centerx - player.rect.centerx) / (WIDTH - 80),  # horizontal distance to opponent
        (opponent.rect.centery - player.rect.centery) / HEIGHT  # vertical distance to opponent
    ], dtype=np.float32)

    state_tensor = torch.tensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
        action_idx = q_values.argmax().item()
    return ACTIONS[action_idx]


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

def pre_game_countdown():
    font = pygame.font.Font(None, 100)

    messages = ["Are you ready?", "3", "2", "1", "Let's go!"]
    for msg in messages:
        screen.blit(background, (0, 0))
        pygame.draw.rect(screen, WHITE, (0, HEIGHT - 20, WIDTH, 20))
        text = font.render(msg, True, YELLOW)
        screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        pygame.display.flip()
        pygame.time.delay(1000)  # Show each message for 1 second


def restart_game():
    return "restart"

# Print "Goal!" every time a player gets a point
def show_goal_message():
    font = pygame.font.Font(None, 100)
    text = font.render("GOAL!", True, YELLOW)
    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
    pygame.display.flip()
    pygame.time.delay(4500)

def show_game_over_screen(score1, score2):
    font = pygame.font.Font(None, 90)
    score_font = pygame.font.Font(None, 70)
    button_font = pygame.font.Font(None, 50)

    # Determine winner and color
    winner = "It's a Draw!"
    color = (255, 215, 0)  # Gold for draw
    if score1 > score2:
        winner = "Player 1 Wins!"
        color = (0, 200, 0)  # Green
    elif score2 > score1:
        winner = "Player 2 Wins!"
        color = (0, 100, 255)  # Blue

    # Dark overlay background
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)  # Transparency
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    # Winner text
    text = font.render(winner, True, color)
    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 120)))

    # Final score text
    score_text = score_font.render(f"{score1} - {score2}", True, (255, 255, 255))
    screen.blit(score_text, score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40)))

    # Button rectangles
    restart_rect = pygame.Rect(WIDTH // 2 - 200, HEIGHT // 2 + 60, 180, 60)
    quit_rect = pygame.Rect(WIDTH // 2 + 20, HEIGHT // 2 + 60, 180, 60)

    running = True
    while running:
        # Draw buttons (with hover effect)
        mouse_pos = pygame.mouse.get_pos()

        for rect, label in [(restart_rect, "Restart"), (quit_rect, "Quit")]:
            if rect.collidepoint(mouse_pos):
                pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=12)   # white fill
                pygame.draw.rect(screen, (200, 200, 200), rect, 3, border_radius=12)  # gray border
                text_surf = button_font.render(label, True, (0, 0, 0))
            else:
                pygame.draw.rect(screen, (50, 50, 50), rect, border_radius=12)  # dark fill
                pygame.draw.rect(screen, (200, 200, 200), rect, 3, border_radius=12)
                text_surf = button_font.render(label, True, (255, 255, 255))

            screen.blit(text_surf, text_surf.get_rect(center=rect.center))

        pygame.display.update()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "back_to_menu"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if restart_rect.collidepoint(event.pos):
                    return "restart"
                elif quit_rect.collidepoint(event.pos):
                    return "back_to_menu"




def run_head_soccer():
    model = QNet()
    model.load_state_dict(torch.load("nn_model.pth"))
    model.eval()

    player1 = Player(150, left_player_img, left_player_kick_img)
    player2 = Player(650, right_player_img, right_player_kick_img)
    ball = Ball()
    all_sprites = pygame.sprite.Group(player1, player2, ball)

    score1 = 0
    score2 = 0
    pre_game_countdown()
    start_ticks = pygame.time.get_ticks()
    whistle_sound.play()

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
        player1.update_keyboard(keys, pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_k)

        # Player 2 is controlled by NN
        action = get_nn_action(model, player2, ball, player1)
        player2.apply_action(action)

        ball.update()

        # Player-player collision
        if player1.rect.colliderect(player2.rect):
            if player1.rect.centerx < player2.rect.centerx:
                overlap = (player1.rect.right - player2.rect.left) // 2
                player1.rect.x -= overlap
                player2.rect.x += overlap
            else:
                overlap = (player2.rect.right - player1.rect.left) // 2
                player1.rect.x += overlap
                player2.rect.x -= overlap


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
            ball.vel.y *= -0.85  # bounce off with some energy loss

        # Draw goal images
        screen.blit(left_goal_img, (0, goal_y))
        screen.blit(right_goal_img, (WIDTH - goal_width, goal_y))

        # GOAL DETECTION - must be BEFORE player collision
        if ball.rect.colliderect(left_goal_rect):
            score2 += 1
            goal_sound.play()
            show_goal_message()
            whistle_sound.play()
            ball.rect.center = (WIDTH // 2, HEIGHT // 2)
            ball.vel = pygame.Vector2(random.choice([-4, 4]), -5)
            player1.rect.midbottom = (150, HEIGHT - 20)
            player2.rect.midbottom = (650, HEIGHT - 20)

        elif ball.rect.colliderect(right_goal_rect):
            score1 += 1
            goal_sound.play()
            show_goal_message()
            whistle_sound.play()
            ball.rect.center = (WIDTH // 2, HEIGHT // 2)
            ball.vel = pygame.Vector2(random.choice([-4, 4]), -5)
            player1.rect.midbottom = (150, HEIGHT - 20)
            player2.rect.midbottom = (650, HEIGHT - 20)

        # Track collision state outside the loop
        if not hasattr(ball, "colliding_with_player"):
            ball.colliding_with_player = False

        collision = False
        for p, is_kicking in [(player1, keys[pygame.K_k]), (player2, action == "kick")]:
            if ball.rect.colliderect(p.rect):
                if not ball.colliding_with_player:  # <-- Only play sound on new collision
                    kick_sound.play()
                if ball.rect.centerx < p.rect.centerx:
                    ball.vel.x = -abs(ball.vel.x)
                else:
                    ball.vel.x = abs(ball.vel.x)
                ball.vel.y = -12 if is_kicking else -9
                collision = True

        ball.colliding_with_player = collision

        all_sprites.draw(screen)
        draw_back_arrow()

        font = pygame.font.Font(None, 40)
        timer_text = font.render(f"Time Left: {minutes:02}:{secs:02}", True, WHITE)
        score_text = font.render(f"{score1} - {score2}", True, WHITE)
        screen.blit(timer_text, (30, 30))
        screen.blit(score_text, (WIDTH // 2 - 30, 30))

        pygame.display.flip()

        if remaining <= 0:
            running = False

    ending_whistle_sound.play()
    result = show_game_over_screen(score1, score2)
    if result != "restart":
        return result


if __name__ == "__main__":
    run_head_soccer()
