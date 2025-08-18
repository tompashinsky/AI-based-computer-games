import pygame
import sys
from tic_tac_toe import run_tic_tac_toe
from head_soccer_nn2 import run_head_soccer

pygame.init()
pygame.mixer.init()
pygame.display.set_caption('NTGames')
screen = pygame.display.set_mode((800, 850))
mainClock = pygame.time.Clock()

# Fonts
header_font = pygame.font.SysFont("arial black", 48)  # Bold gaming vibe
font = pygame.font.SysFont("consolas", 25)

# Load graphics
ball_img = pygame.image.load("football.png").convert_alpha()
ball_img = pygame.transform.scale(ball_img, (30, 30))
controller_img = pygame.image.load("game_controller.png").convert_alpha()
controller_img = pygame.transform.scale(controller_img, (70, 70))
bird_img = pygame.image.load("flappy_bird.png").convert_alpha()
bird_img = pygame.transform.scale(bird_img, (50, 50))
mario_img = pygame.image.load("mario.png").convert_alpha()
mario_img = pygame.transform.scale(mario_img, (60, 60))

# Background Particles-
particles = [{"x": i * 80, "y": i * 60, "vx": (i % 3) - 1, "vy": ((i + 1) % 3) - 1}
             for i in range(12)]
controller_particles = [{"x": i * 65, "y": i * 45, "vx": (i % 3) - 1, "vy": ((i + 1) % 3) - 1}
             for i in range(12)]
bird_particles = [{"x": i * 40, "y": i * 20, "vx": (-i % 3) - 1, "vy": ((-i + 1) % 3) - 1}
             for i in range(12)]
mario_particles = [{"x": i * 60, "y": i * 40, "vx": (i % 3) - 1, "vy": ((i + 1) % 3) + 1}
             for i in range(12)]

# Linear interpolation between two colors
def lerp_color(c1, c2, t):
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )

# Smooth gradient between colors
def draw_dynamic_background(surface, time_tick):
    width, height = surface.get_size()

    # Pick two shifting base colors (cycling gently)
    color1 = (
        50 + (time_tick // 30) % 100,   # Red slowly shifts
        80,                             # Fixed-ish Green
        150 + (time_tick // 40) % 100   # Blue slowly shifts
    )
    color2 = (
        0,
        180 + (time_tick // 50) % 60,   # Green shifts slightly
        255
    )

    # Draw vertical gradient
    for y in range(height):
        t = y / (height - 1)  # interpolation factor (0 → top, 1 → bottom)
        color = lerp_color(color1, color2, t)
        pygame.draw.line(surface, color, (0, y), (width, y))

    # Particles (same as before)
    for p in particles:
        p["x"] = (p["x"] + p["vx"]) % width
        p["y"] = (p["y"] + p["vy"]) % height
        pygame.draw.circle(surface, (255, 255, 255, 120), (int(p["x"]), int(p["y"])), 6)
        pygame.draw.circle(surface, (0, 255, 0), (int(p["x"]), int(p["y"])), 5)

    for cp in controller_particles:
        cp["x"] = (cp["x"] + cp["vx"]) % width
        cp["y"] = (cp["y"] + cp["vy"]) % height

    for bp in bird_particles:
        bp["x"] = (bp["x"] + bp["vx"]) % width
        bp["y"] = (bp["y"] + bp["vy"]) % height

    for mp in mario_particles:
        mp["x"] = (mp["x"] + mp["vx"]) % width
        mp["y"] = (mp["y"] + mp["vy"]) % height

    surface.blit(ball_img, (int(p["x"]) - ball_img.get_width() // 2, int(p["y"]) - ball_img.get_height() // 2))
    surface.blit(controller_img, (int(cp["x"]) - controller_img.get_width() // 2, int(cp["y"]) - controller_img.get_height() // 2))
    surface.blit(bird_img, (int(bp["x"]) - bird_img.get_width() // 2, int(bp["y"]) - bird_img.get_height() // 2))
    surface.blit(mario_img, (int(mp["x"]) - mario_img.get_width() // 2, int(mp["y"]) - mario_img.get_height() // 2))


def draw_text(text, font, color, surface, x, y, glow=False):
    """Draw text, optionally with glow effect"""
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect(center=(x, y))

    if glow:
        glow_surf = font.render(text, True, (255, 240, 0))
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                surface.blit(glow_surf, text_rect.move(dx, dy))
    surface.blit(text_obj, text_rect)


def main_menu():
    is_clicked = False
    pygame.mixer.music.load("Komiku_shopping_list.mp3")
    pygame.mixer.music.play(-1)
    while True:
        # Background
        time_tick = pygame.time.get_ticks() // 10
        draw_dynamic_background(screen, time_tick)

        # Headline
        draw_text("Welcome to NTGames!", header_font, (255, 125, 0),
                  screen, 400, 120, glow=True)

        mx, my = pygame.mouse.get_pos()

        button_1 = pygame.Rect(150, 360, 200, 50)
        button_2 = pygame.Rect(470, 360, 200, 50)
        button_3 = pygame.Rect(150, 460, 200, 50)
        button_4 = pygame.Rect(470, 460, 200, 50)

        if button_1.collidepoint((mx, my)) and is_clicked:
            return "tic_tac_toe"
        elif button_2.collidepoint((mx, my)) and is_clicked:
            return "head_soccer"

        # --- Buttons (dark navy with neon cyan border) ---
        button_color = (20, 30, 60)
        border_color = (0, 200, 255)
        for button in [button_1, button_2, button_3, button_4]:
            pygame.draw.rect(screen, button_color, button, border_radius=12)
            pygame.draw.rect(screen, border_color, button, 3, border_radius=12)

        draw_text('Tic Tac Toe', font, (200, 255, 255), screen, 250, 385)
        draw_text('Head Soccer', font, (200, 255, 255), screen, 570, 385)
        draw_text('Checkers', font, (200, 255, 255), screen, 250, 485)
        draw_text('Bubbles', font, (200, 255, 255), screen, 570, 485)

        is_clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pygame.mixer.music.pause()
                is_clicked = True

        pygame.display.update()
        mainClock.tick(60)


def game_controller():
    current_screen = "menu"
    while True:
        if current_screen == "menu":
            result = main_menu()
            if result == "tic_tac_toe":
                current_screen = "tic_tac_toe"
            elif result == "head_soccer":
                current_screen = "head_soccer"

        elif current_screen == "tic_tac_toe":
            result = run_tic_tac_toe()
            if result == "back_to_menu":
                current_screen = "menu"

        elif current_screen == "head_soccer":
            result = run_head_soccer()
            if result == "back_to_menu":
                current_screen = "menu"


if __name__ == "__main__":
    game_controller()
