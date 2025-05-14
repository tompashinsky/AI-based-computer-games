import pygame
import sys
from tic_tac_toe import run_tic_tac_toe

pygame.init()
pygame.display.set_caption('NTGames')
screen = pygame.display.set_mode((800, 850))
mainClock = pygame.time.Clock()
header_font = pygame.font.SysFont(None, 70)
font = pygame.font.SysFont(None, 30)


def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)


def main_menu():
    is_clicked = False
    while True:
        screen.fill((0, 220, 250))
        draw_text('Welcome to NTGames!', header_font, (0, 0, 0), screen, 140, 140)

        mx, my = pygame.mouse.get_pos()

        button_1 = pygame.Rect(150, 360, 200, 50)
        button_2 = pygame.Rect(470, 360, 200, 50)
        button_3 = pygame.Rect(310, 460, 200, 50)

        if button_1.collidepoint((mx, my)) and is_clicked:
            return "tic_tac_toe"

        pygame.draw.rect(screen, (255, 0, 0), button_1, border_radius=10)
        pygame.draw.rect(screen, (255, 0, 0), button_2, border_radius=10)
        pygame.draw.rect(screen, (255, 0, 0), button_3, border_radius=10)

        draw_text('Tic Tac Toe', font, (255, 255, 255), screen, 197, 377)
        draw_text('Checkers', font, (255, 255, 255), screen, 525, 377)
        draw_text('Bubbles', font, (255, 255, 255), screen, 372, 477)

        is_clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                is_clicked = True

        pygame.display.update()
        mainClock.tick(60)


# The game_controller function helps to navigate between the games and the main screen
def game_controller():
    current_screen = "menu"
    while True:
        if current_screen == "menu":
            result = main_menu()
            if result == "tic_tac_toe":
                current_screen = "tic_tac_toe"

        elif current_screen == "tic_tac_toe":
            result = run_tic_tac_toe()
            if result == "back_to_menu":
                current_screen = "menu"


if __name__ == "__main__":
    game_controller()
