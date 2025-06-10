import pygame
import sys
import random
import math
from typing import List, Tuple, Optional, Dict

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BUBBLE_RADIUS = 20
GRID_ROWS = 15
GRID_COLS = 8
BUBBLE_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
]
SHOOT_SPEED = 10
MATCH_THRESHOLD = 3
LOSE_THRESHOLD = 100  # Distance from right edge to lose

class Bubble:
    def __init__(self, x: int, y: int, color: Tuple[int, int, int], is_moving: bool = False):
        self.x = x
        self.y = y
        self.color = color
        self.radius = BUBBLE_RADIUS
        self.is_moving = is_moving
        self.velocity_x = 0
        self.velocity_y = 0
        self.snapped = False
        self.grid_pos = None  # Will store (row, col) position in the grid

    def draw(self, screen: pygame.Surface):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius, 2)

    def update(self):
        if self.is_moving and not self.snapped:
            self.x += self.velocity_x
            self.y += self.velocity_y

    def check_wall_collision(self):
        if self.y - self.radius <= 0 or self.y + self.radius >= SCREEN_HEIGHT:
            self.velocity_y *= -1

    def snap_to_grid(self, grid_pos: Tuple[int, int], grid_to_screen: Dict[Tuple[int, int], Tuple[int, int]]):
        self.grid_pos = grid_pos
        self.x, self.y = grid_to_screen[grid_pos]
        self.is_moving = False
        self.snapped = True

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bubble Shooter")
        self.clock = pygame.time.Clock()
        self.bubbles: List[Bubble] = []
        self.grid: Dict[Tuple[int, int], Bubble] = {}  # Maps (row, col) to Bubble
        self.grid_to_screen: Dict[Tuple[int, int], Tuple[int, int]] = {}  # Maps (row, col) to (x, y)
        self.shooter_x = BUBBLE_RADIUS * 2
        self.shooter_y = SCREEN_HEIGHT // 2
        self.current_bubble = None
        self.angle = 0
        self.game_over = False
        self.score = 0
        self.initialize_grid()
        self.initialize_bubbles()
        self.initialize_shooter()

    def initialize_grid(self):
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x = SCREEN_WIDTH - (col + 1) * BUBBLE_RADIUS * 2
                y = (row - GRID_ROWS // 2) * BUBBLE_RADIUS * 2 + SCREEN_HEIGHT // 2
                self.grid_to_screen[(row, col)] = (x, y)

    def initialize_bubbles(self):
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x, y = self.grid_to_screen[(row, col)]
                color = random.choice(BUBBLE_COLORS)
                bubble = Bubble(x, y, color)
                bubble.grid_pos = (row, col)
                self.bubbles.append(bubble)
                self.grid[(row, col)] = bubble

    def initialize_shooter(self):
        color = random.choice(BUBBLE_COLORS)
        self.current_bubble = Bubble(self.shooter_x, self.shooter_y, color)

    def shoot(self, angle: float):
        if self.current_bubble and not self.current_bubble.is_moving:
            self.current_bubble.is_moving = True
            self.current_bubble.velocity_x = math.cos(math.radians(angle)) * SHOOT_SPEED
            self.current_bubble.velocity_y = math.sin(math.radians(angle)) * SHOOT_SPEED
            self.bubbles.append(self.current_bubble)
            self.initialize_shooter()  # Create new bubble before removing the old one

    def find_nearest_grid_position(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        min_dist = float('inf')
        nearest_pos = None
        
        for (row, col), (grid_x, grid_y) in self.grid_to_screen.items():
            if (row, col) not in self.grid:  # Only consider empty grid positions
                dist = math.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_pos = (row, col)
        
        return nearest_pos if min_dist < BUBBLE_RADIUS * 2 else None

    def check_collision(self, moving_bubble: Bubble) -> Optional[Tuple[int, int]]:
        for bubble in self.bubbles:
            if not bubble.is_moving:
                distance = math.sqrt((moving_bubble.x - bubble.x) ** 2 + (moving_bubble.y - bubble.y) ** 2)
                if distance < BUBBLE_RADIUS * 2:
                    return self.find_nearest_grid_position(moving_bubble.x, moving_bubble.y)
        return None

    def check_matches(self, new_bubble: Bubble) -> List[Bubble]:
        if not new_bubble.grid_pos:
            return []
            
        matches = []
        to_check = [new_bubble]
        checked = set()

        while to_check:
            current = to_check.pop()
            if current in checked:
                continue
            checked.add(current)
            matches.append(current)

            # Check adjacent bubbles in the grid
            row, col = current.grid_pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                adj_pos = (row + dr, col + dc)
                if adj_pos in self.grid:
                    adj_bubble = self.grid[adj_pos]
                    if adj_bubble not in checked and adj_bubble.color == current.color:
                        to_check.append(adj_bubble)

        return matches if len(matches) >= MATCH_THRESHOLD else []

    def remove_bubbles(self, bubbles_to_remove: List[Bubble]):
        for bubble in bubbles_to_remove:
            if bubble.grid_pos:
                del self.grid[bubble.grid_pos]
        self.bubbles = [b for b in self.bubbles if b not in bubbles_to_remove]
        self.score += len(bubbles_to_remove)

    def check_lose_condition(self) -> bool:
        for bubble in self.bubbles:
            # Only check bubbles that have been snapped to the grid (not moving and have a grid position)
            if not bubble.is_moving and bubble.grid_pos is not None:
                if bubble.x <= self.shooter_x + LOSE_THRESHOLD:
                    return True
        return False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEMOTION and not self.game_over:
                # Update shooter angle based on mouse position
                mouse_x, mouse_y = event.pos
                dx = mouse_x - self.shooter_x
                dy = mouse_y - self.shooter_y
                self.angle = math.degrees(math.atan2(dy, dx))
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                # Shoot bubble
                self.shoot(self.angle)
        return True

    def update(self):
        if self.game_over:
            return

        for bubble in self.bubbles:
            bubble.update()
            bubble.check_wall_collision()

            if bubble.is_moving and not bubble.snapped:
                # Check for collisions with other bubbles
                collision = self.check_collision(bubble)
                if collision:
                    bubble.snap_to_grid(collision, self.grid_to_screen)
                    self.grid[collision] = bubble
                    
                    # Check for matches
                    matches = self.check_matches(bubble)
                    if matches:
                        self.remove_bubbles(matches)

        # Check for game over
        if self.check_lose_condition():
            self.game_over = True

    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # Draw bubbles
        for bubble in self.bubbles:
            bubble.draw(self.screen)
        
        # Draw shooter
        pygame.draw.circle(self.screen, (200, 200, 200), (self.shooter_x, self.shooter_y), BUBBLE_RADIUS)
        
        # Draw current bubble
        if self.current_bubble:
            self.current_bubble.draw(self.screen)
        
        # Draw angle indicator
        end_x = self.shooter_x + math.cos(math.radians(self.angle)) * 50
        end_y = self.shooter_y + math.sin(math.radians(self.angle)) * 50
        pygame.draw.line(self.screen, (255, 255, 255), (self.shooter_x, self.shooter_y), (end_x, end_y), 2)

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            font = pygame.font.Font(None, 74)
            text = font.render("Game Over!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()
