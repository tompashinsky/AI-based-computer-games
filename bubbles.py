import pygame
import sys
import random
import math
from typing import List, Tuple, Optional, Dict
import torch
from bubbles_target_dqn import TargetDQNAgent, decode_target_action, calculate_angle_to_target
import numpy as np

# Initialize Pygame
pygame.init()

# Next Bubble Preview Feature:
# Both human and AI players can see their next bubble color, allowing for strategic planning.
# The next bubble appears as a smaller, semi-transparent preview next to the current bubble.
# This information is included in the AI's state representation for better decision making.
#
# DEBUG FEATURES:
# - Press 'D' key to toggle debug mode on/off
# - Console output shows AI decision details (target, angle, Q-values)
# - Visual debug panel shows real-time AI information
# - Red circle shows AI's current target position
# - Yellow line shows AI's shooting trajectory
# - Green line shows calculated shooting trajectory
#
# ANGLE CALCULATION:
# - NEW: Grid Hit Angle calculation that properly targets grid positions
# - Handles honeycomb grid layout and valid shooting angles (90-270°)
# - Prevents downward shooting and calculates bounce angles when needed
#
# TARGET VALIDATION:
# - NEW: Only shows targets that can actually be reached by shooting
# - Prevents AI from choosing unreachable positions that would clog the board
# - Predicts where bubbles will actually land vs where AI targets them
#
# ENVIRONMENT MATCHING:
# - Game environment now matches training environment constraints
# - AI player (right side) uses constrained target selection
# - Human player (left side) has no constraints
# - Both environments use same shooting cone validation and bounce physics

# Import shared geometry module for consistency
from bubble_geometry import (
    SCREEN_WIDTH, SCREEN_HEIGHT, BUBBLE_RADIUS, GRID_ROWS, GRID_COLS,
    BUBBLE_COLORS, LOSE_THRESHOLD, CENTER_LINE_PUSH_AMOUNT, MAX_CENTER_LINE_OFFSET,
    grid_to_screen, screen_to_grid, is_valid_placement, get_valid_targets,
    check_lose_condition, encode_color_planes, calculate_angle_to_target,
    get_adjacent_positions, simulate_shot_trajectory, encode_compact_state
)

# Game-specific constants
BUBBLE_SPEED = 10
SHOOT_SPEED = 10
MATCH_THRESHOLD = 3
PLAYER_ONE_COLOR = (255, 0, 0)  # Red
PLAYER_TWO_COLOR = (0, 0, 255)  # Blue
MIDDLE_LINE_WIDTH = 4  # Thicker middle line
LOSING_THRESHOLD_ROW = 10  # Row at which a player loses
LOSING_THRESHOLD_COL = 5  # Column at which a player loses
LOSING_LINE_COLOR = (255, 0, 0, 128)  # Semi-transparent red

# Add these constants for Target-based DQN
DQN_MODEL_PATH = 'target_bubbles_dqn_model.pth'
# FIXED: Only current bubble color to prevent AI from confusing current vs next
# compact_state = 700 (grid colors) + 1 (current color) = 701
# total state = 701 + 700 (neighbor counts) + 3 (features) = 1404
DQN_STATE_SIZE = 701 + 700 + 3
# FIXED: AI chooses targets (positions) based on CURRENT bubble color analysis only
DQN_ACTION_SIZE = 700  # 700 actions (one per grid position)
DQN_COLOR_ACTION_SIZE = 6  # Keep for backward compatibility during transition

# Debug constants
DEBUG_AI = True  # Enable AI debug overlays and console logs by default (toggle with 'D')

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
        self.grid_pos = None  # Will store (row, col, player) position in the grid
        self.bounce_count = 0  # Track number of wall bounces
        self.shot_by_player = None  # Track which player shot this bubble

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
            self.bounce_count += 1
            return True
        return False

    def snap_to_grid(self, grid_pos: Tuple[int, int], grid_to_screen: Dict[Tuple[int, int], Tuple[int, int]]):
        self.grid_pos = grid_pos
        self.x, self.y = grid_to_screen[grid_pos]
        self.is_moving = False
        self.snapped = True

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Two Player Bubble Shooter")
        self.clock = pygame.time.Clock()
        self.bubbles: List[Bubble] = []
        self.grid: Dict[Tuple[int, int], Bubble] = {}  # Maps (row, col) to Bubble
        self.grid_to_screen: Dict[Tuple[int, int], Tuple[int, int]] = {}  # Maps (row, col) to (x, y)
        self.current_player = 1  # 1 for player one, 2 for player two
        self.game_over = False
        self.score_player_one = 0
        self.score_player_two = 0
        self.center_line_offset = 0  # Current offset (animated)
        self.target_center_line_offset = 0  # Where the offset should animate to
        self.falling_bubbles: List[Bubble] = []  # Track bubbles that are falling
        self.initialize_grid()
        self.initialize_bubbles()
        self.initialize_shooters()
        # Target-based DQN agent for right-side AI
        self.dqn_agent = TargetDQNAgent(DQN_STATE_SIZE, DQN_ACTION_SIZE, device='cpu')
        try:
            self.dqn_agent.load(DQN_MODEL_PATH)
            print(f"AI model loaded successfully from {DQN_MODEL_PATH}")
        except FileNotFoundError:
            print(f"Warning: No trained model found at {DQN_MODEL_PATH}")
            print("AI will play with random actions until a model is trained.")
        except Exception as e:
            print(f"Error loading AI model: {e}")
            print("AI will play with random actions.")
        self.ai_action_cooldown = 0  # Frames until next AI action

    def initialize_grid(self):
        # Calculate the starting position for both players' grids using shared geometry
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                # Player one's grid (left side, starting from middle)
                x_one, y_one = grid_to_screen(row, col, 1, self.center_line_offset)
                self.grid_to_screen[(row, col, 1)] = (x_one, y_one)
                
                # Player two's grid (right side, starting from middle)
                x_two, y_two = grid_to_screen(row, col, 2, self.center_line_offset)
                self.grid_to_screen[(row, col, 2)] = (x_two, y_two)

    def initialize_bubbles(self):
        # Initialize bubbles for both players
        initial_cols = 7
        for row in range(GRID_ROWS):
            for col in range(initial_cols):
                # Player one's bubbles
                x_one, y_one = self.grid_to_screen[(row, col, 1)]
                color = random.choice(BUBBLE_COLORS)
                bubble_one = Bubble(x_one, y_one, color)
                bubble_one.grid_pos = (row, col, 1)
                self.bubbles.append(bubble_one)
                self.grid[(row, col, 1)] = bubble_one
                
                # Player two's bubbles
                x_two, y_two = self.grid_to_screen[(row, col, 2)]
                color = random.choice(BUBBLE_COLORS)
                bubble_two = Bubble(x_two, y_two, color)
                bubble_two.grid_pos = (row, col, 2)
                self.bubbles.append(bubble_two)
                self.grid[(row, col, 2)] = bubble_two

    def initialize_shooters(self):
        # Initialize shooters for both players
        self.shooter_one = {
            'x': BUBBLE_RADIUS * 2,
            'y': SCREEN_HEIGHT // 2,
            'angle': 0,
            'current_bubble': None,
            'next_bubble': None
        }
        self.shooter_two = {
            'x': SCREEN_WIDTH - BUBBLE_RADIUS * 2,
            'y': SCREEN_HEIGHT // 2,
            'angle': 180,  # Point left
            'current_bubble': None,
            'next_bubble': None
        }
        self.update_shooters()

    def update_shooters(self):
        # Update current bubbles for both shooters
        color = random.choice(BUBBLE_COLORS)
        self.shooter_one['current_bubble'] = Bubble(
            self.shooter_one['x'],
            self.shooter_one['y'],
            color
        )
        # Set next bubble for player 1
        next_color = random.choice(BUBBLE_COLORS)
        self.shooter_one['next_bubble'] = Bubble(
            self.shooter_one['x'] + BUBBLE_RADIUS * 2.5,  # Position to the right
            self.shooter_one['y'],
            next_color
        )
        
        color = random.choice(BUBBLE_COLORS)
        self.shooter_two['current_bubble'] = Bubble(
            self.shooter_two['x'],
            self.shooter_two['y'],
            color
        )
        # Set next bubble for player 2
        next_color = random.choice(BUBBLE_COLORS)
        self.shooter_two['next_bubble'] = Bubble(
            self.shooter_two['x'] - BUBBLE_RADIUS * 2.5,  # Position to the left
            self.shooter_two['y'],
            next_color
        )

    def shoot(self, angle: float, force_player=None):
        # Only shoot if there's no moving bubble for the relevant player
        if force_player is not None:
            shooter = self.shooter_one if force_player == 1 else self.shooter_two
            
            # Check if there's already a moving bubble for this player
            if any(b.is_moving and b.shot_by_player == force_player for b in self.bubbles):
                return  # Don't shoot if there's already a moving bubble
            
            if shooter['current_bubble']:
                shooter['current_bubble'].is_moving = True
                shooter['current_bubble'].velocity_x = math.cos(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].velocity_y = math.sin(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].shot_by_player = force_player  # Track which player shot this bubble
                self.bubbles.append(shooter['current_bubble'])
                if force_player == 1:
                    # For human player, move next bubble to current and generate new next
                    if shooter['next_bubble']:
                        shooter['current_bubble'] = shooter['next_bubble']
                        shooter['current_bubble'].x = shooter['x']
                        shooter['current_bubble'].y = shooter['y']
                        # Generate new next bubble
                        next_color = random.choice(BUBBLE_COLORS)
                        shooter['next_bubble'] = Bubble(
                            shooter['x'] + BUBBLE_RADIUS * 2.5,
                            shooter['y'],
                            next_color
                        )
                    else:
                        shooter['current_bubble'] = None
                else:
                    # For AI player, move next bubble to current and generate new next
                    if shooter['next_bubble']:
                        shooter['current_bubble'] = shooter['next_bubble']
                        shooter['current_bubble'].x = shooter['x']
                        shooter['current_bubble'].y = shooter['y']
                        # Generate new next bubble
                        next_color = random.choice(BUBBLE_COLORS)
                        shooter['next_bubble'] = Bubble(
                            shooter['x'] - BUBBLE_RADIUS * 2.5,
                            shooter['y'],
                            next_color
                        )
                    else:
                        color = random.choice(BUBBLE_COLORS)
                        shooter['current_bubble'] = Bubble(shooter['x'], shooter['y'], color)
            return
        # Legacy turn-based fallback
        if not any(b.is_moving for b in self.bubbles):
            shooter = self.shooter_one if self.current_player == 1 else self.shooter_two
            if shooter['current_bubble']:
                shooter['current_bubble'].is_moving = True
                shooter['current_bubble'].velocity_x = math.cos(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].velocity_y = math.sin(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].shot_by_player = self.current_player  # Track which player shot this bubble
                self.bubbles.append(shooter['current_bubble'])
                # Move next bubble to current and generate new next
                if shooter['next_bubble']:
                    shooter['current_bubble'] = shooter['next_bubble']
                    shooter['current_bubble'].x = shooter['x']
                    shooter['current_bubble'].y = shooter['y']
                    # Generate new next bubble
                    next_color = random.choice(BUBBLE_COLORS)
                    if self.current_player == 1:
                        shooter['next_bubble'] = Bubble(
                            shooter['x'] + BUBBLE_RADIUS * 2.5,
                            shooter['y'],
                            next_color
                        )
                    else:
                        shooter['next_bubble'] = Bubble(
                            shooter['x'] - BUBBLE_RADIUS * 2.5,
                            shooter['y'],
                            next_color
                        )
                else:
                    color = random.choice(BUBBLE_COLORS)
                    shooter['current_bubble'] = Bubble(shooter['x'], shooter['y'], color)
                self.current_player = 3 - self.current_player  # Switches between 1 and 2

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEMOTION and not self.game_over:
                # Always update left shooter angle based on mouse position
                mouse_x, mouse_y = event.pos
                shooter = self.shooter_one
                dx = mouse_x - shooter['x']
                dy = mouse_y - shooter['y']
                angle = math.degrees(math.atan2(dy, dx))
                # Clamp angle to [-90, 90] (up-right to down-right)
                angle = max(-90, min(90, angle))
                shooter['angle'] = angle
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                # Shoot if shooter_one has a bubble (the shoot method will check for moving bubbles)
                if self.shooter_one['current_bubble']:
                    shooter = self.shooter_one
                    self.shoot(shooter['angle'], force_player=1)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:  # Press 'D' to toggle debug mode
                    global DEBUG_AI
                    DEBUG_AI = not DEBUG_AI
                    print(f"Debug mode {'enabled' if DEBUG_AI else 'disabled'}")
        return True

    def draw_ai_shooting_angles(self):
        """Draw all possible shooting angles for the AI opponent (right side)"""
        shooter = self.shooter_two
        # Do not draw generic angle fan to avoid implying walls are targets
        # If needed for debugging, set show_angle_fan=True
        show_angle_fan = False
        if show_angle_fan:
            line_length = 80
            for action_idx in range(min(50, DQN_ACTION_SIZE)):
                angle = 90 + (action_idx * 180 / min(50, DQN_ACTION_SIZE - 1))
                end_x = shooter['x'] + math.cos(math.radians(angle)) * line_length
                end_y = shooter['y'] + math.sin(math.radians(angle)) * line_length
                pygame.draw.line(self.screen, (120,120,120), (shooter['x'], shooter['y']), (end_x, end_y), 1)
        
        # NEW: Draw all valid targets the AI can choose from
        if DEBUG_AI:
            valid_targets = self.get_valid_targets_constrained(2)  # Get AI's valid targets
            font = pygame.font.Font(None, 20)
            
            for i, (row, col) in enumerate(valid_targets):
                if (row, col, 2) in self.grid_to_screen:
                    target_x, target_y = self.grid_to_screen[(row, col, 2)]
                    
                    # Draw valid target positions as small green circles
                    pygame.draw.circle(self.screen, (0, 255, 0), (int(target_x), int(target_y)), 8, 2)
                    
                    # Add target number for identification
                    target_num = font.render(f"{i+1}", True, (0, 255, 0))
                    self.screen.blit(target_num, (int(target_x) + 12, int(target_y) - 8))
        
        # DEBUG: Draw the AI's current target and decision
        if DEBUG_AI and hasattr(self, 'last_ai_target'):
            target_x, target_y = self.last_ai_target
            # Draw target position as a bright red circle
            pygame.draw.circle(self.screen, (255, 0, 0), (int(target_x), int(target_y)), 15, 3)
            # Draw line from shooter to target
            pygame.draw.line(self.screen, (255, 255, 0), 
                           (shooter['x'], shooter['y']), 
                           (target_x, target_y), 4)
            # Draw text showing target coordinates
            font = pygame.font.Font(None, 24)
            target_text = font.render(f"Target: ({int(target_x)}, {int(target_y)})", True, (255, 255, 255))
            self.screen.blit(target_text, (target_x + 20, target_y - 10))
            
            # Draw the calculated shooting trajectory
            if hasattr(self, 'last_ai_angle'):
                trajectory_length = 200
                end_x = shooter['x'] + math.cos(math.radians(self.last_ai_angle)) * trajectory_length
                end_y = shooter['y'] + math.sin(math.radians(self.last_ai_angle)) * trajectory_length
                # Draw trajectory line in bright green
                pygame.draw.line(self.screen, (0, 255, 0), 
                               (shooter['x'], shooter['y']), 
                               (end_x, end_y), 3)
                # Draw trajectory end point
                pygame.draw.circle(self.screen, (0, 255, 0), (int(end_x), int(end_y)), 8, 2)
                
                # NEW: Draw predicted landing position vs target (only if available)
                if hasattr(self, 'last_predicted_landing') and self.last_predicted_landing is not None:
                    pred_x, pred_y = self.last_predicted_landing
                    # Draw predicted landing as a blue circle
                    pygame.draw.circle(self.screen, (0, 0, 255), (int(pred_x), int(pred_y)), 12, 3)
                    # Draw line from target to predicted landing to show the error
                    pygame.draw.line(self.screen, (255, 0, 255), 
                                   (int(target_x), int(target_y)), 
                                   (int(pred_x), int(pred_y)), 2)
                    # Add text showing the error
                    error_text = font.render(f"Error: {math.sqrt((target_x - pred_x)**2 + (target_y - pred_y)**2):.0f}px", True, (255, 0, 255))
                    self.screen.blit(error_text, (int(pred_x) + 20, int(pred_y) - 10))

    def draw(self):
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Calculate current center line position
        current_center_x = SCREEN_WIDTH // 2 + self.center_line_offset
        
        # Draw the vertical line first
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (current_center_x, 0), 
                        (current_center_x, SCREEN_HEIGHT), 
                        MIDDLE_LINE_WIDTH)
        
        # Draw losing threshold lines
        # Left player's threshold
        pygame.draw.line(self.screen, LOSING_LINE_COLOR,
                        (LOSE_THRESHOLD, 0),
                        (LOSE_THRESHOLD, SCREEN_HEIGHT),
                        2)
        # Right player's threshold
        pygame.draw.line(self.screen, LOSING_LINE_COLOR,
                        (SCREEN_WIDTH - LOSE_THRESHOLD, 0),
                        (SCREEN_WIDTH - LOSE_THRESHOLD, SCREEN_HEIGHT),
                        2)
        
        # Draw all bubbles
        for bubble in self.bubbles:
            bubble.draw(self.screen)
        
        # DEBUG: Draw per-bubble adjacent-same-color counts (self-color adjacency)
        if DEBUG_AI:
            try:
                # Build AI grid as color indices
                ai_grid = {}
                for bubble in self.bubbles:
                    if bubble.grid_pos is not None and bubble.grid_pos[2] == 2:
                        row, col, _ = bubble.grid_pos
                        try:
                            color_idx = BUBBLE_COLORS.index(bubble.color)
                            ai_grid[(row, col)] = color_idx
                        except ValueError:
                            continue
                # Render per-bubble self-color adjacency counts
                font_cnt = pygame.font.Font(None, 18)
                for bubble in self.bubbles:
                    if bubble.grid_pos is not None and bubble.grid_pos[2] == 2:
                        row, col, _ = bubble.grid_pos
                        # Get this bubble's color index
                        try:
                            bubble_color_idx = BUBBLE_COLORS.index(bubble.color)
                        except ValueError:
                            bubble_color_idx = -1
                        # Count adjacent same-color neighbors in honeycomb pattern
                        if row % 2 == 0:
                            adj = [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col-1), (row+1, col-1)]
                        else:
                            adj = [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col+1), (row+1, col+1)]
                        val = 0
                        for ar, ac in adj:
                            if (ar, ac) in ai_grid and ai_grid[(ar, ac)] == bubble_color_idx:
                                val += 1
                        txt = font_cnt.render(str(val), True, (0, 255, 255))
                        self.screen.blit(txt, (int(bubble.x) - 5, int(bubble.y) - 28))
            except Exception:
                pass
        
        # DEBUG: Draw per-cell Q-values on the AI grid (player 2)
        if DEBUG_AI and hasattr(self, 'last_q_values') and self.last_q_values is not None:
            font_small = pygame.font.Font(None, 22)  # Bigger font for readability
            best_idx = getattr(self, 'last_best_action_idx', None)
            # Show Q-values ONLY for reachable bubble targets (occupied cells) to match masking
            valid_set = set(self.get_reachable_bubble_targets(2))
            for row, col in valid_set:
                key = (row, col, 2)
                if key not in self.grid_to_screen:
                    continue
                idx = row * 35 + col  # GRID_COLS = 35
                x, y = self.grid_to_screen[key]
                q_val = float(self.last_q_values[idx])
                if not math.isfinite(q_val):
                    text_str = "-inf"
                    color = (210, 210, 210)  # Brighter
                else:
                    text_str = f"{q_val:.1f}"
                    color = (255, 255, 220)  # Brighter
                if best_idx is not None and idx == best_idx:
                    color = (255, 240, 120)  # Highlight best, brighter
                    pygame.draw.circle(self.screen, (255, 240, 120), (int(x), int(y)), 12, 1)
                txt = font_small.render(text_str, True, color)
                tx = int(x) - 12
                ty = int(y) - 10
                # Draw black background behind the chosen target's Q-value for readability
                try:
                    if hasattr(self, 'last_chosen_action_idx') and self.last_chosen_action_idx is not None and idx == int(self.last_chosen_action_idx):
                        bg_rect = pygame.Rect(tx - 2, ty - 2, txt.get_width() + 4, txt.get_height() + 4)
                        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
                except Exception:
                    pass
                self.screen.blit(txt, (tx, ty))
        
        # Draw AI shooting angles (all possible angles)
        self.draw_ai_shooting_angles()
        
        # NEW: Draw shooting cone for debugging
        # Removed cone visualization per new design
        # if DEBUG_AI:
        #     self.draw_shooting_cone()
        
        # Draw shooters
        for player_num, shooter in [(1, self.shooter_one), (2, self.shooter_two)]:
            # Draw shooter base
            pygame.draw.circle(self.screen, (200, 200, 200), 
                             (shooter['x'], shooter['y']), BUBBLE_RADIUS)
            
            # Draw current bubble
            if shooter['current_bubble']:
                shooter['current_bubble'].draw(self.screen)
            
            # Draw next bubble preview (smaller and semi-transparent)
            if shooter['next_bubble']:
                # Create a smaller, semi-transparent version of the next bubble
                preview_radius = int(BUBBLE_RADIUS * 0.7)
                preview_surface = pygame.Surface((preview_radius * 2, preview_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(preview_surface, (*shooter['next_bubble'].color, 180), 
                                 (preview_radius, preview_radius), preview_radius)
                pygame.draw.circle(preview_surface, (255, 255, 255, 180), 
                                 (preview_radius, preview_radius), preview_radius, 2)
                self.screen.blit(preview_surface, 
                               (shooter['next_bubble'].x - preview_radius, 
                                shooter['next_bubble'].y - preview_radius))
            
            # Draw angle indicator (current angle)
            end_x = shooter['x'] + math.cos(math.radians(shooter['angle'])) * 50
            end_y = shooter['y'] + math.sin(math.radians(shooter['angle'])) * 50
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (shooter['x'], shooter['y']), 
                           (end_x, end_y), 3)  # Thicker line for current angle
        
        # Draw scores
        font = pygame.font.Font(None, 36)
        # Left player (Player 1)
        left_score = font.render(f"Player 1: {self.score_player_one}", True, PLAYER_ONE_COLOR)
        self.screen.blit(left_score, (10, 10))
        # Right player (Player 2)
        right_score = font.render(f"Player 2: {self.score_player_two}", True, PLAYER_TWO_COLOR)
        self.screen.blit(right_score, (SCREEN_WIDTH - right_score.get_width() - 10, 10))
        
        # Draw current player indicator
        current_player_text = font.render(f"Current Player: {self.current_player}", True, (255, 255, 255))
        self.screen.blit(current_player_text, (SCREEN_WIDTH // 2 - 100, 10))
        
        # Draw center line offset indicator
        offset_text = font.render(f"Center Offset: {self.center_line_offset}", True, (255, 255, 0))
        self.screen.blit(offset_text, (SCREEN_WIDTH // 2 - 100, 50))
        
        # DEBUG: Draw AI debug info panel
        if DEBUG_AI:
            self.draw_ai_debug_panel()
        
        if self.game_over:
            font = pygame.font.Font(None, 74)
            text = font.render("Game Over!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()

    def check_collision(self, moving_bubble: Bubble) -> Optional[Tuple[int, int, int]]:
        # Check collision with walls first
        if moving_bubble.y - moving_bubble.radius <= 0 or moving_bubble.y + moving_bubble.radius >= SCREEN_HEIGHT:
            return None
            
        # Check collision with other bubbles
        for bubble in self.bubbles:
            if not bubble.is_moving and bubble.grid_pos is not None:
                # Calculate distance between bubble centers
                dx = moving_bubble.x - bubble.x
                dy = moving_bubble.y - bubble.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                # If distance is less than sum of radii, we have a collision
                if distance < BUBBLE_RADIUS * 2:
                    # Calculate the position where the bubble should snap
                    snap_x = bubble.x + (dx / distance) * BUBBLE_RADIUS * 2
                    snap_y = bubble.y + (dy / distance) * BUBBLE_RADIUS * 2
                    return self.find_nearest_grid_position(snap_x, snap_y)
        
        # Check if bubble has reached the edge
        current_center_x = SCREEN_WIDTH // 2 + self.center_line_offset
        player_num = 1 if moving_bubble.x < current_center_x else 2
        if (player_num == 1 and moving_bubble.x + moving_bubble.radius >= current_center_x) or \
           (player_num == 2 and moving_bubble.x - moving_bubble.radius <= current_center_x):
            return self.find_nearest_grid_position(
                current_center_x - BUBBLE_RADIUS if player_num == 1 else current_center_x + BUBBLE_RADIUS,
                moving_bubble.y
            )
            
        return None

    def find_nearest_grid_position(self, x: float, y: float) -> Optional[Tuple[int, int, int]]:
        min_dist = float('inf')
        nearest_pos = None
        current_center_x = SCREEN_WIDTH // 2 + self.center_line_offset
        player_num = 1 if x < current_center_x else 2
        
        # First, find the closest existing bubble
        closest_bubble = None
        min_bubble_dist = float('inf')
        for bubble in self.bubbles:
            if not bubble.is_moving and bubble.grid_pos is not None and bubble.grid_pos[2] == player_num:
                dist = math.sqrt((x - bubble.x) ** 2 + (y - bubble.y) ** 2)
                if dist < min_bubble_dist:
                    min_bubble_dist = dist
                    closest_bubble = bubble
        
        if closest_bubble and min_bubble_dist < BUBBLE_RADIUS * 4:
            # Find the nearest empty grid position around the closest bubble
            row, col, _ = closest_bubble.grid_pos
            # Check positions in a honeycomb pattern
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                new_pos = (row + dr, col + dc, player_num)
                if new_pos not in self.grid and new_pos in self.grid_to_screen:
                    grid_x, grid_y = self.grid_to_screen[new_pos]
                    dist = math.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pos = new_pos
        else:
            # If no nearby bubble, find the nearest grid position
            for (row, col, p_num), (grid_x, grid_y) in self.grid_to_screen.items():
                if p_num == player_num and (row, col, p_num) not in self.grid:
                    dist = math.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pos = (row, col, p_num)
        
        return nearest_pos if min_dist < BUBBLE_RADIUS * 4 else None

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
            row, col, player_num = current.grid_pos
            # For even rows, check these positions
            if row % 2 == 0:
                adjacent_positions = [
                    (0, 1),   # right
                    (1, 0),   # down
                    (0, -1),  # left
                    (-1, 0),  # up
                    (1, -1),  # down-left
                    (-1, -1)  # up-left
                ]
            # For odd rows, check these positions
            else:
                adjacent_positions = [
                    (0, 1),   # right
                    (1, 1),   # down-right
                    (0, -1),  # left
                    (-1, 1),  # up-right
                    (1, 0),   # down
                    (-1, 0)   # up
                ]

            for dr, dc in adjacent_positions:
                adj_pos = (row + dr, col + dc, player_num)
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
        
        # Update score and push center line for the correct player
        if bubbles_to_remove:
            popping_player = bubbles_to_remove[0].grid_pos[2]  # Use the player number of the popped bubbles
            if popping_player == 1:
                self.score_player_one += len(bubbles_to_remove)
            else:
                self.score_player_two += len(bubbles_to_remove)
            self.push_center_line_away_from_player(popping_player, len(bubbles_to_remove))

    def check_isolated_bubbles(self) -> List[Bubble]:
        # First, find all bubbles that are connected to the top row for each player
        connected_to_top = set()
        to_check = []
        
        # Start with all bubbles in the top row for both players
        for player_num in [1, 2]:
            for col in range(GRID_COLS):
                if (0, col, player_num) in self.grid:
                    to_check.append((0, col, player_num))
                    connected_to_top.add((0, col, player_num))
        
        # Breadth-first search to find all connected bubbles
        while to_check:
            row, col, player_num = to_check.pop(0)
            
            # Get adjacent positions based on row parity
            if row % 2 == 0:
                adjacent_positions = [
                    (0, 1),   # right
                    (1, 0),   # down
                    (0, -1),  # left
                    (-1, 0),  # up
                    (1, -1),  # down-left
                    (-1, -1)  # up-left
                ]
            else:
                adjacent_positions = [
                    (0, 1),   # right
                    (1, 1),   # down-right
                    (0, -1),  # left
                    (-1, 1),  # up-right
                    (1, 0),   # down
                    (-1, 0)   # up
                ]
            
            # Check each adjacent position
            for dr, dc in adjacent_positions:
                new_pos = (row + dr, col + dc, player_num)
                if (new_pos in self.grid and 
                    new_pos not in connected_to_top and 
                    0 <= new_pos[0] < GRID_ROWS and 
                    0 <= new_pos[1] < GRID_COLS):
                    connected_to_top.add(new_pos)
                    to_check.append(new_pos)
        
        # Any bubble not in connected_to_top is isolated
        isolated = []
        for bubble in self.bubbles:
            if (not bubble.is_moving and 
                bubble.grid_pos is not None and 
                bubble.grid_pos not in connected_to_top):
                isolated.append(bubble)
        
        return isolated

    def check_lose_condition(self) -> bool:
        # Check if any bubble has reached the edge for either player
        current_center_x = SCREEN_WIDTH // 2 + self.center_line_offset
        for bubble in self.bubbles:
            if not bubble.is_moving and bubble.grid_pos is not None:
                row, col, player_num = bubble.grid_pos
                if player_num == 1 and bubble.x <= LOSE_THRESHOLD:
                    return True
                elif player_num == 2 and bubble.x >= SCREEN_WIDTH - LOSE_THRESHOLD:
                    return True
        return False

    def update(self):
        if self.game_over:
            return

        # AI (right side) acts in real time
        if not any(b.is_moving and b.shot_by_player == 2 for b in self.bubbles):
            # Only act if AI's bubble is ready
            if self.shooter_two['current_bubble'] and self.ai_action_cooldown == 0:
                dqn_state = self.encode_dqn_state()
                
                # NEW: Bubble-target action system using hybrid masking
                reachable_targets = self.get_hybrid_masked_targets(2)
                
                # If no targets reachable, let AI shoot randomly (will lose anyway)
                if not reachable_targets:
                    # No valid targets - AI will shoot randomly and lose
                    # This prevents the game from getting stuck
                    reachable_targets = []  # Empty list triggers random shooting
                
                # Select target action (inference mode - no learning or exploration)
                # AI chooses which target to shoot at based on color analysis
                # Unreachable targets are automatically masked with -infinity
                ai_target_action = self.dqn_agent.select_action(dqn_state, reachable_targets, training_mode=False)
                # Store chosen action index for debug Q-value highlight
                try:
                    self.last_chosen_action_idx = int(ai_target_action)
                except Exception:
                    self.last_chosen_action_idx = None
                
                # DEBUG: Get Q-values for analysis
                if DEBUG_AI:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(dqn_state).unsqueeze(0).to('cpu')
                        q_values = self.dqn_agent.policy_net(state_tensor)
                        
                        # Apply masking: front-most occupied bubble per row only (simple mask)
                        front_targets = self.get_front_bubble_targets(2)
                        if front_targets:
                            masked_q_values = self.dqn_agent.apply_action_mask(q_values, front_targets)
                            # Show top 5 valid actions
                            valid_action_indices = []
                            for row, col in reachable_targets:
                                action_idx = row * 35 + col  # GRID_COLS = 35
                                valid_action_indices.append(action_idx)
                            
                            if valid_action_indices:
                                valid_q_values = masked_q_values[0][valid_action_indices]
                                top_5_indices = torch.topk(valid_q_values, min(5, len(valid_q_values))).indices
                                top_5_actions = [(valid_action_indices[idx.item()], valid_q_values[idx.item()]) for idx in top_5_indices]
                                print(f"Top 5 Q-values (valid targets): {[(idx, val.item()) for idx, val in top_5_actions]}")
                        else:
                            # No front targets - visualize as all -inf
                            masked_q_values = torch.full_like(q_values, float('-inf'))
                            print("No front targets - Qs shown as -inf")

                        # Store masked Q-values for on-screen visualization
                        try:
                            self.last_q_values = masked_q_values[0].detach().cpu().numpy()
                        except Exception:
                            self.last_q_values = None
                        # Determine best finite action index (if any)
                        try:
                            best_idx = int(torch.argmax(masked_q_values, dim=1).item())
                            best_val = float(masked_q_values[0][best_idx].item())
                            if math.isfinite(best_val):
                                self.last_best_action_idx = best_idx
                            else:
                                self.last_best_action_idx = None
                        except Exception:
                            self.last_best_action_idx = None
                
                # Convert target action index to row, col coordinates (occupied cell)
                target_row = ai_target_action // 35  # GRID_COLS = 35
                target_col = ai_target_action % 35
                
                # Validate target position
                if not (0 <= target_row < 20 and 0 <= target_col < 35):  # GRID_ROWS = 20
                    # Invalid target - use fallback
                    if reachable_targets:
                        target_row, target_col = reachable_targets[0]
                    else:
                        # No valid targets - shoot randomly (will lose anyway)
                        target_row, target_col = random.randint(0, 19), random.randint(0, 34)
                
                # Calculate angle to target
                shooter_x = self.shooter_two['x']
                shooter_y = self.shooter_two['y']
                
                # Get target bubble screen position (must exist)
                if (target_row, target_col, 2) in self.grid_to_screen and (target_row, target_col, 2) in self.grid:
                    target_x, target_y = self.grid_to_screen[(target_row, target_col, 2)]
                else:
                    # If chosen target is no longer occupied, fallback to first reachable
                    if reachable_targets:
                        fr, fc = reachable_targets[0]
                        target_x, target_y = self.grid_to_screen[(fr, fc, 2)]
                        target_row, target_col = fr, fc
                    else:
                        # Emergency fallback: aim at center line
                        target_x = SCREEN_WIDTH // 2 + self.center_line_offset
                        target_y = self.shooter_two['y']
                
                # DEBUG: Print AI decision details
                if DEBUG_AI:
                    print(f"\n=== AI DECISION DEBUG ===")
                    print(f"Reachable Targets: {len(reachable_targets)} positions")
                    print(f"AI Target Action: {ai_target_action}")
                    print(f"Target Position: Row={target_row}, Col={target_col}")
                    print(f"Current Bubble Color: {self.shooter_two['current_bubble'].color}")
                    print(f"Shooter Position: ({shooter_x:.1f}, {shooter_y:.1f})")
                    print(f"Target Position: ({target_x:.1f}, {target_y:.1f})")
                    print(f"Raw Calculated Angle: {calculate_angle_to_target(shooter_x, shooter_y, target_x, target_y):.1f}°")
                
                # Direct-LOS-only policy: compute direct angle to the bubble center
                ai_angle = calculate_angle_to_target(shooter_x, shooter_y, target_x, target_y)

                # Training parity: apply slight nudge based on verticality and clamp to [90°, 270°]
                try:
                    rad = math.radians(ai_angle)
                    bump_deg = 3.0 * abs(math.sin(rad))
                    if 90.0 <= ai_angle <= 270.0:
                        if target_y < shooter_y:
                            # Upward shot -> decrease angle toward 90° (higher)
                            ai_angle -= bump_deg
                        else:
                            # Downward shot -> increase angle toward 270° (lower)
                            ai_angle += bump_deg
                        # Clamp for right-shooter
                        if ai_angle < 90.0:
                            ai_angle = 90.0
                        elif ai_angle > 270.0:
                            ai_angle = 270.0
                except Exception:
                    pass

                # Training parity: refine angle with small offsets to prefer exact landing on the intended target
                try:
                    ai_angle = self._refine_angle_to_hit_target(ai_angle, 2, shooter_x, shooter_y, target_row, target_col)
                except Exception:
                    pass
                
                # Store target and angle for visual debugging
                self.last_ai_target = (target_x, target_y)
                self.last_ai_angle = ai_angle
                self.last_predicted_landing = None
                
                if DEBUG_AI:
                    print(f"Grid Hit Angle: {ai_angle:.1f}°")
                    print(f"Angle Difference: {abs(ai_angle - calculate_angle_to_target(shooter_x, shooter_y, target_x, target_y)):.1f}°")
                    print(f"Target Position: ({target_x:.1f}, {target_y:.1f})")
                    # Predicted landing disabled in fast LOS mode
                    print(f"Reachable Targets Count: {len(reachable_targets)}")
                    print(f"Current Bubble Color: {self.shooter_two['current_bubble'].color}")
                    print(f"FIXED: AI only sees current bubble color (no next bubble planning)")
                    print(f"========================\n")
                
                self.shooter_two['angle'] = ai_angle
                self.shoot(ai_angle, force_player=2)  # Ensure only right shooter acts
                self.ai_action_cooldown = 2 * 60  # 2 seconds at 60 FPS
        if self.ai_action_cooldown > 0:
            self.ai_action_cooldown -= 1

        # Smoothly animate the center line offset toward the target
        if self.center_line_offset != self.target_center_line_offset:
            # Move a fraction of the distance each frame for smooth animation
            diff = self.target_center_line_offset - self.center_line_offset
            step = max(1, int(abs(diff) * 0.2))  # 20% of the distance, at least 1 pixel
            if diff > 0:
                self.center_line_offset += step
                if self.center_line_offset > self.target_center_line_offset:
                    self.center_line_offset = self.target_center_line_offset
            else:
                self.center_line_offset -= step
                if self.center_line_offset < self.target_center_line_offset:
                    self.center_line_offset = self.target_center_line_offset
            # Recalculate all grid positions and update bubble positions
            self.initialize_grid()
            for bubble in self.bubbles:
                if bubble.grid_pos and not bubble.is_moving:
                    bubble.x, bubble.y = self.grid_to_screen[bubble.grid_pos]

        # Animate falling bubbles non-blocking
        if self.falling_bubbles:
            still_falling = []
            for bubble in self.falling_bubbles:
                if bubble.y < SCREEN_HEIGHT + BUBBLE_RADIUS:
                    bubble.y += bubble.velocity_y
                    bubble.velocity_y += 0.5  # Accelerate falling
                    still_falling.append(bubble)
            self.falling_bubbles = still_falling
            # Remove fallen bubbles when done
            if not self.falling_bubbles:
                self.remove_bubbles([b for b in self.bubbles if b.is_moving and b.y >= SCREEN_HEIGHT + BUBBLE_RADIUS])
            return  # Skip the rest of the update until animation is complete

        # Check for isolated bubbles first
        isolated = self.check_isolated_bubbles()
        if isolated:
            self.animate_falling_bubbles(isolated)
            return  # Skip the rest of the update until animation is complete

        for bubble in self.bubbles:
            if bubble.is_moving and not bubble.snapped:
                bubble.update()
                if bubble.check_wall_collision():
                    # If bubble has bounced more than twice, remove it and update shooter color
                    if bubble.bounce_count > 2:
                        self.bubbles.remove(bubble)
                        # Update only the shooter that matches the bubble's shot_by_player
                        if bubble.shot_by_player == 1:
                            shooter = self.shooter_one
                        elif bubble.shot_by_player == 2:
                            shooter = self.shooter_two
                        else:
                            shooter = None
                        if shooter and shooter['current_bubble']:
                            shooter['current_bubble'].color = bubble.color
                        continue

                # Check for collisions with other bubbles
                collision = self.check_collision(bubble)
                if collision:
                    bubble.snap_to_grid(collision, self.grid_to_screen)
                    self.grid[collision] = bubble
                    
                    # Check for matches
                    matches = self.check_matches(bubble)
                    if matches:
                        self.remove_bubbles(matches)
                        # Check for isolated bubbles after removing matches
                        isolated = self.check_isolated_bubbles()
                        if isolated:
                            self.animate_falling_bubbles(isolated)
                            return  # Skip the rest of the update until animation is complete

        # Note: Bubble replenishment is now handled in the shoot method to avoid conflicts

        # Check for game over
        if self.check_lose_condition():
            self.game_over = True

    def animate_falling_bubbles(self, bubbles_to_fall: List[Bubble]):
        # Mark bubbles as falling and add to self.falling_bubbles
        for bubble in bubbles_to_fall:
            bubble.is_moving = True
            bubble.velocity_y = 5  # Initial falling speed
            bubble.velocity_x = 0
        self.falling_bubbles.extend(bubbles_to_fall)

    def update_center_line_offset(self, offset_change: int):
        """Update the target center line offset for smooth animation"""
        old_target = self.target_center_line_offset
        self.target_center_line_offset = max(-MAX_CENTER_LINE_OFFSET, min(MAX_CENTER_LINE_OFFSET, self.target_center_line_offset + offset_change))

    def push_center_line_away_from_player(self, player_num: int, num_bubbles_popped: int):
        """Push the center line away from the player who popped bubbles"""
        push_amount = num_bubbles_popped * CENTER_LINE_PUSH_AMOUNT
        
        # Calculate the current center line position
        current_center_x = SCREEN_WIDTH // 2 + self.center_line_offset
        
        if player_num == 1:
            # Player 1 popped bubbles, push toward player 2 (positive offset)
            # This adds to the current offset, making it relative to current position
            self.update_center_line_offset(push_amount)
        else:
            # Player 2 popped bubbles, push toward player 1 (negative offset)
            # This subtracts from the current offset, making it relative to current position
            self.update_center_line_offset(-push_amount)

    def get_valid_targets(self, player_num):
        """Get all valid target positions where a bubble can be placed AND can be reached by shooting"""
        # Convert grid format for shared function
        grid_dict = {}
        for (row, col, p), bubble in self.grid.items():
            if p == player_num:
                grid_dict[(row, col)] = 0  # Color doesn't matter for placement validation
        
        # Use shared function for consistency
        return get_valid_targets(grid_dict, player_num, self.center_line_offset)
    
    def get_valid_targets_constrained(self, player_num):
        """Valid targets filtered by actual reachability (direct LOS or with top/bottom bounce)."""
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        shooter_x, shooter_y = shooter['x'], shooter['y']

        raw_targets = self.get_valid_targets(player_num)
        filtered = []

        for (row, col) in raw_targets:
            key = (row, col, player_num)
            if key not in self.grid_to_screen:
                continue
            target_x, target_y = self.grid_to_screen[key]

            # 1) Fast-path: direct line-of-sight without intersecting own bubbles
            if self._has_clear_path(player_num, shooter_x, shooter_y, row, col):
                filtered.append((row, col))
                continue

            # 2) Deterministic bounce via wall based on LOS from target to wall
            found = False
            for wall in ("top", "bottom"):
                if self._target_has_los_to_wall(player_num, row, col, wall):
                    ang = self._compute_bounce_angle_via_wall(player_num, shooter_x, shooter_y, row, col, wall)
                    if ang is not None:
                        pred_x, pred_y = self.predict_bubble_landing(shooter_x, shooter_y, ang, player_num)
                        nearest = self.find_nearest_grid_position(pred_x, pred_y)
                        if nearest is not None:
                            r, c, p = nearest
                            if p == player_num and r == row and c == col:
                                filtered.append((row, col))
                                found = True
                                break
            if found:
                continue

        return filtered

    def _target_has_los_to_wall(self, player_num: int, target_row: int, target_col: int, wall: str) -> bool:
        key = (target_row, target_col, player_num)
        if key not in self.grid_to_screen:
            return False
        tx, ty = self.grid_to_screen[key]
        wy = 0 if wall == "top" else SCREEN_HEIGHT
        ax, ay = tx, ty
        bx, by = tx, wy
        abx, aby = bx - ax, by - ay
        ab_len2 = max(abx * abx + aby * aby, 1e-6)
        corridor = BUBBLE_RADIUS * 0.85
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            if row == target_row and col == target_col:
                continue
            cx, cy = bubble.x, bubble.y
            t = ((cx - ax) * abx + (cy - ay) * aby) / ab_len2
            t = max(0.0, min(1.0, t))
            px, py = ax + t * abx, ay + t * aby
            dx, dy = cx - px, cy - py
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < corridor:
                return False
        return True

    def _compute_bounce_angle_via_wall(self, player_num: int, shooter_x: float, shooter_y: float,
                                       target_row: int, target_col: int, wall: str) -> Optional[float]:
        key = (target_row, target_col, player_num)
        if key not in self.grid_to_screen:
            return None
        tx, ty = self.grid_to_screen[key]
        my = -ty if wall == "top" else (2 * SCREEN_HEIGHT - ty)
        base_angle = math.degrees(math.atan2(my - shooter_y, tx - shooter_x))
        a = (base_angle % 360.0 + 360.0) % 360.0
        if player_num == 2 and not (60.0 <= a <= 300.0):
            return None
        return base_angle

    def _has_clear_path(self, player_num: int, shooter_x: float, shooter_y: float, target_row: int, target_col: int) -> bool:
        """Check if the straight segment from shooter to target cell center passes near any existing bubble.
        Allows passing near the intended landing cell itself."""
        key = (target_row, target_col, player_num)
        if key not in self.grid_to_screen:
            return False
        tx, ty = self.grid_to_screen[key]
        ax, ay = shooter_x, shooter_y
        bx, by = tx, ty
        abx, aby = bx - ax, by - ay
        ab_len2 = max(abx * abx + aby * aby, 1e-6)
        corridor = BUBBLE_RADIUS * 0.85
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            cx, cy = bubble.x, bubble.y
            # Projection of C onto segment AB
            t = ((cx - ax) * abx + (cy - ay) * aby) / ab_len2
            t = max(0.0, min(1.0, t))
            px, py = ax + t * abx, ay + t * aby
            dx, dy = cx - px, cy - py
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < corridor:
                # allow passing near the landing cell itself
                if not (row == target_row and col == target_col):
                    return False
        return True

    def _candidate_angles_to_target(self, shooter_x: float, shooter_y: float, target_x: float, target_y: float, player_num: int):
        """Return a small set of angles that can reach target: direct, top-bounce, bottom-bounce (mirror trick)."""
        angles = []
        # Direct line
        angles.append(math.degrees(math.atan2(target_y - shooter_y, target_x - shooter_x)))
        # Mirror across top wall (y=0)
        mirror_top_y = -target_y
        angles.append(math.degrees(math.atan2(mirror_top_y - shooter_y, target_x - shooter_x)))
        # Mirror across bottom wall (y=SCREEN_HEIGHT)
        mirror_bottom_y = 2 * SCREEN_HEIGHT - target_y
        angles.append(math.degrees(math.atan2(mirror_bottom_y - shooter_y, target_x - shooter_x)))
        return angles

    def is_target_shootable(self, row, col, player_num, shooter_x, shooter_y):
        """
        Check if a target position can actually be hit by shooting from the shooter position.
        This prevents the AI from choosing unreachable targets.
        """
        # Get the target screen coordinates
        if (row, col, player_num) not in self.grid_to_screen:
            return False
            
        target_x, target_y = self.grid_to_screen[(row, col, player_num)]
        
        # Calculate the angle needed to hit this target
        dx = target_x - shooter_x
        dy = target_y - shooter_y
        required_angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize angle to 0-360 range
        if required_angle < 0:
            required_angle += 360
        
        # For player 1 (left side): valid angles are roughly -90 to 90 degrees
        # For player 2 (right side): valid angles are roughly 90 to 270 degrees
        if player_num == 1:
            # Left player can shoot right (towards center)
            valid_angle_range = (-90, 90)
        else:
            # Right player can shoot left (towards center)
            valid_angle_range = (90, 270)
        
        # Check if the required angle is within the valid range
        if valid_angle_range[0] <= required_angle <= valid_angle_range[1]:
            return True
        
        # If not directly shootable, check if it can be hit with a bounce
        # This is a simplified check - in reality, we'd need to calculate bounce trajectories
        if player_num == 1:
            # Left player can bounce off left wall
            bounce_angle = 180 - required_angle
            if valid_angle_range[0] <= bounce_angle <= valid_angle_range[1]:
                return True
        else:
            # Right player can bounce off right wall
            bounce_angle = 180 - required_angle
            if valid_angle_range[0] <= bounce_angle <= valid_angle_range[1]:
                return True
        
        return False

    def predict_bubble_landing(self, shooter_x, shooter_y, angle, player_num):
        """
        Predict where a bubble will land when shot at a specific angle.
        This helps debug the difference between AI's target and where the bubble actually goes.
        Enhanced to match training environment's bounce simulation.
        """
        # Simulate bubble movement with proper bounce physics
        x, y = shooter_x, shooter_y
        velocity_x = math.cos(math.radians(angle)) * SHOOT_SPEED
        velocity_y = math.sin(math.radians(angle)) * SHOOT_SPEED
        
        # Track bounce count and path
        bounce_count = 0
        max_bounces = 3  # Limit bounces to prevent infinite loops
        path_points = [(x, y)]
        
        # Move bubble until it hits something or bounces too many times
        max_steps = 1000  # Prevent infinite loops
        for step in range(max_steps):
            x += velocity_x
            y += velocity_y
            path_points.append((x, y))
            
            # Check wall collision (top/bottom)
            if y - BUBBLE_RADIUS <= 0 or y + BUBBLE_RADIUS >= SCREEN_HEIGHT:
                velocity_y *= -1
                bounce_count += 1
                if bounce_count >= max_bounces:
                    # Too many bounces, return current position
                    return x, y
                continue
            
            # Check collision with other bubbles
            for bubble in self.bubbles:
                if not bubble.is_moving and bubble.grid_pos is not None:
                    dx = x - bubble.x
                    dy = y - bubble.y
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance < BUBBLE_RADIUS * 2:
                        # Found collision, return the position where it would snap
                        snap_x = bubble.x + (dx / distance) * BUBBLE_RADIUS * 2
                        snap_y = bubble.y + (dy / distance) * BUBBLE_RADIUS * 2
                        return snap_x, snap_y
            
            # Check if bubble has reached the center line
            current_center_x = SCREEN_WIDTH // 2 + self.center_line_offset
            if (player_num == 1 and x + BUBBLE_RADIUS >= current_center_x) or \
               (player_num == 2 and x - BUBBLE_RADIUS <= current_center_x):
                return (current_center_x - BUBBLE_RADIUS if player_num == 1 else current_center_x + BUBBLE_RADIUS), y
        
        # If we reach here, the bubble didn't hit anything (shouldn't happen)
        return x, y

    def _refine_angle_to_hit_target(self, base_angle: float, player_num: int, shooter_x: float, shooter_y: float,
                                    target_row: int, target_col: int) -> float:
        """
        Nudge the angle by small increments to avoid near-collisions that cause unintended snapping.
        Tries a symmetric set of small offsets around the base angle and picks the first that lands on the exact target.
        Mirrors the training environment's refinement logic.
        """
        if base_angle is None:
            return base_angle
        # Candidate small offsets (degrees)
        offsets = [0, -2, 2, -4, 4, -6, 6]
        for off in offsets:
            ang = base_angle + off
            # Predict landing for this angle
            px, py = self.predict_bubble_landing(shooter_x, shooter_y, ang, player_num)
            nearest = self.find_nearest_grid_position(px, py)
            if nearest is None:
                continue
            r, c, p = nearest
            if p == player_num and r == target_row and c == target_col:
                return ang
        return base_angle
    
    def is_valid_placement(self, row, col, player_num):
        """Check if a position is valid for bubble placement"""
        # Must be connected to existing bubbles or the top
        if row == 0:  # Top row is always valid
            return True
        
        # Check if connected to existing bubbles
        if row % 2 == 0:  # Even row
            adjacent = [(row-1, col), (row, col-1), (row, col+1), (row-1, col-1)]
        else:  # Odd row
            adjacent = [(row-1, col), (row, col-1), (row, col+1), (row-1, col+1)]
        
        for nr, nc in adjacent:
            if (nr, nc, player_num) in self.grid:
                return True
        
        return False

    def encode_dqn_state(self):
        """
        FIXED: Encodes the current game state using only CURRENT bubble color (1501 dimensions).
        AI should NOT see next bubble color to prevent confusion between current vs next!
        """
        # Convert RGB colors to indices for the shared function
        ai_grid = {}
        for bubble in self.bubbles:
            if bubble.grid_pos is not None:
                row, col, player = bubble.grid_pos
                if player == 2:  # AI side grid only
                    try:
                        color_idx = BUBBLE_COLORS.index(bubble.color)
                        ai_grid[(row, col)] = color_idx
                    except ValueError:
                        continue
        
        # FIXED: Use encoding with only CURRENT bubble color (no next bubble confusion)
        from bubble_geometry import encode_compact_state_consistent, neighbor_same_color_counts
        
        shooter = self.shooter_two
        current_bubble_color = BUBBLE_COLORS.index(shooter['current_bubble'].color) if shooter['current_bubble'] else 0
        # REMOVED: next_bubble_color - AI should NOT see next bubble color
        
        # Get clean state with only current bubble color (701 dimensions)
        compact_state = encode_compact_state_consistent(ai_grid, current_bubble_color)
        neighbor_counts = neighbor_same_color_counts(ai_grid, current_bubble_color)
        
        # Add strategic features
        strategic_features = self.calculate_strategic_features(2)
        
        # Combine all features: compact_state + neighbor_counts + features
        # Result: 701 + 700 + 3 = 1404 dimensions (no next bubble confusion!)
        return np.concatenate([
            compact_state,           # 701 (700 colors + current)
            neighbor_counts,         # 700 neighbor same-color counts for current bubble color
            strategic_features       # 100 strategic features
        ])

    def calculate_strategic_features(self, player_num):
        """Calculate strategic features for the AI (simplified version, no center line pressure)"""
        features = np.zeros(3, dtype=np.float32)
        # Count AI bubbles and opponent bubbles
        ai_bubbles = 0
        opponent_bubbles = 0
        for bubble in self.bubbles:
            if bubble.grid_pos is not None:
                row, col, bubble_player = bubble.grid_pos
                if bubble_player == player_num:
                    ai_bubbles += 1
                else:
                    opponent_bubbles += 1
        # Basic strategic features
        features[0] = min(ai_bubbles / 50.0, 1.0)  # AI bubble density
        features[1] = min(opponent_bubbles / 50.0, 1.0)  # Opponent bubble density
        # features[2] = self.center_line_offset / MAX_CENTER_LINE_OFFSET  # Removed
        features[2] = self.score_player_two / 100.0 if player_num == 2 else self.score_player_one / 100.0  # Score
        return features

    def calculate_grid_hit_angle(self, shooter_x, shooter_y, target_x, target_y, target_row, target_col):
        """
        Calculate the exact shooting angle needed to hit a specific grid position.
        This function properly handles the honeycomb grid layout and calculates angles
        that will actually hit the target position.
        """
        # For the right-side AI player (player 2), we need to shoot LEFT (towards center)
        # The valid shooting angles are roughly 90-270 degrees
        
        # Calculate the direct angle to the target
        dx = target_x - shooter_x
        dy = target_y - shooter_y
        
        # Calculate the base angle to the target
        base_angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize angle to 0-360 range
        if base_angle < 0:
            base_angle += 360
        
        # For the right-side AI, we want to shoot LEFT (180° ± range)
        # The AI should primarily shoot in the 90-270 degree range
        
        # If the base angle is already in the valid range (90-270), use it
        if 90 <= base_angle <= 270:
            return base_angle
        
        # If the angle is outside the valid range, we need to find a valid shooting angle
        # that will still hit the target
        
        # Case 1: Angle is too far right (0-90 degrees)
        if 0 <= base_angle < 90:
            # We can't shoot right, so we need to bounce off the right wall
            # Calculate the angle that would hit the target after bouncing
            # This is a simplified bounce calculation
            bounce_angle = 180 - base_angle
            if 90 <= bounce_angle <= 270:
                return bounce_angle
            else:
                # Fallback: shoot at a good leftward angle
                return 180
        
        # Case 2: Angle is too far left (270-360 degrees)
        elif 270 < base_angle <= 360:
            # We can't shoot left, so we need to bounce off the left wall
            # Calculate the angle that would hit the target after bouncing
            bounce_angle = 180 - (base_angle - 360)
            if 90 <= bounce_angle <= 270:
                return bounce_angle
            else:
                # Fallback: shoot at a good leftward angle
                return 180
        
        # Fallback: ensure angle is in valid range
        return max(90, min(270, base_angle))

    def find_best_angle_to_target(self, target_row: int, target_col: int, player_num: int, shooter_x: float, shooter_y: float) -> float:
        """Find an angle that lands on the desired target using direct/top/bottom-bounce candidates.
        Returns a best-effort angle; prefers exact landing on target, otherwise the closest.
        """
        key = (target_row, target_col, player_num)
        if key not in self.grid_to_screen:
            # Fallback to direct atan2 if mapping missing
            # Approximate screen coords
            middle_x = SCREEN_WIDTH // 2 + self.center_line_offset
            start_y = BUBBLE_RADIUS * 2
            row_offset = BUBBLE_RADIUS if target_row % 2 == 1 else 0
            target_x = middle_x + (target_col * BUBBLE_RADIUS * 2) + row_offset + BUBBLE_RADIUS
            target_y = start_y + (target_row * BUBBLE_RADIUS * 1.8)
        else:
            target_x, target_y = self.grid_to_screen[key]

        # Candidate angles: direct + mirrored top/bottom, with small offsets
        bases = self._candidate_angles_to_target(shooter_x, shooter_y, target_x, target_y, player_num) if hasattr(self, '_candidate_angles_to_target') else [math.degrees(math.atan2(target_y - shooter_y, target_x - shooter_x))]
        offsets = [-10, -6, -3, 0, 3, 6, 10]

        best_angle = None
        best_err = float('inf')

        # Precompute target center
        tx, ty = target_x, target_y

        for base in bases:
            for off in offsets:
                angle = base + off
                # Keep right-player angles roughly valid (90-270)
                if player_num == 2 and not (90 - 30 <= ((angle % 360) if angle >= 0 else (angle % 360 + 360)) <= 270 + 30):
                    # allow small slack but skip extreme angles
                    continue
                px, py = self.predict_bubble_landing(shooter_x, shooter_y, angle, player_num)
                nearest = self.find_nearest_grid_position(px, py)
                if nearest is None:
                    continue
                r, c, p = nearest
                if p != player_num:
                    continue
                if r == target_row and c == target_col:
                    return angle
                # Track closest landing to target center
                err = (tx - px) * (tx - px) + (ty - py) * (ty - py)
                if err < best_err:
                    best_err = err
                    best_angle = angle

        if best_angle is not None:
            return best_angle

        # Fallback to direct angle
        return self.calculate_grid_hit_angle(shooter_x, shooter_y, target_x, target_y, target_row, target_col)

    def draw_ai_debug_panel(self):
        """Draw a debug panel showing AI information"""
        if not hasattr(self, 'last_ai_target'):
            return
            
        # Create a semi-transparent debug panel on the LEFT side
        panel_width = 350  # Made wider for better readability
        panel_height = 280  # Made taller to fit all information
        panel_x = 10  # Left side
        panel_y = 60  # Moved up slightly to avoid overlapping with score
        
        # Draw panel background with better visibility
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 200))  # More opaque for better readability
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw panel border with better visibility
        pygame.draw.rect(self.screen, (0, 255, 0), 
                        (panel_x, panel_y, panel_width, panel_height), 3)  # Green border, thicker
        
        # Draw panel title
        font_title = pygame.font.Font(None, 28)
        title_text = font_title.render("AI DEBUG PANEL", True, (0, 255, 0))
        self.screen.blit(title_text, (panel_x + 10, panel_y + 5))
        
        # Draw debug information
        font = pygame.font.Font(None, 22)  # Slightly smaller font to fit more info
        y_offset = panel_y + 35  # Start below title
        
        # AI Target Info
        target_x, target_y = self.last_ai_target
        target_text = font.render(f"AI Target: ({int(target_x)}, {int(target_y)})", True, (255, 255, 255))
        self.screen.blit(target_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # AI Angle Info
        angle_text = font.render(f"Grid Hit Angle: {self.shooter_two['angle']:.1f}°", True, (255, 255, 255))
        self.screen.blit(angle_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Show angle calculation method
        method_text = font.render("Using: Grid Hit Calculation", True, (0, 255, 0))
        self.screen.blit(method_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # NEW: Show predicted landing vs target (only if available)
        if hasattr(self, 'last_predicted_landing') and self.last_predicted_landing is not None:
            pred_x, pred_y = self.last_predicted_landing
            landing_text = font.render(f"Predicted: ({int(pred_x)}, {int(pred_y)})", True, (0, 0, 255))
            self.screen.blit(landing_text, (panel_x + 10, y_offset))
            y_offset += 25
            
            # Calculate and show error
            target_x, target_y = self.last_ai_target
            error = math.sqrt((target_x - pred_x)**2 + (target_y - pred_y)**2)
            error_text = font.render(f"Landing Error: {error:.0f}px", True, (255, 0, 255))
            self.screen.blit(error_text, (panel_x + 10, y_offset))
            y_offset += 25
        
        # Shooter Position
        shooter_text = font.render(f"Shooter: ({int(self.shooter_two['x'])}, {int(self.shooter_two['y'])})", True, (255, 255, 255))
        self.screen.blit(shooter_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Current Bubble Info
        if self.shooter_two['current_bubble']:
            color_text = font.render(f"Current: {self.shooter_two['current_bubble'].color}", True, (255, 255, 255))
            self.screen.blit(color_text, (panel_x + 10, y_offset))
            y_offset += 25
        
        # FIXED: AI only sees current bubble color (no next bubble planning)
        next_info_text = font.render("AI Strategy: Current Color Only", True, (0, 255, 0))
        self.screen.blit(next_info_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Valid Targets Count
        valid_targets = self.get_valid_targets_constrained(2)
        targets_text = font.render(f"Valid Targets: {len(valid_targets)}", True, (255, 255, 255))
        self.screen.blit(targets_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Show target selection info
        if hasattr(self, 'last_ai_target'):
            # Find which target number was selected
            target_x, target_y = self.last_ai_target
            selected_target_num = None
            for i, (row, col) in enumerate(valid_targets):
                if (row, col, 2) in self.grid_to_screen:
                    grid_x, grid_y = self.grid_to_screen[(row, col, 2)]
                    if abs(target_x - grid_x) < 10 and abs(target_y - grid_y) < 10:
                        selected_target_num = i + 1
                        break
            
            if selected_target_num:
                selection_text = font.render(f"Selected Target #{selected_target_num}", True, (255, 255, 0))
                self.screen.blit(selection_text, (panel_x + 10, y_offset))
                y_offset += 25
                
                # Show target coordinates
                target_coords = font.render(f"Grid Pos: ({row}, {col})", True, (255, 255, 255))
                self.screen.blit(target_coords, (panel_x + 10, y_offset))
                y_offset += 25
        
        # Add grid statistics
        y_offset += 10  # Add some spacing
        grid_stats_title = font.render("GRID STATISTICS:", True, (255, 255, 0))
        self.screen.blit(grid_stats_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        # Count bubbles by color
        color_counts = {}
        for bubble in self.bubbles:
            if bubble.grid_pos and bubble.grid_pos[2] == 2:  # AI side only
                color = bubble.color
                color_counts[color] = color_counts.get(color, 0) + 1
        
        # Show color distribution
        for i, (color, count) in enumerate(list(color_counts.items())[:3]):  # Show top 3 colors
            color_name = f"Color {BUBBLE_COLORS.index(color) + 1}"
            color_text = font.render(f"{color_name}: {count}", True, (255, 255, 255))
            self.screen.blit(color_text, (panel_x + 10, y_offset))
            y_offset += 18
        
        # Show total AI bubbles
        total_ai_bubbles = len([b for b in self.bubbles if b.grid_pos and b.grid_pos[2] == 2])
        total_text = font.render(f"Total AI Bubbles: {total_ai_bubbles}", True, (0, 255, 255))
        self.screen.blit(total_text, (panel_x + 10, y_offset))
        y_offset += 20
        
        # Show current score
        score_text = font.render(f"AI Score: {self.score_player_two}", True, (255, 255, 0))
        self.screen.blit(score_text, (panel_x + 10, y_offset))

    # REMOVED: Unused color-based functions from old approach
    # - get_available_colors_for_ai(): AI no longer analyzes available colors
    # - find_best_position_for_color(): AI no longer needs color-based positioning
    # 
    # NEW APPROACH: AI gets random colored bubbles and chooses targets directly
    # based on color analysis learned through experience and rewards.
    
    def get_valid_targets(self, player_num):
        """Get all valid target positions where a bubble can be placed AND can be reached by shooting"""
        # Convert grid format for shared function
        grid_dict = {}
        for (row, col, p), bubble in self.grid.items():
            if p == player_num:
                grid_dict[(row, col)] = 0  # Color doesn't matter for placement validation
        
        # Use shared function for consistency
        return get_valid_targets(grid_dict, player_num, self.center_line_offset)
    
    def get_valid_targets_constrained(self, player_num):
        """Valid targets filtered by actual reachability (direct or via bounce), no cone restriction."""
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        shooter_x, shooter_y = shooter['x'], shooter['y']
        
        raw_targets = self.get_valid_targets(player_num)
        filtered = []
        
        for (row, col) in raw_targets:
            if (row, col, player_num) not in self.grid_to_screen:
                continue
            target_x, target_y = self.grid_to_screen[(row, col, player_num)]
            # 1) Fast-path: direct line-of-sight without intersecting own bubbles
            if self._has_clear_path(player_num, shooter_x, shooter_y, row, col):
                filtered.append((row, col))
                continue

            # 2) Precise bounce validation using physics simulation + expanded angle sampling
            if self._has_bounce_path_to_target(player_num, shooter_x, shooter_y, row, col):
                filtered.append((row, col))
        
        return filtered

    def get_reachable_bubble_targets(self, player_num):
        """Return (row,col) bubbles reachable by simple LOS rules: direct shooter→bubble or bubble→wall corridor."""
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        shooter_x, shooter_y = shooter['x'], shooter['y']
        targets = []
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            if (row, col, player_num) not in self.grid_to_screen:
                continue
            if self._has_clear_path(player_num, shooter_x, shooter_y, row, col):
                targets.append((row, col))
                continue
            for wall in ("top", "bottom"):
                if self._target_has_los_to_wall(player_num, row, col, wall):
                    targets.append((row, col))
                    break
        return targets

    def get_direct_los_bubble_targets(self, player_num):
        """Return (row,col) bubbles that have direct shooter→bubble line-of-sight (no bounce)."""
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        shooter_x, shooter_y = shooter['x'], shooter['y']
        targets = []
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            if (row, col, player_num) not in self.grid_to_screen:
                continue
            if self._has_clear_path(player_num, shooter_x, shooter_y, row, col):
                targets.append((row, col))
        return targets

    def get_front_bubble_targets(self, player_num):
        """Return only the far-most (maximum col) occupied bubble in each row for the given player.
        Masks all empty cells implicitly by selecting from occupied grid only."""
        front_by_row = {}
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            if (row not in front_by_row) or (col > front_by_row[row]):
                front_by_row[row] = col
        return [(row, col) for row, col in front_by_row.items()]

    def _has_clear_path_to_bubble(self, player_num: int, shooter_x: float, shooter_y: float, target_row: int, target_col: int) -> bool:
        """Alias to existing LOS checker for clarity when aiming at bubbles."""
        return self._has_clear_path(player_num, shooter_x, shooter_y, target_row, target_col)

    def _is_bubble_reachable(self, player_num: int, shooter_x: float, shooter_y: float, target_row: int, target_col: int) -> bool:
        """Sample a small set of candidate angles (direct + top/bottom mirror with offsets)
        and simulate to see if the predicted landing snaps to a neighbor cell of the target bubble.
        """
        key = (target_row, target_col, player_num)
        if key not in self.grid_to_screen:
            return False
        tx, ty = self.grid_to_screen[key]
        bases = []
        # direct
        bases.append(math.degrees(math.atan2(ty - shooter_y, tx - shooter_x)))
        # mirror top
        bases.append(math.degrees(math.atan2(-ty - shooter_y, tx - shooter_x)))
        # mirror bottom
        bases.append(math.degrees(math.atan2((2 * SCREEN_HEIGHT - ty) - shooter_y, tx - shooter_x)))
        offsets = [-8, -4, 0, 4, 8]
        # Precompute neighbor set of target bubble
        neighbors = set()
        for nr, nc in get_adjacent_positions(target_row, target_col):
            if 0 <= nr < 20 and 0 <= nc < 35:  # GRID_ROWS, GRID_COLS
                if (nr, nc, player_num) not in self.grid:
                    neighbors.add((nr, nc))
        if not neighbors:
            return False
        # Try candidates
        for base in bases:
            for off in offsets:
                ang = base + off
                # Keep angle roughly valid for right player
                a = (ang % 360.0 + 360.0) % 360.0
                if player_num == 2 and not (60.0 <= a <= 300.0):
                    continue
                px, py = self.predict_bubble_landing(shooter_x, shooter_y, ang, player_num)
                nearest = self.find_nearest_grid_position(px, py)
                if nearest is None:
                    continue
                r, c, p = nearest
                if p != player_num:
                    continue
                if (r, c) in neighbors:
                    return True
        return False

    def _has_bounce_path_to_target(self, player_num: int, shooter_x: float, shooter_y: float, target_row: int, target_col: int) -> bool:
        """Return True if a top/bottom-bounce shot can land exactly on (target_row, target_col).
        Uses mirror-trick bases, expanded angle offsets, and precise physics simulation.
        """
        key = (target_row, target_col, player_num)
        if key not in self.grid_to_screen:
            return False
        target_x, target_y = self.grid_to_screen[key]

        # Candidate bases: direct + mirrored top/bottom
        bases = self._candidate_angles_to_target(shooter_x, shooter_y, target_x, target_y, player_num)
        # Expanded, finer sampling around each base angle
        offsets = [-12, -9, -6, -3, 0, 3, 6, 9, 12]

        # For right player, keep angles roughly in [90, 270]; for left, in [-90, 90] (modulo 360)
        def _angle_ok_for_player(a: float) -> bool:
            a = (a % 360.0 + 360.0) % 360.0
            if player_num == 2:
                return 60.0 <= a <= 300.0  # generous but excludes extreme backward shots
            else:
                return (a <= 120.0) or (a >= 240.0)

        for base in bases:
            for off in offsets:
                ang = base + off
                if not _angle_ok_for_player(ang):
                    continue
                # Convert angle to a far aim point along this ray
                ray_len = 2000.0
                aim_x = shooter_x + math.cos(math.radians(ang)) * ray_len
                aim_y = shooter_y + math.sin(math.radians(ang)) * ray_len
                # Precise simulation to check landing cell
                landing, _, _ = simulate_shot_trajectory(
                    shooter_x, shooter_y, aim_x, aim_y, player_num, self.center_line_offset, max_bounces=3, max_steps=2000
                )
                if landing is not None and landing[0] == target_row and landing[1] == target_col:
                    return True

        return False

    def draw_shooting_cone(self):
        """Draw the shooting cone for the AI player (right side) for debugging."""
        shooter = self.shooter_two
        
        # Draw the shooting cone as a visual guide
        cone_length = 100
        cone_center_angle = 180  # Straight left for AI
        
        # Draw cone center line (straight left)
        center_x = shooter['x'] + math.cos(math.radians(cone_center_angle)) * cone_length
        center_y = shooter['y'] + math.sin(math.radians(cone_center_angle)) * cone_length
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (shooter['x'], shooter['y']), 
                        (center_x, center_y), 3)
        
        # Draw cone boundaries (±45 degrees from center)
        top_angle = cone_center_angle + 45
        bottom_angle = cone_center_angle - 45
        
        top_x = shooter['x'] + math.cos(math.radians(top_angle)) * cone_length
        top_y = shooter['y'] + math.sin(math.radians(top_angle)) * cone_length
        bottom_x = shooter['x'] + math.cos(math.radians(bottom_angle)) * cone_length
        bottom_y = shooter['y'] + math.sin(math.radians(bottom_angle)) * cone_length
        
        # Draw cone boundary lines
        pygame.draw.line(self.screen, (255, 0, 255), 
                        (shooter['x'], shooter['y']), 
                        (top_x, top_y), 2)
        pygame.draw.line(self.screen, (255, 0, 255), 
                        (shooter['x'], shooter['y']), 
                        (bottom_x, bottom_y), 2)
        
        # Add text label
        font = pygame.font.Font(None, 24)
        cone_text = font.render("AI Shooting Cone", True, (255, 255, 255))
        self.screen.blit(cone_text, (shooter['x'] + 60, shooter['y'] - 20))

    def get_hybrid_masked_targets(self, player_num):
        """Get targets using hybrid masking (front-most-per-row + angular LOS)."""
        # Build single-side grid mapping {(row,col): color_index}
        grid_map = {}
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            try:
                color_idx = BUBBLE_COLORS.index(bubble.color)
            except ValueError:
                continue
            grid_map[(row, col)] = color_idx
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        shooter_x, shooter_y = shooter['x'], shooter['y']
        from bubble_geometry import get_hybrid_masked_targets as _hybrid
        return _hybrid(shooter_x, shooter_y, grid_map, player_num, self.center_line_offset)

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