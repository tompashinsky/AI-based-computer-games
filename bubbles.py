import pygame
import os
import sys
import random
import math
from typing import List, Tuple, Optional, Dict
import torch
from bubbles_target_dqn import TargetDQNAgent, decode_target_action, calculate_angle_to_target
import numpy as np

# Initialize Pygame
pygame.init()
pygame.event.clear()


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
LOSING_LINE_COLOR = (255, 255, 255)  # White for better visibility on dark bg

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
# Temporary aiming config
# When False, the AI will only consider direct line-of-sight targets and will not
# attempt pre-shot bump adjustments that can leverage bounces.
AI_ENABLE_BOUNCE = False

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
        self.move_frames = 0  # Frames moved while is_moving

    def draw(self, screen: pygame.Surface):
        try:
            # Expect Game to have cached images by RGB color
            from pygame import Rect  # local import ok
            img = getattr(Bubble, "_image_cache", {}).get(self.color)
            if img is None and hasattr(Bubble, "_image_provider") and Bubble._image_provider is not None:
                img = Bubble._image_provider(self.color)
                if getattr(Bubble, "_image_cache", None) is None:
                    Bubble._image_cache = {}
                Bubble._image_cache[self.color] = img
            if img is not None:
                rect = img.get_rect(center=(int(self.x), int(self.y)))
                screen.blit(img, rect)
                return
        except Exception:
            pass
        # Fallback to vector circle if no image available
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius, 2)

    def update(self):
        if self.is_moving and not self.snapped:
            self.x += self.velocity_x
            self.y += self.velocity_y
            self.move_frames += 1

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
        # Load bubble assets
        self._load_bubble_assets()
        # Load background
        try:
            bg_path = os.path.join("assets", "backgrounds", "Blue_Nebula.png")
            bg_img = pygame.image.load(bg_path).convert()
            self.background_image = pygame.transform.smoothscale(bg_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        except Exception:
            self.background_image = None
        
        # Load sound effects
        try:
            sound_path = os.path.join("assets", "sounds", "bubble_boop.wav")
            self.shoot_sound = pygame.mixer.Sound(sound_path)
        except Exception as e:
            print(f"Warning: Could not load shoot sound {sound_path}: {e}")
            self.shoot_sound = None
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
        # Queue clicks so human shot fires ASAP when ready
        self._pending_shot_p1 = False
        # Track how long a human shot has been blocked by a moving human bubble
        self._p1_block_frames = 0
        # Input tracking removed: rely on MOUSEBUTTONDOWN only to avoid double fires
        # Ensure mouse button events are allowed (avoid accidental filtering)
        try:
            pygame.event.set_allowed(None)
        except Exception:
            pass

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

    def _create_circle_surface(self, diameter: int, color: tuple) -> pygame.Surface:
        surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        r = diameter // 2
        pygame.draw.circle(surf, color, (r, r), r)
        pygame.draw.circle(surf, (255, 255, 255), (r, r), r, 2)
        return surf

    def _load_bubble_assets(self):
        # Map color indices (0..5) to filenames by convention
        color_names = ["red", "green", "blue", "yellow", "magenta", "cyan"]
        assets_dir = os.path.join("assets", "bubbles")
        diameter = BUBBLE_RADIUS * 2
        preview_diameter = int(diameter * 0.7)
        self.bubble_images = {}
        self.bubble_preview_images = {}
        for idx, rgb in enumerate(BUBBLE_COLORS):
            name = color_names[idx] if idx < len(color_names) else f"color_{idx}"
            path = os.path.join(assets_dir, f"{name}.png")
            try:
                img = pygame.image.load(path).convert_alpha()
                img = pygame.transform.smoothscale(img, (diameter, diameter))
            except Exception:
                img = self._create_circle_surface(diameter, rgb)
            try:
                prev = pygame.image.load(path).convert_alpha()
                prev = pygame.transform.smoothscale(prev, (preview_diameter, preview_diameter))
            except Exception:
                prev = self._create_circle_surface(preview_diameter, rgb)
            self.bubble_images[rgb] = img
            self.bubble_preview_images[rgb] = prev

    def _ensure_shooter_ready(self, player_num: int):
        """Ensure the given player's shooter has a current and next bubble ready."""
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        # If current is missing, promote next or generate new
        if not shooter.get('current_bubble'):
            if shooter.get('next_bubble'):
                shooter['current_bubble'] = shooter['next_bubble']
                shooter['current_bubble'].x = shooter['x']
                shooter['current_bubble'].y = shooter['y']
                # Generate new next
                next_color = random.choice(BUBBLE_COLORS)
                nx = shooter['x'] + (BUBBLE_RADIUS * 2.5 if player_num == 1 else -BUBBLE_RADIUS * 2.5)
                shooter['next_bubble'] = Bubble(nx, shooter['y'], next_color)
            else:
                # Generate both current and next
                color = random.choice(BUBBLE_COLORS)
                shooter['current_bubble'] = Bubble(shooter['x'], shooter['y'], color)
                next_color = random.choice(BUBBLE_COLORS)
                nx = shooter['x'] + (BUBBLE_RADIUS * 2.5 if player_num == 1 else -BUBBLE_RADIUS * 2.5)
                shooter['next_bubble'] = Bubble(nx, shooter['y'], next_color)

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
        # Only shoot if called explicitly for a player; moving-bubble gating handled by caller
        if force_player is not None:
            shooter = self.shooter_one if force_player == 1 else self.shooter_two
            
            if shooter['current_bubble']:
                shooter['current_bubble'].is_moving = True
                shooter['current_bubble'].velocity_x = math.cos(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].velocity_y = math.sin(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].shot_by_player = force_player  # Track which player shot this bubble
                self.bubbles.append(shooter['current_bubble'])
                
                # Play shoot sound effect
                if self.shoot_sound:
                    self.shoot_sound.play()
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
                return True
            return False
        # Legacy turn-based fallback
        if not any(b.is_moving for b in self.bubbles):
            shooter = self.shooter_one if self.current_player == 1 else self.shooter_two
            if shooter['current_bubble']:
                shooter['current_bubble'].is_moving = True
                shooter['current_bubble'].velocity_x = math.cos(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].velocity_y = math.sin(math.radians(angle)) * SHOOT_SPEED
                shooter['current_bubble'].shot_by_player = self.current_player  # Track which player shot this bubble
                self.bubbles.append(shooter['current_bubble'])
                
                # Play shoot sound effect
                if self.shoot_sound:
                    self.shoot_sound.play()
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
                # Fire on left button down only
                if getattr(event, 'button', 1) == 1:
                    print("Mouse click received")
                    self._ensure_shooter_ready(1)
                    if self.shooter_one.get('current_bubble') is not None:
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
        # Background
        if self.background_image is not None:
            self.screen.blit(self.background_image, (0, 0))
        else:
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
        
        # DEBUG: Draw per-empty-cell same-color counts for CURRENT bubble color on AI side
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
                # Current shooter 2 color index
                try:
                    current_color_idx = BUBBLE_COLORS.index(self.shooter_two['current_bubble'].color) if self.shooter_two['current_bubble'] else 0
                except ValueError:
                    current_color_idx = 0
                # Compute per-cell counts (occupied cells will be 0)
                from bubble_geometry import neighbor_same_color_counts as _ncounts
                counts = _ncounts(ai_grid, current_color_idx)
                # Use last reachable targets if available to avoid clutter, else recompute
                reachable = getattr(self, 'last_reachable_targets', None)
                if reachable is None:
                    reachable = self.get_valid_targets_constrained(2)
                font_cnt = pygame.font.Font(None, 18)
                for (row, col) in reachable:
                    idx = row * 35 + col
                    if 0 <= idx < counts.shape[0]:
                        key = (row, col, 2)
                        if key not in self.grid_to_screen:
                            continue
                        x, y = self.grid_to_screen[key]
                        val = int(counts[idx])
                        color = (0, 255, 255) if val > 0 else (160, 160, 160)
                        txt = font_cnt.render(str(val), True, color)
                        self.screen.blit(txt, (int(x) - 6, int(y) - (BUBBLE_RADIUS + 18)))
            except Exception:
                pass
        
        # DEBUG: Draw per-cell Q-values on the AI grid (player 2) using the SAME reachable target set as selection
        if DEBUG_AI and hasattr(self, 'last_q_values') and self.last_q_values is not None:
            font_small = pygame.font.Font(None, 20)
            font_big = pygame.font.Font(None, 28)
            best_idx = getattr(self, 'last_best_action_idx', None)
            # Use the exact reachable targets used at selection time, if available
            reachable = getattr(self, 'last_reachable_targets', None)
            if reachable is None:
                reachable = []
            for (row, col) in reachable:
                idx = row * 35 + col
                if idx < 0 or idx >= len(self.last_q_values):
                    continue
                q_val = float(self.last_q_values[idx])
                if not math.isfinite(q_val):
                    continue
                key = (row, col, 2)
                if key not in self.grid_to_screen:
                    continue
                x, y = self.grid_to_screen[key]
                label_y = int(y) - (BUBBLE_RADIUS + 20)
                is_best = (best_idx is not None and idx == best_idx)
                text_color = (255, 255, 255) if not is_best else (255, 215, 0)
                font = font_big if is_best else font_small
                text_str = f"{q_val:.2f}"
                txt = font.render(text_str, True, text_color)
                pad_x, pad_y = 4, 2
                tx = int(x) - txt.get_width() // 2
                ty = label_y - txt.get_height() // 2
                bg_rect = pygame.Rect(tx - pad_x, ty - pad_y, txt.get_width() + 2 * pad_x, txt.get_height() + 2 * pad_y)
                pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
                self.screen.blit(txt, (tx, ty))
            # Additionally, outline the chosen action if available
            if hasattr(self, 'last_chosen_action_idx') and self.last_chosen_action_idx is not None:
                try:
                    cidx = int(self.last_chosen_action_idx)
                    crow, ccol = cidx // 35, cidx % 35
                    key = (crow, ccol, 2)
                    if key in self.grid_to_screen:
                        cx, cy = self.grid_to_screen[key]
                        pygame.draw.circle(self.screen, (255, 0, 0), (int(cx), int(cy)), 14, 2)
                except Exception:
                    pass
        
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
                # Supply image provider to Bubble for asset lookup
                def _provider(rgb):
                    return self.bubble_images.get(rgb)
                Bubble._image_provider = _provider
                shooter['current_bubble'].draw(self.screen)
            
            # Draw next bubble preview (smaller)
            if shooter['next_bubble']:
                prev_img = self.bubble_preview_images.get(shooter['next_bubble'].color)
                if prev_img is not None:
                    rect = prev_img.get_rect(center=(int(shooter['next_bubble'].x), int(shooter['next_bubble'].y)))
                    self.screen.blit(prev_img, rect)
                else:
                    # Fallback vector
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
        
        # Draw scores (styled for dark background)
        score_font = pygame.font.Font(None, 60)
        # Colors
        score_fill = (230, 230, 240)
        score_shadow = (20, 20, 30)
        # Player 1
        p1_text = score_font.render(f"{self.score_player_one}", True, score_fill)
        p1_shadow = score_font.render(f"{self.score_player_one}", True, score_shadow)
        self.screen.blit(p1_shadow, (12, 12))
        self.screen.blit(p1_text, (10, 10))
        # Player 2
        p2_text = score_font.render(f"{self.score_player_two}", True, score_fill)
        p2_shadow = score_font.render(f"{self.score_player_two}", True, score_shadow)
        x = SCREEN_WIDTH - p2_text.get_width() - 10
        self.screen.blit(p2_shadow, (x + 2, 12))
        self.screen.blit(p2_text, (x, 10))
        
        # Removed on-screen current player and center offset indicators per request
        
        # DEBUG: Draw AI debug info panel
        if DEBUG_AI:
            self.draw_ai_debug_panel()
        
        if self.game_over:
            font = pygame.font.Font(None, 74)
            text = font.render("Game Over!", True, (255, 255, 255))
            text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()

    def show_game_over_screen(self):
        font = pygame.font.Font(None, 90)
        score_font = pygame.font.Font(None, 70)
        button_font = pygame.font.Font(None, 50)

        # Determine winner and color
        winner = "It's a Draw!"
        color = (255, 215, 0)  # Gold for draw
        if self.score_player_one > self.score_player_two:
            winner = "Player 1 Wins!"
            color = (0, 200, 0)  # Green
        elif self.score_player_two > self.score_player_one:
            winner = "Player 2 Wins!"
            color = (0, 100, 255)  # Blue

        # Dark overlay background
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Winner text
        text = font.render(winner, True, color)
        self.screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 120)))

        # Final score text (Player1 - Player2)
        score_text = score_font.render(f"{self.score_player_one} - {self.score_player_two}", True, (255, 255, 255))
        self.screen.blit(score_text, score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40)))

        # Button rectangles
        restart_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 + 60, 180, 60)
        quit_rect = pygame.Rect(SCREEN_WIDTH // 2 + 20, SCREEN_HEIGHT // 2 + 60, 180, 60)

        running_modal = True
        while running_modal:
            # Draw buttons (with hover effect)
            mouse_pos = pygame.mouse.get_pos()

            for rect, label in [(restart_rect, "Restart"), (quit_rect, "Quit")]:
                if rect.collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=12)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 3, border_radius=12)
                    text_surf = button_font.render(label, True, (0, 0, 0))
                else:
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, border_radius=12)
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 3, border_radius=12)
                    text_surf = button_font.render(label, True, (255, 255, 255))
                self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

            pygame.display.update()

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "back_to_menu"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "back_to_menu"
                    if event.key == pygame.K_r:
                        return "restart"
                elif event.type == pygame.MOUSEBUTTONDOWN and getattr(event, 'button', 1) == 1:
                    if restart_rect.collidepoint(event.pos):
                        return "restart"
                    elif quit_rect.collidepoint(event.pos):
                        return "back_to_menu"

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

            # NEW: Also treat bubbles adjacent to the middle vertical line (col == 0)
            # as anchored seeds, but only for rows that actually touch the center
            # in the honeycomb layout (even rows).
            for row in range(GRID_ROWS):
                if row % 2 == 0:
                    key = (row, 0, player_num)
                    if key in self.grid and key not in connected_to_top:
                        to_check.append(key)
                        connected_to_top.add(key)
        
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

        # Edge-detection fallback removed to avoid double fire; rely on MOUSEBUTTONDOWN events only

        # Human shooting is handled immediately on click (no queue)

        # AI (right side) acts in real time
        if not any(b.is_moving and b.shot_by_player == 2 for b in self.bubbles):
            # Only act if AI's bubble is ready
            if self.shooter_two['current_bubble'] and self.ai_action_cooldown == 0:
                dqn_state = self.encode_dqn_state()
                
                # NEW: Bubble-target action system using hybrid masking
                # Use empty-cell targets: valid placement + direct LOS (no bounce by default)
                reachable_targets = self.get_valid_targets_constrained(2)
                # Store for debug rendering parity with selection
                self.last_reachable_targets = list(reachable_targets)
                
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
                        
                        # Apply masking using the SAME reachable_targets used for selection
                        if reachable_targets:
                            masked_q_values = self.dqn_agent.apply_action_mask(q_values, reachable_targets)
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
                            # No reachable targets - visualize as all -inf
                            masked_q_values = torch.full_like(q_values, float('-inf'))
                            print("No reachable targets - Qs shown as -inf")

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
                
                # Convert target action index to row, col coordinates (empty-cell target)
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
                
                # Get target empty-cell screen position
                if (target_row, target_col, 2) in self.grid_to_screen:
                    target_x, target_y = self.grid_to_screen[(target_row, target_col, 2)]
                else:
                    # Fallback to first reachable target
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

                # Training parity: refine angle first, then simulate without bump
                try:
                    ai_angle = self._refine_angle_to_hit_target(ai_angle, 2, shooter_x, shooter_y, target_row, target_col)
                except Exception:
                    pass

                # Pre-shot simulation & bump disabled when bounces are off
                if AI_ENABLE_BOUNCE:
                    try:
                        px, py = self.predict_bubble_landing(shooter_x, shooter_y, ai_angle, 2)
                        self.last_predicted_landing = (px, py)
                        nearest = self.find_nearest_grid_position(px, py)
                        
                        # Check if simulation failed to hit target
                        if nearest is None or nearest[2] != 2 or nearest[0] != target_row or nearest[1] != target_col:
                            # Simulation failed - apply bump based on verticality
                            rad = math.radians(ai_angle)
                            bump_deg = 6.0 * abs(math.sin(rad))
                            bump_sign = 0.0
                            if 90.0 <= ai_angle <= 270.0:
                                if target_y < shooter_y:
                                    # Upward shot -> increase angle toward 270° (more up)
                                    bump_sign = +1.0
                                else:
                                    # Downward shot -> decrease angle toward 90° (more down)
                                    bump_sign = -1.0
                            
                            # Apply bump and retry simulation (up to 2 tries)
                            tries = 2
                            while tries > 0 and bump_sign != 0.0:
                                ai_angle += bump_sign * bump_deg
                                # Clamp for right-shooter
                                if ai_angle < 90.0:
                                    ai_angle = 90.0
                                elif ai_angle > 270.0:
                                    ai_angle = 270.0
                                
                                # Re-simulate
                                px, py = self.predict_bubble_landing(shooter_x, shooter_y, ai_angle, 2)
                                self.last_predicted_landing = (px, py)
                                nearest = self.find_nearest_grid_position(px, py)
                                
                                # Check if we hit the target now
                                if nearest is not None and nearest[2] == 2 and nearest[0] == target_row and nearest[1] == target_col:
                                    break
                                
                                tries -= 1
                    except Exception:
                        pass
                
                # Store target and angle for visual debugging
                self.last_ai_target = (target_x, target_y)
                self.last_ai_angle = ai_angle
                # last_predicted_landing updated above if simulation ran
                
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
                # Random delay between 0.8 and 1.2 seconds
                delay_seconds = random.uniform(0.8, 1.2)
                self.ai_action_cooldown = int(delay_seconds * 60)  # Convert to frames at 60 FPS
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

    def _reset_for_new_game(self):
        # Reset dynamic game state while retaining loaded assets and AI
        self.bubbles = []
        self.grid = {}
        self.grid_to_screen = {}
        self.score_player_one = 0
        self.score_player_two = 0
        self.center_line_offset = 0
        self.target_center_line_offset = 0
        self.falling_bubbles = []
        self.game_over = False
        # Rebuild grids and starting bubbles/shooters
        self.initialize_grid()
        self.initialize_bubbles()
        self.initialize_shooters()

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
        """Valid targets filtered by actual reachability.
        When AI_ENABLE_BOUNCE is False, only direct line-of-sight targets are allowed.
        """
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        shooter_x, shooter_y = shooter['x'], shooter['y']

        raw_targets = self.get_valid_targets(player_num)
        # Constrain to empty cells adjacent ONLY to the front-most (max col) filled cell per row
        # Build front-most per-row map for this player
        front_by_row = {}
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            prev = front_by_row.get(row)
            if prev is None or col > prev:
                front_by_row[row] = col
        def _adjacent_to_front_only(r, c) -> bool:
            if r not in front_by_row:
                return False
            fcol = front_by_row[r]
            if r % 2 == 0:
                adj = [(r-1, fcol), (r+1, fcol), (r, fcol-1), (r, fcol+1), (r-1, fcol-1), (r+1, fcol-1)]
            else:
                adj = [(r-1, fcol), (r+1, fcol), (r, fcol-1), (r, fcol+1), (r-1, fcol+1), (r+1, fcol+1)]
            # Allow only if the open cell (r,c) is one of these neighbors of the front-most bubble
            return (r, c) in adj
        raw_targets = [(r, c) for (r, c) in raw_targets if _adjacent_to_front_only(r, c)]
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

            # 2) Optional: allow bounce paths only when enabled
            if AI_ENABLE_BOUNCE:
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
        """Valid targets filtered by actual reachability (direct only unless AI_ENABLE_BOUNCE)."""
        shooter = self.shooter_one if player_num == 1 else self.shooter_two
        shooter_x, shooter_y = shooter['x'], shooter['y']
        
        raw_targets = self.get_valid_targets(player_num)
        # Constrain to empty cells adjacent ONLY to the front-most (max col) filled cell per row
        front_by_row = {}
        for (row, col, p), bubble in self.grid.items():
            if p != player_num:
                continue
            prev = front_by_row.get(row)
            if prev is None or col > prev:
                front_by_row[row] = col
        def _adjacent_to_front_only(r, c) -> bool:
            if r not in front_by_row:
                return False
            fcol = front_by_row[r]
            if r % 2 == 0:
                adj = [(r-1, fcol), (r+1, fcol), (r, fcol-1), (r, fcol+1), (r-1, fcol-1), (r+1, fcol-1)]
            else:
                adj = [(r-1, fcol), (r+1, fcol), (r, fcol-1), (r, fcol+1), (r-1, fcol+1), (r+1, fcol+1)]
            return (r, c) in adj
        raw_targets = [(r, c) for (r, c) in raw_targets if _adjacent_to_front_only(r, c)]
        filtered = []
        
        for (row, col) in raw_targets:
            if (row, col, player_num) not in self.grid_to_screen:
                continue
            target_x, target_y = self.grid_to_screen[(row, col, player_num)]
            # 1) Fast-path: direct line-of-sight without intersecting own bubbles
            if self._has_clear_path(player_num, shooter_x, shooter_y, row, col):
                # Verify by simulating angle to ensure landing on intended cell
                try:
                    ang = calculate_angle_to_target(shooter_x, shooter_y, target_x, target_y)
                    if hasattr(self, '_refine_angle_to_hit_target'):
                        ang = self._refine_angle_to_hit_target(ang, player_num, shooter_x, shooter_y, row, col)
                    px, py = self.predict_bubble_landing(shooter_x, shooter_y, ang, player_num)
                    nearest = self.find_nearest_grid_position(px, py)
                    if nearest is not None and nearest[2] == player_num and nearest[0] == row and nearest[1] == col:
                        filtered.append((row, col))
                        continue
                except Exception:
                    pass

            # 2) Optional bounce validation
            if AI_ENABLE_BOUNCE:
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
            # If game over, present modal and handle choice
            if self.game_over:
                choice = self.show_game_over_screen()
                if choice == "restart":
                    self._reset_for_new_game()
                else:
                    running = False

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
    sys.exit() 