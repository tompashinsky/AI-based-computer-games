"""
Single-Player Target-Based DQN Training Environment for Right-Side AI Bubble Shooter

This environment focuses on training the AI to be an expert bubble shooter from the right side.
It removes the left player completely to focus learning on core shooting skills.

Key Features:
- Single right-side grid (20x35 honeycomb pattern)
- Same physics as the actual game (bubbles.py)
- Bubble falling mechanics (isolated bubbles fall off like in real game)
- Enhanced reward system:
  * -1000 penalty for losing (bubbles reach edge)
  * -1 penalty per shot (encourages efficiency)
  * +10 per bubble popped
  * Quadratic bonus for chain reactions (3+ bubbles)
  * +50 bonus for large chains (5+ bubbles)
  * +15 per falling bubble (strategic positioning)
  * +25 bonus for falling cascades (3+ bubbles)
  * +0.1 survival bonus per step

The trained model will work directly in the actual game with identical grid layout and physics.

PERFORMANCE OPTIMIZATION:
- Pygame has been completely removed to speed up training
- To restore debug rendering, uncomment the pygame import and debug_render lines
- Look for "# Removed as per edit hint" comments to find what needs to be restored
- Duplicate target calculations eliminated: get_valid_targets_constrained() now called only once per step
- AI continues playing with random shots when smart targets unavailable - prevents training from getting stuck
- MAJOR PHYSICS OPTIMIZATION: Expensive shot simulation replaced with fast path approximation (10-50x faster)
"""
import numpy as np
import random
import torch
import os
import math
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
try:
    import pygame  # Optional, only used when debug_render=True
except Exception:
    pygame = None
# pygame = None
from bubbles_target_dqn import TargetDQNAgent, decode_target_action, calculate_angle_to_target, TARGET_UPDATE_FREQ

# Import shared geometry module for consistency
from bubble_geometry import (
    GRID_ROWS, GRID_COLS, BUBBLE_COLORS, BUBBLE_COLORS_COUNT,
    SCREEN_WIDTH, SCREEN_HEIGHT, BUBBLE_RADIUS, LOSE_THRESHOLD,
    grid_to_screen, screen_to_grid, is_valid_placement, get_valid_targets,
    check_lose_condition, calculate_angle_to_target,
    encode_compact_state, encode_compact_state_consistent,
    neighbor_same_color_counts, get_hybrid_masked_targets,
    get_angular_los_filtered_targets
)

# Temporary aiming config (match game): when False, training uses direct LOS only and
# skips bump adjustments that can leverage bounces.
AI_ENABLE_BOUNCE = False

# State size calculation using shared constants
# FIXED: Only current bubble color to prevent AI from confusing current vs next
# Reduced strategic features from 100 to 3 (drop 97 unused)
STATE_SIZE = (GRID_ROWS * GRID_COLS) + (GRID_ROWS * GRID_COLS) + 1 + 0 + 0 + 3
# Breakdown: 700 colors + 700 neighbor_counts + current_color + NO_next_color + 3 features
# Total: 700 + 700 + 1 + 0 + 0 + 3 = 1404

# Target Action size: still index by (row,col) but actions now correspond to existing bubbles
TARGET_ACTION_SIZE = GRID_ROWS * GRID_COLS  # keep 700 mapping; mask to occupied cells
COLOR_ACTION_SIZE = BUBBLE_COLORS_COUNT  # Keep for backward compatibility during transition

# IMPORTANT: Learning Strategy
# - AI Player (player 2, right side): Learns with realistic constraints
#   * Shooting cone validation (90-270 degrees)
#   * Upward-only shooting (prevents downward shots)
#   * Target re-sampling for unreachable positions
#   * This ensures AI learns physics that work in the real game
#
# - Human Player (player 1, left side): No constraints
#   * Can make mistakes and learn from them
#   * No angle restrictions
#   * This provides realistic opponent behavior for AI training
#
# This approach ensures the AI model learns realistic physics while maintaining
# a challenging training environment with a human-like opponent.

def encode_state_like_game(grid_player2, shooter_y, shooter_angle, current_color, next_color, ai_player_num):
    """
    Encode state to match bubbles.py encode_dqn_state exactly:
    - Single 700-length plane, 1.0 = AI bubbles, 2.0 = opponent bubbles
    - Shooter info: y, angle (normalized)
    - Current/next color one-hot (size = BUBBLE_COLORS)
    - Strategic features: 100-length (we fill first 3 to match game usage)
    """
    flat_grid = np.zeros(GRID_ROWS * GRID_COLS, dtype=np.float32)
    # Only player 2 exists in this environment
    ai_grid = grid_player2
    for (row, col) in ai_grid.keys():
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            flat_grid[row * GRID_COLS + col] = 1.0
    # No opponent bubbles in single-player environment
    # opp_grid = {} (empty)

    shooter_y_norm = shooter_y / 800.0
    shooter_angle_norm = (shooter_angle - 90) / 180.0

    color_onehot = np.zeros(BUBBLE_COLORS_COUNT, dtype=np.float32)
    color_onehot[current_color] = 1.0
    next_color_onehot = np.zeros(BUBBLE_COLORS_COUNT, dtype=np.float32)
    next_color_onehot[next_color] = 1.0

    strategic_features = calculate_strategic_features_simple(ai_player_num)

    return np.concatenate([
        flat_grid,
        [shooter_y_norm], [shooter_angle_norm],
        color_onehot, next_color_onehot,
        strategic_features
    ])

def calculate_strategic_features_simple(ai_player_num: int) -> np.ndarray:
    """Match bubbles.py strategic features: 100-length vector with first 3 fields.
    features[0] = AI bubble density (capped)
    features[1] = Opponent bubble density (capped)
    features[2] = Score (normalized by 100)
    """
    # Note: this function needs access to env state; we compute it inside env via a wrapper
    raise RuntimeError("calculate_strategic_features_simple should be called via env._calc_features(ai_player_num)")

def count_potential_matches(grid, row, col, color):
    """Count how many bubbles would match if we place 'color' at (row, col)"""
    # Simulate placing the bubble
    temp_grid = grid.copy()
    temp_grid[(row, col)] = color
    
    # Count connected bubbles of same color
    visited = set()
    to_visit = [(row, col)]
    
    while to_visit:
        r, c = to_visit.pop()
        if (r, c) in visited:
            continue
        if r < 0 or r >= GRID_ROWS or c < 0 or c >= GRID_COLS:
            continue
        if (r, c) not in temp_grid or temp_grid[(r, c)] != color:
            continue
        
        visited.add((r, c))
        
        # Check adjacent positions in honeycomb pattern
        if r % 2 == 0:  # Even row
            adjacent = [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r-1, c-1), (r+1, c-1)]
        else:  # Odd row
            adjacent = [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r-1, c+1), (r+1, c+1)]
        
        for nr, nc in adjacent:
            to_visit.append((nr, nc))
    
    return len(visited)

def is_defensive_position(grid, row, col, player_num):
    """Check if this position is important for defense"""
    # Only player 2 exists in this environment
    if player_num == 2 and col >= GRID_COLS - 3:
        return True
    return False

def is_attack_position(grid, row, col, player_num):
    """Check if this position puts pressure on the opponent"""
    # Only player 2 exists in this environment
    if player_num == 2 and col < GRID_COLS // 2:
        return True
    return False

def calculate_grid_balance(grid, player_num):
    """Calculate how balanced the grid is for the player"""
    if not grid:
        return 0.5
    
    # Calculate center of mass
    total_x = sum(col for (row, col) in grid.keys())
    avg_x = total_x / len(grid)
    
    # Only player 2 exists in this environment
    if player_num == 2:
        # For player 2, want bubbles to be more to the left (away from edge)
        return min((GRID_COLS - avg_x) / GRID_COLS, 1.0)
    else:
        # Fallback for any other player_num
        return 0.5

def calculate_center_line_pressure(grid, player_num):
    """Calculate how much pressure this player is putting on the center line"""
    if not grid:
        return 0.0
    
    # Count bubbles near the center
    center_col = GRID_COLS // 2
    center_bubbles = sum(1 for (row, col) in grid.keys() if abs(col - center_col) <= 2)
    
    return min(center_bubbles / 10.0, 1.0)

def compute_enhanced_reward(prev_score, current_score, prev_grid, current_grid, done, lost, player_num):
    """
    Enhanced reward system for single-player AI training:
    - NO penalties for losing (bubbles reach edge) - focus on bubble popping
    - NO rewards for winning (clearing all bubbles) - focus on bubble popping
    - MUCH smaller penalty for each shot (to encourage exploration and color matching)
    - Increased rewards for popping bubbles
    - Balanced chain reaction bonuses for multiple bubbles popped in one shot
    - Bonus rewards for falling bubbles (strategic gameplay)
    - BIGGER rewards for color matching (even without popping)
    """
    reward = 0.0
    
    # NO penalties for losing - we want the AI to learn to pop bubbles, not avoid losing
    # NO rewards for winning - we want the AI to focus on bubble popping strategy
    
    # MUCH smaller penalty for each shot (encourages exploration and color matching)
    reward -= 0.05  # Reduced from -0.3 to -0.05 to encourage more strategic shots
    
    # Calculate bubbles popped in this action
    bubbles_popped = current_score - prev_score
    if bubbles_popped > 0:
        # Increased base reward for popping bubbles
        base_reward = bubbles_popped * 30.0  # Increased from 20.0 to 30.0 per bubble
        
        # Balanced chain reaction bonus: moderate reward for multiple bubbles
        if bubbles_popped >= 3:
            chain_bonus = bubbles_popped ** 1.5  # Reduced from quadratic to 1.5 power
            reward += base_reward + chain_bonus
        else:
            reward += base_reward
        
        # Additional bonus for large chains (5+ bubbles)
        if bubbles_popped >= 3:
            reward += 40.0  # Increased from 30.0 to 40.0 for better balance
        
        # Bonus for falling bubbles (strategic gameplay)
        # If we popped bubbles and the grid got smaller, some bubbles fell
        if len(current_grid) < len(prev_grid) - bubbles_popped:
            fallen_bubbles = len(prev_grid) - len(current_grid) - bubbles_popped
            if fallen_bubbles > 0:
                # Reward for creating falling bubbles (strategic positioning)
                falling_bonus = fallen_bubbles * 20.0  # Increased from 15.0 to 20.0 per fallen bubble
                reward += falling_bonus
                # Extra bonus for large falling cascades
                if fallen_bubbles >= 3:
                    reward += 35.0  # Increased from 25.0 to 35.0 for impressive falling cascades
    
    # BIGGER reward for color matching (even without popping)
    # This encourages the AI to build color clusters for future pops
    if bubbles_popped == 0 and len(current_grid) > len(prev_grid):
        # AI placed a bubble without popping anything
        # Check if it created color matches (adjacent same-color bubbles)
        # Note: Color matching reward calculation moved to step() method where self is available
        pass  # Will be handled in the step() method
    
    # Increased bonus for surviving longer (encourages defensive play)
    if not done:
        reward += 0.8  # Increased from 0.5 to 0.8 per step
    
    return reward

class TargetBubbleShooterEnv:
    def __init__(self, debug_render: bool = False, debug_fps: int = 1):
        # self.debug_render = debug_render and (pygame is not None)
        self.debug_render = debug_render  # Allow debug render to be enabled
        self._debug_fps = max(1, min(60, int(debug_fps)))
        # Extra slow-down while debug window is on (ms per frame)
        self._debug_extra_delay_ms = 500
        self._debug_initialized = False
        self.reset()
    
    def reset(self):
        # Initialize only the right-side grid (AI player)
        self.grid_player2 = {}  # (row, col): color
        self.center_line_offset = 0
        self.target_center_line_offset = 0
        self.last_shift_offset = 0
        
        # Initial bubbles for right player only (match bubbles.py initialize_bubbles):
        # for all rows, first 7 columns are filled with random colors
        initial_cols = 7
        for row in range(GRID_ROWS):
            for col in range(initial_cols):
                color = random.randint(0, BUBBLE_COLORS_COUNT-1)  # 0-5 to match game encoding
                self.grid_player2[(row, col)] = color
        
        # Only right shooter exists
        self.shooter_y = {2: 400}  # Only player 2
        self.shooter_angle = {2: 180}  # Point left
        self.current_bubble_color = {2: random.randint(0, BUBBLE_COLORS_COUNT-1)}
        self.next_bubble_color = {2: random.randint(0, BUBBLE_COLORS_COUNT-1)}
        self.scores = {2: 0}  # Only player 2 score
        self.done = False
        self.lost = {2: False}  # Only player 2 can lose
        self.step_count = 0
        self.natural_endings = 0
        self.artificial_endings = 0
        self.game_counted = False
        
        return self.get_state(2)  # Only return state for player 2

    # ---------- Debug rendering ----------
    def _debug_init(self):
        if not self.debug_render:
            return
        pygame.init()
        self._DBG_SCREEN_W, self._DBG_SCREEN_H = 1200, 800
        self._dbg_screen = pygame.display.set_mode((self._DBG_SCREEN_W, self._DBG_SCREEN_H))
        pygame.display.set_caption("Training Debug View")
        self._dbg_font = pygame.font.Font(None, 24)
        self._DBG_WHITE = (240, 240, 240)
        self._DBG_GREY = (50, 50, 60)
        self._DBG_LINE = (255, 215, 0)
        self._DBG_TARGET = (255, 80, 80)
        self._DBG_SHOT = (80, 180, 255)
        self._DBG_LOSE_L = (255, 120, 120)
        self._DBG_LOSE_R = (120, 255, 120)
        self._DBG_CONE = (100, 200, 255)
        # Use shared color constants for perfect consistency
        from bubble_geometry import BUBBLE_COLORS
        self._DBG_COLORS = BUBBLE_COLORS
        self._debug_initialized = True

    def _debug_draw(self, player_num: int, target_row: int, target_col: int,
                     shooter_x: int, shooter_y: int, target_x: int, target_y: int,
                     angle: float, path_points=None, bounce_points=None):
        if not self.debug_render:
            return
        self._debug_init()
        # Handle window events so it stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.debug_render = False
                return
        screen = self._dbg_screen
        screen.fill(self._DBG_GREY)

        SCREEN_WIDTH = 1200
        BUBBLE_RADIUS = 20
        middle_x = SCREEN_WIDTH // 2 + int(self.center_line_offset)
        start_y = BUBBLE_RADIUS * 2

        # Draw center line
        pygame.draw.line(screen, self._DBG_LINE, (middle_x, 0), (middle_x, self._DBG_SCREEN_H), 2)

        # Draw losing threshold lines (fixed relative to screen edges)
        LOSE_THRESHOLD = 100
        # Player 1 lose line (left)
        pygame.draw.line(screen, self._DBG_LOSE_L, (LOSE_THRESHOLD, 0), (LOSE_THRESHOLD, self._DBG_SCREEN_H), 1)
        # Player 2 lose line (right)
        pygame.draw.line(screen, self._DBG_LOSE_R, (SCREEN_WIDTH - LOSE_THRESHOLD, 0), (SCREEN_WIDTH - LOSE_THRESHOLD, self._DBG_SCREEN_H), 1)

        # Cone visualization removed per design

        # Draw grids using shared geometry functions
        def draw_grid(grid: dict, side: int):
            for (row, col), color in grid.items():
                x, y = grid_to_screen(row, col, side, self.center_line_offset)
                c = self._DBG_COLORS[color if 0 <= color < len(self._DBG_COLORS) else 0]
                pygame.draw.circle(screen, c, (int(x), int(y)), BUBBLE_RADIUS - 2)

        # Only draw right-side grid (player 2)
        draw_grid(self.grid_player2, 2)
        # Draw reachable empty targets (mask visualization)
        try:
            reachable = self.get_reachable_empty_targets(2)
            # Also compute per-cell same-color counts for current bubble color
            from bubble_geometry import neighbor_same_color_counts as _ncounts
            counts = _ncounts(self.grid_player2, self.current_bubble_color.get(2, 0))
            for (r, c) in reachable:
                x, y = grid_to_screen(r, c, 2, self.center_line_offset)
                idx = r * GRID_COLS + c
                val = int(counts[idx]) if 0 <= idx < counts.shape[0] else 0
                # Hollow circle for reachable empty cell
                pygame.draw.circle(screen, (80, 220, 255), (int(x), int(y)), BUBBLE_RADIUS - 3, 2)
                # Overlay the same-color count
                txt = self._dbg_font.render(str(val), True, (0, 255, 255) if val > 0 else (160, 160, 160))
                screen.blit(txt, (int(x) - 6, int(y) - (BUBBLE_RADIUS + 12)))
            # Highlight chosen target if available
            try:
                act = getattr(self, '_agent_act', None)
                if act is not None and isinstance(act, int) and act >= 0:
                    cr, cc = act // GRID_COLS, act % GRID_COLS
                    cx, cy = grid_to_screen(cr, cc, 2, self.center_line_offset)
                    pygame.draw.circle(screen, (255, 200, 0), (int(cx), int(cy)), BUBBLE_RADIUS - 1, 3)
            except Exception:
                pass
            # Highlight predicted physical landing (if simulated)
            try:
                if getattr(self, '_dbg_landing_rc', None) is not None:
                    lr, lc = self._dbg_landing_rc
                    lx, ly = grid_to_screen(lr, lc, 2, self.center_line_offset)
                    pygame.draw.circle(screen, (255, 80, 80), (int(lx), int(ly)), BUBBLE_RADIUS - 2, 2)
                    # line between intended and landing for error visualization
                    pygame.draw.line(screen, (255, 80, 80), (int(target_x), int(target_y)), (int(lx), int(ly)), 2)
            except Exception:
                pass
        except Exception:
            pass

        # Draw only right shooter
        right_shooter_x = SCREEN_WIDTH - BUBBLE_RADIUS * 2
        pygame.draw.circle(screen, self._DBG_SHOT, (int(right_shooter_x), int(self.shooter_y.get(2, 400))), 8)
        # Highlight acting shooter (always player 2 now)
        pygame.draw.circle(screen, (255, 255, 255), (int(shooter_x), int(shooter_y)), 4)
        
        # Draw the current bubble the shooter has (with its actual color)
        # The color that will be placed on the grid is current_bubble_color (0-5)
        grid_color_index = self.current_bubble_color.get(2, 0)
        if 0 <= grid_color_index < len(self._DBG_COLORS):
            bubble_color = self._DBG_COLORS[grid_color_index]
        else:
            bubble_color = (128, 128, 128)  # Default gray if color index is invalid
        
        # Draw current bubble slightly above and to the right of the shooter
        current_bubble_x = int(shooter_x) + 30
        current_bubble_y = int(shooter_y) - 10
        pygame.draw.circle(screen, bubble_color, (current_bubble_x, current_bubble_y), 15)
        # Add a small border to make it stand out
        pygame.draw.circle(screen, (255, 255, 255), (current_bubble_x, current_bubble_y), 15, 2)

        # Draw target and aiming line
        pygame.draw.circle(screen, self._DBG_TARGET, (int(target_x), int(target_y)), 6)
        pygame.draw.line(screen, self._DBG_TARGET, (int(shooter_x), int(shooter_y)), (int(target_x), int(target_y)), 2)

        # Draw bounce path if provided
        if path_points:
            pts = [(int(px), int(py)) for (px, py) in path_points]
            if len(pts) >= 2:
                # Throttle density to keep it light
                sampled = pts[::3]
                try:
                    pygame.draw.lines(screen, (0, 200, 255), False, sampled, 2)
                except Exception:
                    pass
        # Draw bounce markers
        if bounce_points:
            for (bx, by) in bounce_points:
                pygame.draw.circle(screen, (255, 255, 0), (int(bx), int(by)), 5, 2)

        # HUD text
        # HUD lines: include mask size and row-wise distribution
        try:
            mask_sz = len(self.get_reachable_empty_targets(2))
        except Exception:
            mask_sz = -1
        # Compute per-row counts for reachable targets
        row_counts_str = ""
        try:
            _reachable = self.get_reachable_empty_targets(2)
            row_counts = [0] * GRID_ROWS
            for (rr, cc) in _reachable:
                if 0 <= rr < GRID_ROWS:
                    row_counts[rr] += 1
            # Compact representation to fit HUD: show as comma-separated list
            row_counts_str = ",".join(str(v) for v in row_counts)
        except Exception:
            row_counts_str = "err"
        hud_lines = [
            f"Player: {player_num}  Step: {self.step_count}",
            f"Target: ({target_row}, {target_col})  Angle: {angle:.1f}°",
            f"Score: {self.scores.get(2, 0)}  Offset: {self.target_center_line_offset}",
            f"Reachable empty targets: {mask_sz}",
            f"Row counts: [{row_counts_str}]",
            # Selection diagnostics from agent (random vs greedy, epsilon, action idx)
            (f"Pick: RANDOM  eps={getattr(self,'_agent_eps',-1):.2f}  action={getattr(self,'_agent_act',-1)}"
             if getattr(self,'_agent_rand',False) else
             f"Pick: GREEDY  eps={getattr(self,'_agent_eps',-1):.2f}  action={getattr(self,'_agent_act',-1)}"),
            # Chosen row/col for clarity
            (lambda a: f"Chosen rc: ({a//GRID_COLS},{a%GRID_COLS})" if isinstance(a,int) and a>=0 else "Chosen rc: (-,-)")(getattr(self,'_agent_act',-1)),
        ]
        y_text = 10
        for line in hud_lines:
            txt = self._dbg_font.render(line, True, self._DBG_WHITE)
            screen.blit(txt, (10, y_text))
            y_text += 18

        pygame.display.flip()
        delay_ms = int(1000 / self._debug_fps) + getattr(self, '_debug_extra_delay_ms', 0)
        pygame.time.delay(delay_ms)
    
    def get_state(self, player_num):
        # Encode state for AI player (player 2) only
        return self.encode_state_game_like(2)

    def encode_state_game_like(self, ai_player_num: int) -> np.ndarray:
        # FIXED: Only current bubble color to prevent AI from confusing current vs next
        current_color = self.current_bubble_color[ai_player_num]
        # REMOVED: next_color = self.next_bubble_color[ai_player_num]  # AI should NOT see next bubble

        # Use consistent integer encoding with only current bubble color
        from bubble_geometry import encode_compact_state_consistent, neighbor_same_color_counts
        # FIXED: Only pass current color (compact_state is 701 dims: 700 grid + 1 current)
        compact_state = encode_compact_state_consistent(self.grid_player2, current_color)
        neighbor_counts = neighbor_same_color_counts(self.grid_player2, current_color)
        
        # Strategic features: simplified for single player (3 features only)
        features = np.zeros(3, dtype=np.float32)
        features[0] = min(len(self.grid_player2) / 50.0, 1.0)  # AI bubble density
        features[1] = 0.0  # No opponent bubbles
        features[2] = self.scores[2] / 100.0  # AI score

        # Combine compact state with additional features
        # Result: 701 (compact) + 700 (neighbor) + 3 (features) = 1404 dims
        return np.concatenate([
            compact_state,           # 700 colors + current_color (701 total)
            neighbor_counts,         # 700 neighbor same-color counts
            features                 # 100 strategic features
        ])
    
    def get_bubble_targets(self, player_num):
        """Return all occupied (row,col) cells in the player's grid."""
        return [(r, c) for (r, c) in self.grid_player2.keys()]

    def get_reachable_empty_targets(self, player_num: int):
        """Return empty (row,col) cells that are valid placements and reachable by angular LOS (swept corridor)."""
        SCREEN_WIDTH = 1200
        BUBBLE_RADIUS = 20
        shooter_x = SCREEN_WIDTH - BUBBLE_RADIUS * 2
        shooter_y = self.shooter_y[player_num]
        # Build candidate empty cells (valid placement) constrained to adjacency to the front-most bubble per row
        candidates = []
        # Front-most (max col) per row map
        front_by_row = {}
        for (row, col), color in self.grid_player2.items():
            prev = front_by_row.get(row)
            if prev is None or col > prev:
                front_by_row[row] = col
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                if (row, col) in self.grid_player2:
                    continue
                if not self.is_valid_placement(row, col, self.grid_player2):
                    continue
                # Require adjacency ONLY to the front-most bubble in this row
                fcol = front_by_row.get(row)
                if fcol is None:
                    continue
                if row % 2 == 0:
                    adj_front = [(row-1, fcol), (row+1, fcol), (row, fcol-1), (row, fcol+1), (row-1, fcol-1), (row+1, fcol-1)]
                else:
                    adj_front = [(row-1, fcol), (row+1, fcol), (row, fcol-1), (row, fcol+1), (row-1, fcol+1), (row+1, fcol+1)]
                if (row, col) not in adj_front:
                    continue
                candidates.append((row, col))
        # Use shared angular LOS filter with narrower corridor to avoid over-blocking
        grid_map = self.grid_player2
        prelim = get_angular_los_filtered_targets(
            shooter_x, shooter_y, candidates, player_num, grid_map, self.center_line_offset, corridor_width=0.5 * BUBBLE_RADIUS
        )
        # Post-simulation verification: keep only targets where a simulated shot lands on the intended cell
        verified = []
        for (row, col) in prelim:
            tx, ty = self._grid_to_screen(row, col, player_num)
            ang = math.degrees(math.atan2(ty - shooter_y, tx - shooter_x))
            rx = shooter_x + math.cos(math.radians(ang)) * 2000.0
            ry = shooter_y + math.sin(math.radians(ang)) * 2000.0
            landing, _, _ = self._simulate_shot_with_bounce(player_num, shooter_x, shooter_y, rx, ry, preferred_target=(row, col))
            if landing is not None and landing[0] == row and landing[1] == col:
                verified.append((row, col))
        return verified

    def _angle_for_target(self, player_num: int, row: int, col: int):
        # Use shared geometry functions for consistency
        tx, ty = grid_to_screen(row, col, player_num, self.center_line_offset)
        shooter_x = SCREEN_WIDTH - BUBBLE_RADIUS * 2  # Right side shooter
        shooter_y = self.shooter_y[player_num]
        ang = calculate_angle_to_target(shooter_x, shooter_y, tx, ty)
        return ang

    def get_reachable_bubble_targets(self, player_num: int):
        """Return occupied (row,col) bubbles reachable by direct LOS only (no bounces)."""
        SCREEN_WIDTH = 1200
        BUBBLE_RADIUS = 20
        shooter_x = SCREEN_WIDTH - BUBBLE_RADIUS * 2
        shooter_y = self.shooter_y[player_num]
        filtered = []
        for (r, c) in list(self.grid_player2.keys()):
            tx, ty = self._grid_to_screen(r, c, player_num)
            if self._is_direct_path_clear(player_num, shooter_x, shooter_y, tx, ty):
                filtered.append((r, c))
        return filtered

    def _target_has_los_to_wall(self, player_num: int, target_row: int, target_col: int, wall: str) -> bool:
        tx, ty = self._grid_to_screen(target_row, target_col, player_num)
        wy = 0 if wall == "top" else 800  # SCREEN_HEIGHT
        ax, ay = tx, ty
        bx, by = tx, wy
        abx, aby = bx - ax, by - ay
        ab_len2 = max(abx * abx + aby * aby, 1e-6)
        corridor = 20 * 0.85  # BUBBLE_RADIUS
        for (row, col), _ in self.grid_player2.items():
            if row == target_row and col == target_col:
                continue
            cx, cy = self._grid_to_screen(row, col, player_num)
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
        tx, ty = self._grid_to_screen(target_row, target_col, player_num)
        my = -ty if wall == "top" else (2 * 800 - ty)  # SCREEN_HEIGHT
        base_angle = math.degrees(math.atan2(my - shooter_y, tx - shooter_x))
        a = (base_angle % 360.0 + 360.0) % 360.0
        if not (60.0 <= a <= 300.0):
            return None
        return base_angle

    def _has_bounce_path_to_target(self, player_num: int, shooter_x: float, shooter_y: float, target_row: int, target_col: int) -> bool:
        """Return True if a top/bottom-bounce shot can land exactly on (target_row, target_col).
        Uses mirror bases + expanded angle offsets with precise simulation.
        """
        tx, ty = self._grid_to_screen(target_row, target_col, player_num)
        # Candidate bases: direct + mirrored top/bottom
        bases = []
        # direct
        bases.append(math.degrees(math.atan2(ty - shooter_y, tx - shooter_x)))
        # mirror top
        bases.append(math.degrees(math.atan2(-ty - shooter_y, tx - shooter_x)))
        # mirror bottom
        bases.append(math.degrees(math.atan2((2 * 800 - ty) - shooter_y, tx - shooter_x)))  # SCREEN_HEIGHT = 800

        offsets = [-12, -9, -6, -3, 0, 3, 6, 9, 12]

        def _angle_ok_for_player(a: float) -> bool:
            a = (a % 360.0 + 360.0) % 360.0
            # right player (2) should shoot roughly 90..270
            return 60.0 <= a <= 300.0

        for base in bases:
            for off in offsets:
                ang = base + off
                if not _angle_ok_for_player(ang):
                    continue
                ray_len = 2000.0
                aim_x = shooter_x + math.cos(math.radians(ang)) * ray_len
                aim_y = shooter_y + math.sin(math.radians(ang)) * ray_len
                landing, _, _ = self._simulate_shot_with_bounce(player_num, shooter_x, shooter_y, aim_x, aim_y)
                if landing is not None and landing[0] == target_row and landing[1] == target_col:
                    return True
        return False

    def find_best_angle_to_target(self, target_row: int, target_col: int, player_num: int,
                                  shooter_x: float, shooter_y: float) -> float:
        """Best-effort angle to land on (target_row, target_col) using direct/bounce candidates.
        Returns the exact matching angle if found, otherwise the closest landing.
        """
        tx, ty = self._grid_to_screen(target_row, target_col, player_num)
        SCREEN_HEIGHT = 800
        bases = [
            math.degrees(math.atan2(ty - shooter_y, tx - shooter_x)),
            math.degrees(math.atan2(-ty - shooter_y, tx - shooter_x)),
            math.degrees(math.atan2((2 * SCREEN_HEIGHT - ty) - shooter_y, tx - shooter_x)),
        ]
        offsets = [-10, -6, -3, 0, 3, 6, 10]
        def angle_ok(a: float) -> bool:
            a = (a % 360.0 + 360.0) % 360.0
            return 60.0 <= a <= 300.0
        best_angle = None
        best_err = float('inf')
        for base in bases:
            for off in offsets:
                ang = base + off
                if not angle_ok(ang):
                    continue
                rx = shooter_x + math.cos(math.radians(ang)) * 2000.0
                ry = shooter_y + math.sin(math.radians(ang)) * 2000.0
                landing, _, _ = self._simulate_shot_with_bounce(player_num, shooter_x, shooter_y, rx, ry)
                if landing is not None:
                    r, c = landing
                    if r == target_row and c == target_col:
                        return ang
                    lx, ly = self._grid_to_screen(r, c, player_num)
                    err = (tx - lx) * (tx - lx) + (ty - ly) * (ty - ly)
                    if err < best_err:
                        best_err = err
                        best_angle = ang
        if best_angle is not None:
            return best_angle
        return calculate_angle_to_target(shooter_x, shooter_y, tx, ty)

    # REMOVED: Unused color-based functions from old approach
    # - get_available_colors_for_ai(): AI no longer analyzes available colors
    # - get_color_based_targets(): AI no longer needs color-based targeting  
    # - select_color_action(): AI no longer chooses colors
    # - find_best_position_for_color(): AI no longer needs color-based positioning
    # 
    # NEW APPROACH: AI gets random colored bubbles and chooses targets directly
    # based on color analysis learned through experience and rewards.

    # ---------- Geometry helpers matching bubbles.py mapping ----------
    def _grid_to_screen(self, row: int, col: int, player_num: int):
        SCREEN_WIDTH = 1200
        BUBBLE_RADIUS = 20
        middle_x = SCREEN_WIDTH // 2 + int(self.center_line_offset)
        start_y = BUBBLE_RADIUS * 2
        row_offset = BUBBLE_RADIUS if row % 2 == 1 else 0
        # Only player 2 exists in this environment
        x = middle_x + (col * BUBBLE_RADIUS * 2) + row_offset + BUBBLE_RADIUS
        y = start_y + (row * BUBBLE_RADIUS * 1.8)
        return x, y

    def _has_clear_path(self, player_num: int, shooter_x: float, shooter_y: float, target_row: int, target_col: int) -> bool:
        """Block shots that pass through existing bubbles. Simple LOS check against player's grid bubbles."""
        BUBBLE_RADIUS = 20
        tx, ty = self._grid_to_screen(target_row, target_col, player_num)
        # Segment AB from shooter (sx,sy) to target (tx,ty)
        sx, sy = shooter_x, shooter_y
        ax, ay = sx, sy
        bx, by = tx, ty
        abx, aby = bx - ax, by - ay
        ab_len2 = max(abx*abx + aby*aby, 1e-6)
        grid = self.grid_player2  # Only player 2 exists in this environment
        # Use same corridor width as game for consistency
        corridor = BUBBLE_RADIUS * 0.85
        for (row, col), color in grid.items():
            cx, cy = self._grid_to_screen(row, col, player_num)
            # Project C onto AB
            t = ((cx - ax) * abx + (cy - ay) * aby) / ab_len2
            t = max(0.0, min(1.0, t))
            px, py = ax + t * abx, ay + t * aby
            dx, dy = cx - px, cy - py
            dist = (dx*dx + dy*dy) ** 0.5
            # If the path goes too close to an existing bubble, block
            if dist < corridor:
                # allow passing very near to the intended landing cell itself
                if not (row == target_row and col == target_col):
                    return False
        return True

    # ---------- Fast Path Approximation (Performance Optimization) ----------
    def _fast_shot_landing(self, player_num: int, shooter_x: float, shooter_y: float, target_x: float, target_y: float):
        """Fast approximation of shot landing without expensive physics simulation.
        This is 10-50x faster than _simulate_shot_with_bounce for training purposes.
        """
        SCREEN_WIDTH = 1200
        SCREEN_HEIGHT = 800
        BUBBLE_RADIUS = 20
        middle_x = SCREEN_WIDTH // 2 + int(self.center_line_offset)
        
        # Simple line-of-sight check
        dx = target_x - shooter_x
        dy = target_y - shooter_y
        distance = (dx*dx + dy*dy) ** 0.5
        
        # Check if direct path is blocked by any bubbles
        if self._is_direct_path_clear(player_num, shooter_x, shooter_y, target_x, target_y):
            # Direct path is clear - target is reachable
            return self._screen_to_grid(target_x, target_y, player_num), [], []
        
        # Path is blocked - try simple bounce approximations using mirror trick (top/bottom)
        # Mirror across top wall (y = 0)
        mirror_top_y = -target_y
        # Mirror across bottom wall (y = SCREEN_HEIGHT)
        mirror_bottom_y = 2 * SCREEN_HEIGHT - target_y

        candidate_points = [
            (target_x, mirror_top_y),
            (target_x, mirror_bottom_y),
        ]

        angle_offsets = [-8, -4, 0, 4, 8]
        for mx, my in candidate_points:
            # Base direction to mirrored point
            base_angle = math.degrees(math.atan2(my - shooter_y, target_x - shooter_x))
            for off in angle_offsets:
                angle = base_angle + off
                # Convert angle back to a temporary target point along the ray
                # and test LOS to that point as an approximation
                ray_len = 1000.0
                rx = shooter_x + math.cos(math.radians(angle)) * ray_len
                ry = shooter_y + math.sin(math.radians(angle)) * ray_len
                if self._is_direct_path_clear(player_num, shooter_x, shooter_y, rx, ry):
                    # If LOS is clear towards the mirrored direction, accept and map
                    return self._screen_to_grid(target_x, target_y, player_num), [], []
        
        # If all else fails, return None (target not reachable)
        return None, [], []
    
    def _is_direct_path_clear(self, player_num: int, sx: float, sy: float, tx: float, ty: float) -> bool:
        """Quick check if direct path from shooter to target is blocked.
        Allows passing near the TARGET bubble itself (skip it as an obstacle)."""
        BUBBLE_RADIUS = 20
        corridor = BUBBLE_RADIUS * 0.85
        
        # Vector from shooter to target
        dx = tx - sx
        dy = ty - sy
        path_length = (dx*dx + dy*dy) ** 0.5
        if path_length < 1e-6:
            return True
        
        # Normalize direction
        dx /= path_length
        dy /= path_length
        
        # Determine target grid cell to skip as obstacle
        target_rc = self._screen_to_grid(tx, ty, player_num)
        
        # Check each bubble for collision
        for (row, col) in self.grid_player2.keys():
            # Skip the target bubble itself
            if (row, col) == target_rc:
                continue
            bx, by = self._grid_to_screen(row, col, player_num)
            
            # Vector from shooter to bubble
            bx_rel = bx - sx
            by_rel = by - sy
            
            # Project bubble onto path
            projection = bx_rel * dx + by_rel * dy
            if projection < 0 or projection > path_length:
                continue
            
            # Perpendicular distance from path
            proj_x = sx + projection * dx
            proj_y = sy + projection * dy
            perp_dist = ((bx - proj_x) ** 2 + (by - proj_y) ** 2) ** 0.5
            if perp_dist < corridor:
                return False
        return True

    def _refine_angle_to_hit_target(self, base_angle: float, player_num: int, shooter_x: float, shooter_y: float,
                                    target_row: int, target_col: int) -> float:
        """Nudge the angle by small increments to avoid passing too close to other bubbles causing unintended snaps.
        Tries a symmetric set of small offsets around the base angle and picks the first that lands on the exact target.
        """
        if base_angle is None:
            return base_angle
        # Target center
        tx, ty = self._grid_to_screen(target_row, target_col, player_num)
        # Small, conservative nudges (degrees)
        offsets = [0, -2, 2, -4, 4, -6, 6]
        for off in offsets:
            ang = base_angle + off
            rx = shooter_x + math.cos(math.radians(ang)) * 2000.0
            ry = shooter_y + math.sin(math.radians(ang)) * 2000.0
            landing, _, _ = self._simulate_shot_with_bounce(
                player_num, shooter_x, shooter_y, rx, ry, preferred_target=(target_row, target_col)
            )
            if landing is not None and landing == (target_row, target_col):
                return ang
        return base_angle
    
    def _screen_to_grid(self, x: float, y: float, player_num: int) -> tuple:
        """Convert screen coordinates back to grid coordinates."""
        SCREEN_WIDTH = 1200
        BUBBLE_RADIUS = 20
        middle_x = SCREEN_WIDTH // 2 + int(self.center_line_offset)
        start_y = BUBBLE_RADIUS * 2
        
        # Reverse the grid_to_screen calculation
        y_rel = y - start_y
        row = int(y_rel / (BUBBLE_RADIUS * 1.8))
        
        x_rel = x - middle_x
        if player_num == 2:
            x_rel -= BUBBLE_RADIUS
            if row % 2 == 1:
                x_rel -= BUBBLE_RADIUS
        
        col = int(x_rel / (2 * BUBBLE_RADIUS))
        
        # Clamp to valid range
        row = max(0, min(GRID_ROWS - 1, row))
        col = max(0, min(GRID_COLS - 1, col))
        
        return (row, col)

    # PERFORMANCE NOTE: This expensive simulation is now commented out for training speed
    # To restore full physics simulation, uncomment the function below
    """
    def _simulate_shot_with_bounce(self, player_num: int, shooter_x: float, shooter_y: float, target_x: float, target_y: float):
        # ORIGINAL EXPENSIVE PHYSICS SIMULATION - COMMENTED OUT FOR PERFORMANCE
        # This function was doing expensive calculations for every target
        # Uncomment below to restore full physics simulation
        pass
    """
    
    # ORIGINAL EXPENSIVE PHYSICS SIMULATION - COMMENTED OUT FOR PERFORMANCE
    # This function was doing expensive calculations for every target (up to 700 times per step!)
    # Uncomment the entire function below to restore full physics simulation
    """
    def _simulate_shot_with_bounce(self, player_num: int, shooter_x: float, shooter_y: float, target_x: float, target_y: float):
        # Simulate a shot with wall bounces (top/bottom) and return a landing (row, col).
        # Snap when colliding with an existing bubble or when reaching the center line band.
        # Returns (row, col), path_points, bounce_points or (None, path_points, bounce_points) if cannot determine.
        SCREEN_WIDTH = 1200
        SCREEN_HEIGHT = 800
        BUBBLE_RADIUS = 20
        # Center line x
        middle_x = SCREEN_WIDTH // 2 + int(self.center_line_offset)
        # Top band start
        start_y = BUBBLE_RADIUS * 2
        # Direction vector
        dx = target_x - shooter_x
        dy = target_y - shooter_y
        norm = max((dx*dx + dy*dy) ** 0.5, 1e-6)
        vx = dx / norm
        vy = dy / norm
        
        # SAFETY CHECK: For right shooter (player 2), ensure shot goes leftward
        if player_num == 2 and vx > 0:
            # Shot is going right (backward) - flip it leftward
            vx = -abs(vx)  # Force leftward direction
            # Recalculate vy to maintain unit vector
            vy = vy / abs(vx) * (1.0 - vx*vx)**0.5 if abs(vx) < 1.0 else 0.0
        
        # Position
        x = shooter_x
        y = shooter_y
        px, py = x, y  # previous position for swept collision
        # Step parameters - smaller step size for better collision detection
        step_len = 3.0  # Reduced from 6.0 for more accurate collision detection
        max_steps = 5000  # Increased to compensate for smaller step size
        grid = self.grid_player2  # Only player 2 exists in this environment
        # Precompute bubble centers list
        bubbles = [(rc[0], rc[1], *self._grid_to_screen(rc[0], rc[1], player_num)) for rc in grid.keys()]
        path_points = []
        bounce_points = []
        for _ in range(max_steps):
            # advance
            x += vx * step_len
            y += vy * step_len
            if len(path_points) == 0 or (abs(x - path_points[-1][0]) + abs(y - path_points[-1][1]) > 3.0):
                path_points.append((x, y))
            # bounce on walls (top/bottom)
            if y <= BUBBLE_RADIUS:
                y = BUBBLE_RADIUS + (BUBBLE_RADIUS - y)
                vy = -vy
                bounce_points.append((x, y))
            elif y >= SCREEN_HEIGHT - BUBBLE_RADIUS:
                y = (SCREEN_HEIGHT - BUBBLE_RADIUS) - (y - (SCREEN_HEIGHT - BUBBLE_RADIUS))
                vy = -vy
                bounce_points.append((x, y))
            # reached center line band -> snap to nearest empty valid grid cell near current position
            if player_num == 2 and x <= middle_x + BUBBLE_RADIUS:
                # find nearest empty valid cell to current (x,y)
                best = None
                best_dist = 1e9
                for row in range(GRID_ROWS):
                    for col in range(GRID_COLS):
                        if (row, col) in grid:
                            continue
                        if not self.is_valid_placement(row, col, grid):
                            continue
                        gx, gy = self._grid_to_screen(row, col, player_num)
                        d2 = (gx - x) * (gx - x) + (gy - y) * (gy - y)
                        if d2 < best_dist:
                            best_dist = d2
                            best = (row, col)
                if best is not None:
                    return best, path_points, bounce_points
            # collision with existing bubble -> snap to nearest valid neighbor (improved collision detection)
            seg_dx = x - px
            seg_dy = y - py
            seg_len2 = max(seg_dx*seg_dx + seg_dy*dy, 1e-9)
            collided = False
            cx_hit = cy_hit = None
            for (r, c, bx, by) in bubbles:
                # closest point on segment P(px,py)->Q(x,y) to circle center C(bx,by)
                t = ((bx - px) * seg_dx + (by - py) * seg_dy) / seg_len2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = px + t * seg_dx
                qy = py + t * seg_dy
                ddx = bx - qx
                ddy = by - qy
                # More strict collision detection - no more tunneling through bubbles
                if ddx*ddx + ddy*ddy <= (2*BUBBLE_RADIUS) ** 2:  # Removed the -2 tolerance
                    collided = True
                    cx_hit, cy_hit = qx, qy
                    # pick nearest empty neighbor
                    nbrs = self._neighbors(r, c)
                    best = None
                    best_dist = 1e9
                    for (nr, nc) in nbrs:
                        if nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS:
                            continue
                        if (nr, nc) in grid:
                            continue
                        if not self.is_valid_placement(nr, nc, grid):
                            continue
                        nx, ny = self._grid_to_screen(nr, nc, player_num)
                        d = (nx - cx_hit) * (nx - cx_hit) + (ny - cy_hit) * (ny - cy_hit)
                        if d < best_dist:
                            best_dist = d
                            best = (nr, nc)
                    if best is not None:
                        return best, path_points, bounce_points
            # update previous point after tests
            px, py = x, y
        return None, path_points, bounce_points
    """

    def _neighbors(self, row: int, col: int):
        if row % 2 == 0:
            return [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col-1), (row+1, col-1)]
        else:
            return [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col+1), (row+1, col+1)]

    def _approx_col_from_x(self, row: int, x: float, player_num: int) -> int:
        SCREEN_WIDTH = 1200
        BUBBLE_RADIUS = 20
        middle_x = SCREEN_WIDTH // 2 + int(self.center_line_offset)
        row_offset = BUBBLE_RADIUS if row % 2 == 1 else 0
        # Only player 2 exists in this environment
        # x = middle_x + (col*2R) + row_offset + R
        val = (x - middle_x - row_offset - BUBBLE_RADIUS) / (2 * BUBBLE_RADIUS)
        col = int(round(val))
        return max(0, min(GRID_COLS - 1, col))

    def _simulate_shot_with_bounce(self, player_num: int, shooter_x: float, shooter_y: float, target_x: float, target_y: float, preferred_target: tuple = None):
        """Simulate a shot with wall bounces (top/bottom) and return a landing (row, col).
        Snap when colliding with an existing bubble or when reaching the center line band.
        Returns (row, col), path_points, bounce_points or (None, path_points, bounce_points) if cannot determine.
        """
        SCREEN_WIDTH = 1200
        SCREEN_HEIGHT = 800
        BUBBLE_RADIUS = 20
        # Center line x
        middle_x = SCREEN_WIDTH // 2 + int(self.center_line_offset)
        # Top band start
        start_y = BUBBLE_RADIUS * 2
        # Direction vector
        dx = target_x - shooter_x
        dy = target_y - shooter_y
        norm = max((dx*dx + dy*dy) ** 0.5, 1e-6)
        vx = dx / norm
        vy = dy / norm
        
        # SAFETY CHECK: For right shooter (player 2), ensure shot goes leftward
        if player_num == 2 and vx > 0:
            # Shot is going right (backward) - flip it leftward
            vx = -abs(vx)  # Force leftward direction
            # Recalculate vy to maintain unit vector
            vy = vy / abs(vx) * (1.0 - vx*vx)**0.5 if abs(vx) < 1.0 else 0.0
        
        # Position
        x = shooter_x
        y = shooter_y
        px, py = x, y  # previous position for swept collision
        # Step parameters - smaller step size for better collision detection
        step_len = 3.0  # Reduced from 6.0 for more accurate collision detection
        max_steps = 5000  # Increased to compensate for smaller step size
        grid = self.grid_player2  # Only player 2 exists in this environment
        # Precompute bubble centers list
        bubbles = [(rc[0], rc[1], *self._grid_to_screen(rc[0], rc[1], player_num)) for rc in grid.keys()]
        path_points = []
        bounce_points = []
        for _ in range(max_steps):
            # advance
            x += vx * step_len
            y += vy * step_len
            if len(path_points) == 0 or (abs(x - path_points[-1][0]) + abs(y - path_points[-1][1]) > 3.0):
                path_points.append((x, y))
            # bounce on walls (top/bottom)
            if y <= BUBBLE_RADIUS:
                y = BUBBLE_RADIUS + (BUBBLE_RADIUS - y)
                vy = -vy
                bounce_points.append((x, y))
            elif y >= SCREEN_HEIGHT - BUBBLE_RADIUS:
                y = (SCREEN_HEIGHT - BUBBLE_RADIUS) - (y - (SCREEN_HEIGHT - BUBBLE_RADIUS))
                vy = -vy
                bounce_points.append((x, y))
            # reached center line band -> snap to empty valid grid cell
            if player_num == 2 and x <= middle_x + BUBBLE_RADIUS:
                # If we have an intended empty target and it's valid, snap directly to it
                if preferred_target is not None:
                    tr, tc = preferred_target
                    if (tr, tc) not in grid and self.is_valid_placement(tr, tc, grid):
                        return (tr, tc), path_points, bounce_points
                # Otherwise, find best empty valid cell using a bias toward the preferred target if provided
                best = None
                best_score = 1e18
                for row in range(GRID_ROWS):
                    for col in range(GRID_COLS):
                        if (row, col) in grid:
                            continue
                        if not self.is_valid_placement(row, col, grid):
                            continue
                        gx, gy = self._grid_to_screen(row, col, player_num)
                        # positional proximity to current crossing point
                        d_pos2 = (gx - x) * (gx - x) + (gy - y) * (gy - y)
                        # bias toward intended target cell if available
                        if preferred_target is not None:
                            tr, tc = preferred_target
                            tx, ty = self._grid_to_screen(tr, tc, player_num)
                            d_tar2 = (gx - tx) * (gx - tx) + (gy - ty) * (gy - ty)
                        else:
                            d_tar2 = 0.0
                        # score: strongly prefer cells close to intended target, weakly prefer close to crossing point
                        score = d_tar2 + 0.1 * d_pos2
                        if score < best_score:
                            best_score = score
                            best = (row, col)
                if best is not None:
                    return best, path_points, bounce_points
            # collision with existing bubble -> snap to nearest valid neighbor (improved collision detection)
            seg_dx = x - px
            seg_dy = y - py
            seg_len2 = max(seg_dx*seg_dx + seg_dy*dy, 1e-9)
            collided = False
            cx_hit = cy_hit = None
            for (r, c, bx, by) in bubbles:
                # closest point on segment P(px,py)->Q(x,y) to circle center C(bx,by)
                t = ((bx - px) * seg_dx + (by - py) * seg_dy) / seg_len2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = px + t * seg_dx
                qy = py + t * seg_dy
                ddx = bx - qx
                ddy = by - qy
                # More strict collision detection - no more tunneling through bubbles
                if ddx*ddx + ddy*ddy <= (2*BUBBLE_RADIUS) ** 2:  # Removed the -2 tolerance
                    collided = True
                    cx_hit, cy_hit = qx, qy
                    # If intended empty target is a neighbor and valid, use it
                    nbrs = self._neighbors(r, c)
                    if preferred_target is not None:
                        tr, tc = preferred_target
                        if (tr, tc) in nbrs and (tr, tc) not in grid and self.is_valid_placement(tr, tc, grid):
                            return (tr, tc), path_points, bounce_points
                    # pick best empty neighbor (bias toward preferred target if provided)
                    best = None
                    best_score = 1e18
                    for (nr, nc) in nbrs:
                        if nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS:
                            continue
                        if (nr, nc) in grid:
                            continue
                        if not self.is_valid_placement(nr, nc, grid):
                            continue
                        nx, ny = self._grid_to_screen(nr, nc, player_num)
                        d_pos2 = (nx - cx_hit) * (nx - cx_hit) + (ny - cy_hit) * (ny - cy_hit)
                        if preferred_target is not None:
                            tr, tc = preferred_target
                            tx, ty = self._grid_to_screen(tr, tc, player_num)
                            d_tar2 = (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty)
                        else:
                            d_tar2 = 0.0
                        score = d_tar2 + 0.1 * d_pos2
                        if score < best_score:
                            best_score = score
                            best = (nr, nc)
                    if best is not None:
                        return best, path_points, bounce_points
            # update previous point after tests
            px, py = x, y
        return None, path_points, bounce_points

    # ---------- Angle cone limiting based on losing thresholds ----------
    @staticmethod
    def _norm_angle_deg(angle: float) -> float:
        a = angle % 360.0
        if a < 0:
            a += 360.0
        return a

    @staticmethod
    def _cone_bounds(a1: float, a2: float):
        a1 = TargetBubbleShooterEnv._norm_angle_deg(a1)
        a2 = TargetBubbleShooterEnv._norm_angle_deg(a2)
        w = (a2 - a1) % 360.0
        if w > 180.0:
            # pick the shorter arc by swapping
            a1, a2 = a2, a1
            w = (a2 - a1) % 360.0
        return a1, a2, w

    @staticmethod
    def _in_cone(test: float, a1: float, a2: float) -> bool:
        a1, a2, w = TargetBubbleShooterEnv._cone_bounds(a1, a2)
        t = TargetBubbleShooterEnv._norm_angle_deg(test)
        d = (t - a1) % 360.0
        return d <= w

    def _get_shoot_cone(self, player_num: int, shooter_x: float, shooter_y: float):
        SCREEN_WIDTH = 1200
        SCREEN_HEIGHT = 800
        LOSE_THRESHOLD = 100
        BUBBLE_RADIUS = 20
        # Only player 2 exists in this environment
        lx = SCREEN_WIDTH - LOSE_THRESHOLD - BUBBLE_RADIUS
        top_angle = math.degrees(math.atan2(0 - shooter_y, lx - shooter_x))
        bot_angle = math.degrees(math.atan2(SCREEN_HEIGHT - shooter_y, lx - shooter_x))
        return self._cone_bounds(top_angle, bot_angle)
    
    def is_valid_placement(self, row, col, grid):
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
            if (nr, nc) in grid:
                return True
        
        return False
    
    def step(self, player_num, target_action_idx, pre_calculated_targets=None):
        """Execute one step for the given player using target-based actions."""
        # Check for win condition first (grid already empty)
        if len(self.grid_player2) == 0:
            self.done = True
            self.natural_endings += 1
            print(f"WIN: Grid cleared! Episode ending naturally.")
            return self.get_state(player_num), 0.0, True, False  # No win reward
        
        # NEW: Empty-cell target action system
        # target_action_idx: 0-699 representing grid positions; we mask to reachable empty cells
        target_row = target_action_idx // GRID_COLS
        target_col = target_action_idx % GRID_COLS
        
        # Validate that target is an empty cell, valid placement, and in direct LOS
        reachable_empty = self.get_reachable_empty_targets(player_num)
        if (target_row, target_col) in self.grid_player2 or (target_row, target_col) not in reachable_empty:
            if reachable_empty:
                # Choose the reachable cell closest to the originally intended (row,col),
                # instead of defaulting to the first (which biases top row)
                intended_rc = (target_row, target_col)
                best_rc = None
                best_d2 = 1e18
                # Use screen-space distance for consistency with aiming
                tx, ty = self._grid_to_screen(intended_rc[0], intended_rc[1], player_num)
                for rr, cc in reachable_empty:
                    rx, ry = self._grid_to_screen(rr, cc, player_num)
                    d2 = (rx - tx) * (rx - tx) + (ry - ty) * (ry - ty)
                    if d2 < best_d2:
                        best_d2 = d2
                        best_rc = (rr, cc)
                if best_rc is not None:
                    target_row, target_col = best_rc
                else:
                    # Fallback to first as last resort
                    target_row, target_col = reachable_empty[0]
            else:
                # No direct-LOS targets: no-op step
                return self.get_state(player_num), 0.0, False, False
        
        # Get shooter position (match game exactly: right shooter at SCREEN_WIDTH - 2*BUBBLE_RADIUS)
        shooter_x = SCREEN_WIDTH - BUBBLE_RADIUS * 2
        shooter_y = self.shooter_y[player_num]
        
        # Calculate target bubble center position
        target_x, target_y = self._grid_to_screen(target_row, target_col, player_num)
        
        # Calculate angle to hit target: prefer direct LOS, fallback to deterministic bounce (try both walls)
        # Direct-LOS-only: compute direct angle
        angle = calculate_angle_to_target(shooter_x, shooter_y, target_x, target_y)
        # Refine toward intended cell using preferred_target-aware simulation
        angle = self._refine_angle_to_hit_target(angle, player_num, shooter_x, shooter_y, target_row, target_col)

        # Pre-shot simulation & bump disabled when bounces are off
        if AI_ENABLE_BOUNCE:
            try:
                rx = shooter_x + math.cos(math.radians(angle)) * 2000.0
                ry = shooter_y + math.sin(math.radians(angle)) * 2000.0
                landing, _, _ = self._simulate_shot_with_bounce(player_num, shooter_x, shooter_y, rx, ry, preferred_target=(target_row, target_col))
                
                # Check if simulation failed to hit target
                if landing is None or landing[0] != target_row or landing[1] != target_col:
                    # Simulation failed - apply bump based on verticality
                    rad = math.radians(angle)
                    bump_deg = 6.0 * abs(math.sin(rad))
                    bump_sign = 0.0
                    if 90.0 <= angle <= 270.0:
                        if target_y < shooter_y:
                            bump_sign = +1.0
                        else:
                            bump_sign = -1.0
                    
                    # Apply bump and retry simulation (up to 2 tries)
                    tries = 2
                    while tries > 0 and bump_sign != 0.0:
                        angle += bump_sign * bump_deg
                        if angle < 90.0:
                            angle = 90.0
                        elif angle > 270.0:
                            angle = 270.0
                        
                        rx = shooter_x + math.cos(math.radians(angle)) * 2000.0
                        ry = shooter_y + math.sin(math.radians(angle)) * 2000.0
                        landing, _, _ = self._simulate_shot_with_bounce(player_num, shooter_x, shooter_y, rx, ry)
                        
                        # Check if we hit the target now
                        if landing is not None and landing[0] == target_row and landing[1] == target_col:
                            break
                        
                        tries -= 1
            except Exception:
                pass
        
        # Determine landing cell: directly the chosen empty cell
        intended_target = (target_row, target_col)

        # DEBUG: Predict landing for HUD. With bounces disabled, reflect intended placement
        if not AI_ENABLE_BOUNCE:
            self._dbg_landing_rc = intended_target
        else:
            try:
                rx = shooter_x + math.cos(math.radians(angle)) * 2000.0
                ry = shooter_y + math.sin(math.radians(angle)) * 2000.0
                dbg_landing, _, _ = self._simulate_shot_with_bounce(player_num, shooter_x, shooter_y, rx, ry)
                self._dbg_landing_rc = dbg_landing
            except Exception:
                self._dbg_landing_rc = None
        
        # Debug render if enabled (no path/bounce points now)
        if self.debug_render:
            self._debug_draw(player_num, target_row, target_col, shooter_x, shooter_y, target_x, target_y, angle, None, None)
        
        # Place bubble at landing position
        prev_score = self.scores[player_num]
        prev_grid = self.grid_player2.copy()
        
        color = self.current_bubble_color[player_num]  # Already 0-5, consistent with grid
        player_grid = self.grid_player2
        
        if (target_row, target_col) not in player_grid:
            player_grid[(target_row, target_col)] = color
            popped = self._pop_bubbles(target_row, target_col, color, player_num)
            self.scores[player_num] += popped
            # Suppress detailed pop logs for cleaner output
            
            # NEW: Color-matching incentives are reward-only (do not alter score)
            if popped == 0:
                color_matches = self._count_color_matches(player_grid, prev_grid)
                if color_matches > 0:
                    # Deferred to reward calculation below
                    pass
            
            if popped > 0:
                push_amount = popped * 6
                # Push center line away from AI (toward left)
                self.target_center_line_offset -= push_amount
                self.target_center_line_offset = max(-200, min(200, self.target_center_line_offset))
                self._shift_grids()
        
        # If grid cleared, mark as win and end episode
        if len(player_grid) == 0:
            self.done = True
            self.natural_endings += 1
            print(f"🎯 GAME END: WIN! Grid cleared in {self.step_count} steps. Final score: {self.scores[player_num]}")
        
        # Check for losing condition (bubbles reach right edge)
        if self._check_lose_condition(player_num):
            self.done = True
            self.lost[player_num] = True
            self.natural_endings += 1
            # Suppress loss print for clean output
        
        self.step_count += 1
        if self.step_count >= 400:
            self.done = True
            # Game ends if AI loses (bubbles reach edge) or time runs out
            if self.lost[player_num]:
                pass  # Already set
            else:
                # Time limit reached - AI wins if not lost
                pass
            self.artificial_endings += 1
            # Suppress timeout print for clean output
        
        self.current_bubble_color[player_num] = self.next_bubble_color[player_num]
        self.next_bubble_color[player_num] = random.randint(0, BUBBLE_COLORS_COUNT-1)
        
        new_grid = self.grid_player2.copy()
        
        # FIXED REWARD STRATEGY: Explicitly reward CURRENT bubble color matching only
        # - Small shot penalty (-0.05) encourages exploration and strategic placement
        # - Big bubble popping rewards (+30 per bubble) make popping highly desirable
        # - CURRENT bubble color matching rewards (+15 per match) encourage immediate color strategy
        # - NO rewards for next bubble planning - AI must focus on current shot only
        reward = compute_enhanced_reward(
            prev_score, self.scores[player_num],
            prev_grid, new_grid,
            self.done, self.lost[player_num],
            player_num
        )

        # FIXED: Only reward CURRENT bubble color matches (not next bubble planning)
        if popped == 0:
            # Count neighbors at the landing cell that match the PLACED bubble color
            # IMPORTANT: Use the just-shot color (not the updated current/next colors)
            current_bubble_color = color
            neighbor_matches = 0
            if (target_row, target_col) in player_grid:
                row, col = target_row, target_col
                if row % 2 == 0:
                    adjacent = [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col-1), (row+1, col-1)]
                else:
                    adjacent = [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col+1), (row+1, col+1)]
                for nr, nc in adjacent:
                    if (nr, nc) in player_grid and player_grid[(nr, nc)] == current_bubble_color:
                        neighbor_matches += 1
            if neighbor_matches > 0:
                # HIGHER reward for current bubble color matching to make it the primary strategy
                reward += neighbor_matches * 25.0  # Increased from 15.0 to 25.0 for stronger color matching
            else:
                # Small penalty for not matching current bubble color
                reward -= 5.0  # Encourage color matching
                # Additional small penalty for zero neighbor_same_color_counts feature at landing
                from bubble_geometry import neighbor_same_color_counts
                ncounts = neighbor_same_color_counts(player_grid, current_bubble_color)
                idx = target_row * GRID_COLS + target_col
                if 0 <= idx < ncounts.shape[0] and ncounts[idx] <= 0.0:
                    reward -= 1.0
                # No verbose print for no-match to reduce noise
        
        return self.get_state(player_num), reward, self.done, self.lost[player_num]
    
    def _shift_grids(self):
        """Attach both grids to the center line: update visual offset only.
        Do NOT mutate grid indices; the screen mapping uses center_line_offset.
        """
        # Keep the center line (and thus both grids visually) at the target offset
        self.center_line_offset = self.target_center_line_offset
        self.last_shift_offset = self.target_center_line_offset
    
    def _pop_bubbles(self, row, col, color, player_num):
        """BFS to find connected bubbles in honeycomb pattern"""
        player_grid = self.grid_player2  # Only player 2 exists in this environment
        visited = set()
        to_visit = [(row, col)]
        
        while to_visit:
            r, c = to_visit.pop()
            if (r, c) in visited:
                continue
            if r < 0 or r >= GRID_ROWS or c < 0 or c >= GRID_COLS:
                continue
            if (r, c) not in player_grid or player_grid[(r, c)] != color:
                continue
            
            visited.add((r, c))
            
            # Check adjacent positions in honeycomb pattern
            if r % 2 == 0:  # Even row
                adjacent = [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r-1, c-1), (r+1, c-1)]
            else:  # Odd row
                adjacent = [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r-1, c+1), (r+1, c+1)]
            
            for nr, nc in adjacent:
                to_visit.append((nr, nc))
        
        # Pop if 3 or more connected
        if len(visited) >= 3:
            for r, c in visited:
                del player_grid[(r, c)]
            
            # Check for isolated bubbles after popping (like in the real game)
            isolated_bubbles = self._check_isolated_bubbles(player_num)
            if isolated_bubbles:
                # Remove isolated bubbles (they would fall in the real game)
                for r, c in isolated_bubbles:
                    if (r, c) in player_grid:
                        del player_grid[(r, c)]
                
                # Return total bubbles removed (popped + fallen)
                return len(visited) + len(isolated_bubbles)
            
            return len(visited)
        return 0
    
    def _check_isolated_bubbles(self, player_num):
        """Check for bubbles that are no longer connected to the top row (would fall in real game)"""
        player_grid = self.grid_player2  # Only player 2 exists in this environment
        
        # Find all bubbles connected to the top row
        connected_to_top = set()
        to_check = []
        
        # Start with all bubbles in the top row
        for col in range(GRID_COLS):
            if (0, col) in player_grid:
                to_check.append((0, col))
                connected_to_top.add((0, col))
        
        # Breadth-first search to find all connected bubbles
        while to_check:
            row, col = to_check.pop(0)
            
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
                new_pos = (row + dr, col + dc)
                if (new_pos in player_grid and 
                    new_pos not in connected_to_top and 
                    0 <= new_pos[0] < GRID_ROWS and 
                    0 <= new_pos[1] < GRID_COLS):
                    connected_to_top.add(new_pos)
                    to_check.append(new_pos)
        
        # Any bubble not in connected_to_top is isolated (would fall)
        isolated = []
        for (row, col) in player_grid.keys():
            if (row, col) not in connected_to_top:
                isolated.append((row, col))
        
        return isolated
    
    def _count_color_matches(self, current_grid, prev_grid):
        """Count how many new color matches were created by the latest bubble placement"""
        if len(current_grid) <= len(prev_grid):
            return 0  # No new bubble was placed
        
        # Find the newly added bubble
        new_bubble_pos = None
        for pos in current_grid:
            if pos not in prev_grid:
                new_bubble_pos = pos
                break
        
        if new_bubble_pos is None:
            return 0
        
        new_bubble_color = current_grid[new_bubble_pos]
        color_matches = 0
        
        # Check adjacent positions for same color
        row, col = new_bubble_pos
        if row % 2 == 0:  # Even row
            adjacent = [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col-1), (row+1, col-1)]
        else:  # Odd row
            adjacent = [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col+1), (row+1, col+1)]
        
        for adj_pos in adjacent:
            if adj_pos in current_grid and current_grid[adj_pos] == new_bubble_color:
                color_matches += 1
        
        return color_matches
    
    def _check_lose_condition(self, player_num):
        """Check loss using shared geometry functions for consistency."""
        # Use shared function for consistency
        return check_lose_condition(self.grid_player2, player_num, self.center_line_offset)

    def get_hybrid_masked_targets(self, player_num: int):
        """Return occupied (row,col) bubbles filtered by hybrid masking (front-most + angular LOS)."""
        # Build grid map {(row,col): color_index}
        grid_map = { (r, c): color for (r, c), color in self.grid_player2.items() }
        # Shooter position (right side)
        SCREEN_WIDTH = 1200
        BUBBLE_RADIUS = 20
        shooter_x = SCREEN_WIDTH - BUBBLE_RADIUS * 2
        shooter_y = self.shooter_y[player_num]
        from bubble_geometry import get_hybrid_masked_targets as _hybrid
        return _hybrid(shooter_x, shooter_y, grid_map, player_num, self.center_line_offset)

def train_target_dqn(num_steps=1_000, save_path='target_bubbles_dqn_model.pth', num_envs=1,
                     agent_overrides: dict = None):
    # Prefer CUDA, then Apple Metal (MPS), else CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    # Single agent for right-side AI player only - NEW: Target-based actions
    agent = TargetDQNAgent(STATE_SIZE, TARGET_ACTION_SIZE, device, **(agent_overrides or {}))
    
    # Logging: console only (no file redirection)
    
    # Continue training from existing model if present
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path} for continued training...")
        agent.load(save_path)
    
    print(f"Training target-based DQN for right-side AI player with {num_envs} parallel environments...")
    print(f"Using device: {device}")
    print(f"State size: {STATE_SIZE}, Target Action size: {TARGET_ACTION_SIZE}")
    print(f"NEW: AI now chooses targets based on color analysis - true strategic learning!")
    print(f"TRAINING FOR {num_steps:,} STEPS (not episodes) - AI will continue playing until step count reached!")
    
    # Track cumulative performance across all steps
    total_steps_completed = 0
    total_reward = 0
    total_bubbles_popped = 0
    total_shots_taken = 0
    total_chain_reactions = 0
    total_falling_bubbles = 0
    
    # Track scores for each 1000-step window for learning analysis
    step_window_scores = []  # List to store total scores for each 1000-step window
    current_window_score = 0    # Running total for current 1000-step window
    current_window_steps = 0 # Count of steps in current window
    
    # NEW: Track bubble-popping scores separately (excluding loss penalties)
    step_window_bubble_scores = []  # List to store bubble-popping scores for each 1000-step window
    current_window_bubble_score = 0    # Running total of bubble-popping scores for current window
    
    # NEW: Track individual environment scores for better learning analysis
    step_window_env_scores = [[] for _ in range(num_envs)]  # List of lists: one list per environment
    current_window_env_scores = [0] * num_envs  # Running total for each environment in current window
    
    # Create multiple environments for parallel experience collection (ONCE, outside episode loop)
    enable_debug = False  # Disable debug rendering for training speed
    envs = [
            TargetBubbleShooterEnv(
            debug_render=(i == 0 and enable_debug),  # Only first environment shows debug
                debug_fps=1,
            )
            for i in range(num_envs)
        ]
    
    # Initialize environments
    states = [env.reset() for env in envs]  # Reset and get initial states
    active_envs = [True] * num_envs  # Track which environments are still active
    
    # Main training loop - continue until we reach the target number of steps
    WARMUP_STEPS = 0
    warmup_announced = True
    while total_steps_completed < num_steps:
        # Check if any environments need to be reset (game over or won)
        for i, env in enumerate(envs):
            if env.done:
                # Reset environment for continuous training
                states[i] = env.reset()
                active_envs[i] = True
                env.game_counted = False  # Reset game counter
        
        # AI player (player 2) actions - only for active environments
        valid_targets = []
        actions = []
        active_indices = []
        
        for i, (env, is_active) in enumerate(zip(envs, active_envs)):
            if is_active:
                # Get reachable targets (positions) among empty cells
                reachable_targets = env.get_hybrid_masked_targets(2) if AI_ENABLE_BOUNCE else env.get_reachable_empty_targets(2)
                
                # If no targets reachable, let AI shoot randomly (will lose anyway)
                if not reachable_targets:
                    # No valid targets - AI will shoot randomly and lose
                    # This prevents training from getting stuck
                    reachable_targets = []  # Empty list triggers random shooting
                
                # AI chooses a target action (0-699)
                target_action = agent.select_action(states[i], reachable_targets, training_mode=True)
                # Capture diagnostics for HUD on the environment instance
                try:
                    env._agent_rand = bool(agent.last_selection_random)
                    env._agent_eps = float(agent.last_epsilon)
                    env._agent_act = int(agent.last_selected_action) if agent.last_selected_action is not None else -1
                except Exception:
                    env._agent_rand = False
                    env._agent_eps = -1.0
                    env._agent_act = -1
                
                valid_targets.append(reachable_targets)
                actions.append(target_action)
                active_indices.append(i)
            
            # NOTE: removed legacy duplicate action block that overwrote empty-cell targets with bubble targets
            
            # Step all active environments for AI player
            for idx, (env_idx, target_action) in enumerate(zip(active_indices, actions)):
                env = envs[env_idx]  # Get the actual environment using the index
                if not env.done and active_envs[env_idx]:
                    # Store previous grid size to detect falling bubbles
                    prev_grid_size = len(env.grid_player2)
                    
                    # Pass target action to step method (AI chooses which target to shoot at)
                    result = env.step(2, target_action)  # Use target_action directly
                    agent.store_transition(states[env_idx], target_action, result[1], result[0], result[2])
                    
                    # Warmup: skip updates until buffer has enough samples
                    if len(agent.replay_buffer) >= WARMUP_STEPS:
                        agent.update()
                    
                    # Track individual environment rewards
                    current_window_env_scores[env_idx] += result[1]  # Add to individual environment score
                    total_shots_taken += 1
                    
                    # Track chain reactions (bubbles popped > 1)
                    bubbles_this_shot = 0
                    fallen_bubbles = 0
                    if hasattr(env, '_last_bubbles_popped'):
                        bubbles_this_shot = env.scores[2] - env._last_bubbles_popped
                        if bubbles_this_shot > 1:
                            total_chain_reactions += 1
                        
                        # Track falling bubbles
                        current_grid_size = len(env.grid_player2)
                        if bubbles_this_shot > 0 and current_grid_size < prev_grid_size - bubbles_this_shot:
                            fallen_bubbles = prev_grid_size - current_grid_size - bubbles_this_shot
                            total_falling_bubbles += fallen_bubbles
                    
                    env._last_bubbles_popped = env.scores[2]
                    
                    # Update state for continuing environment
                    states[env_idx] = result[0]
                    
                    # Increment total steps completed
                    total_steps_completed += 1
                    
                    # Update window tracking
                    current_window_score += result[1]
                    current_window_steps += 1
                    
                    # Calculate bubble-popping score for this step
                    if bubbles_this_shot > 0:
                        step_bubble_score = bubbles_this_shot * 20  # Base 20 points per bubble popped
                        if bubbles_this_shot > 1:
                            step_bubble_score += 5  # Bonus for chain reactions
                        if fallen_bubbles > 0:
                            step_bubble_score += fallen_bubbles * 15  # Bonus for falling bubbles
                        
                        current_window_bubble_score += step_bubble_score
                        total_bubbles_popped += bubbles_this_shot
                    
                    # Update total reward
                    total_reward += result[1]
                    
                    # Check if we've reached the target number of steps
                    if total_steps_completed >= num_steps:
                        break
        
        # Update networks every step for continuous learning (only after warmup)
        if len(agent.replay_buffer) >= WARMUP_STEPS:
            agent.update()
        
        # Update target networks periodically
        if total_steps_completed % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
        
        # Update learning rate with decay
        new_lr = agent.update_learning_rate(total_steps_completed // 10000)  # Decay every 10k steps
        
        # Save model more frequently for 20k training
        if total_steps_completed % 5000 == 0:
            agent.save(save_path)
            print(f"Model saved at step {total_steps_completed:,}")
        
        # Log learning rate changes more frequently for 20k training
        if total_steps_completed % 10000 == 0:
            print(f"📉 Learning rate updated to: {new_lr:.2e}")
        
        # When we reach 1000 steps, save the window score and reset
        if current_window_steps >= 1000:
            # Skip logging warmup windows: only log once warmup complete
            if total_steps_completed - current_window_steps < WARMUP_STEPS:
                # Reset counters but do not record this window
                current_window_score = 0
                current_window_steps = 0
                current_window_bubble_score = 0
                current_window_env_scores = [0] * num_envs
                continue
            step_window_scores.append(current_window_score)
            step_window_bubble_scores.append(current_window_bubble_score)
            
            # Save individual environment scores for this window
            for env_idx in range(num_envs):
                step_window_env_scores[env_idx].append(current_window_env_scores[env_idx])
            
            print(f"📊 1000-Step Window {len(step_window_scores)}: Total Reward = {current_window_score:.1f}, Bubble Score = {current_window_bubble_score:.1f}")
            print(f"   Environment Scores: {[f'Env{i}: {score:.1f}' for i, score in enumerate(current_window_env_scores)]}")
            print(f"   Total Steps: {total_steps_completed:,} / {num_steps:,} ({total_steps_completed/num_steps*100:.1f}%)")
            
            # Reset counters for next window
            current_window_score = 0
            current_window_steps = 0
            current_window_bubble_score = 0
            current_window_env_scores = [0] * num_envs
            
                # No intermediate plots - only final plot at end of training
        
        # Progress reporting - only summary every 1000 steps (after warmup)
        if total_steps_completed % 1000 == 0 and total_steps_completed >= WARMUP_STEPS:
            avg_reward = current_window_score / max(current_window_steps, 1)
            print(f"Step {total_steps_completed:,}: Avg Reward: {avg_reward:.2f}, Shots: {total_shots_taken}, Chain: {total_chain_reactions}, Falling: {total_falling_bubbles}")
            print(f"Cumulative: {total_steps_completed:,} steps, {total_reward:.1f} total reward, {total_bubbles_popped} bubbles popped")
            print("="*50)
        
        # Debug output every 5000 steps to help diagnose issues
        if total_steps_completed % 5000 == 0:
            active_count = sum(active_envs)
            done_count = sum(1 for env in envs if env.done)
            print(f"DEBUG: Active envs: {active_count}, Done envs: {done_count}, Total Steps: {total_steps_completed:,}")
        
        # Check if we've reached the target number of steps
        if total_steps_completed >= num_steps:
            break
    
    # Save final model
    agent.save(save_path)
    print("Training complete. Model saved to", save_path)
    
    # NEW: Feature importance analysis (first-layer weight L1 norms)
    try:
        with torch.no_grad():
            # Extract first linear layer weights (shape: [512, STATE_SIZE])
            first_linear = None
            for m in agent.policy_net.net:
                if isinstance(m, torch.nn.Linear):
                    first_linear = m
                    break
            if first_linear is not None:
                W = first_linear.weight.detach().cpu().abs()  # [512, STATE_SIZE]
                importances = W.sum(dim=0).numpy()            # [STATE_SIZE]
                # Build human-readable labels
                labels = []
                # 0..699: grid color indices
                for idx in range(GRID_ROWS * GRID_COLS):
                    r = idx // GRID_COLS
                    c = idx % GRID_COLS
                    labels.append(f"grid_color[{r},{c}]")
                # 700..1399: neighbor counts
                for idx in range(GRID_ROWS * GRID_COLS):
                    r = idx // GRID_COLS
                    c = idx % GRID_COLS
                    labels.append(f"neighbor_count[{r},{c}]")
                # 1400: current bubble color (index)
                labels.append("current_color_idx")
                # 1401..1403: strategic features
                labels.append("feat_density_ai")
                labels.append("feat_density_opp")
                labels.append("feat_score_norm")
                # Sort top 20
                idx_sorted = np.argsort(importances)[::-1]
                top_k = 20
                top_idx = idx_sorted[:top_k]
                top_scores = importances[top_idx]
                top_labels = [labels[i] for i in top_idx]
                # Plot top-20 bar chart
                plt.figure(figsize=(14, 7))
                y_pos = np.arange(len(top_labels))
                plt.barh(y_pos, top_scores, color='tab:blue')
                plt.yticks(y_pos, top_labels)
                plt.gca().invert_yaxis()
                plt.xlabel('Feature Importance (L1 sum of first-layer weights)')
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
                print("📊 Feature importance plot saved as: feature_importance_top20.png")
                
                # Plot top-100 bar chart
                top_k_100 = 100
                top_idx_100 = idx_sorted[:top_k_100]
                top_scores_100 = importances[top_idx_100]
                top_labels_100 = [labels[i] for i in top_idx_100]
                plt.figure(figsize=(16, 18))
                y_pos_100 = np.arange(len(top_labels_100))
                plt.barh(y_pos_100, top_scores_100, color='tab:green')
                plt.yticks(y_pos_100, top_labels_100, fontsize=8)
                plt.gca().invert_yaxis()
                plt.xlabel('Feature Importance (L1 sum of first-layer weights)')
                plt.title('Top 100 Feature Importances')
                plt.tight_layout()
                plt.savefig('feature_importance_top100.png', dpi=300, bbox_inches='tight')
                print("📊 Feature importance plot saved as: feature_importance_top100.png")
                # Print full ranking summary
                print("\n🔎 Feature importance ranking (desc):")
                for rank, i in enumerate(idx_sorted, start=1):
                    print(f"{rank:3d}. {labels[i]}: {importances[i]:.6f}")
    except Exception as e:
        print(f"Feature importance analysis skipped due to error: {e}")

    # FINAL: Generate comprehensive learning progress plot at end of training
    try:
        if len(step_window_scores) > 0:
            generate_learning_progress_plot(step_window_scores, step_window_bubble_scores, step_window_env_scores)
            print("📊 Final learning progress plot generated successfully")
    except Exception as e:
        print(f"Final learning progress plot skipped due to error: {e}")
    
    # NEW: Simple scores-per-1k-steps plot
    try:
        if len(step_window_scores) > 0:
            plt.figure(figsize=(12, 6))
            x = [(i + 1) * 1000 for i in range(len(step_window_scores))]
            plt.plot(x, step_window_scores, marker='o', linewidth=2, color='tab:green')
            plt.xlabel('Steps')
            plt.ylabel('Total Reward per 1000 Steps')
            plt.title('Reward per 1000-Step Window (Reset Each Window)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('reward_per_1k_steps.png', dpi=300, bbox_inches='tight')
            print("📈 Reward per 1k steps plot saved as: reward_per_1k_steps.png")
    except Exception as e:
        print(f"Scores-per-1k plot skipped due to error: {e}")

    # Display final training statistics
    print("\n" + "="*50)
    print("FINAL TRAINING STATISTICS")
    print("="*50)
    print(f"Total steps completed: {total_steps_completed:,}")
    print(f"Total reward accumulated: {total_reward:.1f}")
    print(f"Total bubbles popped: {total_bubbles_popped}")
    print(f"Total shots taken: {total_shots_taken}")
    print(f"Total chain reactions: {total_chain_reactions}")
    print(f"Total falling bubbles: {total_falling_bubbles}")
    if total_steps_completed > 0:
        avg_reward_per_step = total_reward / total_steps_completed
        avg_bubbles_per_step = total_bubbles_popped / total_steps_completed
        avg_shots_per_step = total_shots_taken / total_steps_completed
        chain_reaction_rate = (total_chain_reactions / total_shots_taken * 100) if total_shots_taken > 0 else 0
        falling_bubble_rate = (total_falling_bubbles / total_bubbles_popped * 100) if total_bubbles_popped > 0 else 0
        print(f"Average reward per step: {avg_reward_per_step:.2f}")
        print(f"Average bubbles popped per step: {avg_bubbles_per_step:.1f}")
        print(f"Average shots per step: {avg_shots_per_step:.1f}")
        print(f"Chain reaction rate: {chain_reaction_rate:.1f}%")
        print(f"Falling bubble rate: {falling_bubble_rate:.1f}%")
    print("="*50)

def generate_learning_progress_plot(step_window_scores, step_window_bubble_scores, step_window_env_scores):
    """Generate a plot showing AI learning progress over 1000-step windows with individual environment tracking"""
    print("\n🎯 Generating Learning Progress Plot...")
    
    # Create x-axis (step numbers in thousands)
    steps = [(i + 1) * 1000 for i in range(len(step_window_scores))]
    
    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(16, 12))
    
    # Primary y-axis: Bubble-popping scores (what we care about most)
    color1 = 'tab:blue'
    ax1.set_xlabel('Steps (thousands)', fontsize=14)
    ax1.set_ylabel('Bubble-Popping Score (Last 1000 Steps)', fontsize=14, color=color1)
    
    # Main plot line for bubble scores
    line1 = ax1.plot(steps, step_window_bubble_scores, 'o-', color=color1, 
                     linewidth=3, markersize=8, label='Bubble-Popping Score (All Envs)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot individual environment bubble scores
    env_colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for env_idx in range(len(step_window_env_scores)):
        if len(step_window_env_scores[env_idx]) == len(steps):
            env_color = env_colors[env_idx % len(env_colors)]
            ax1.plot(steps, step_window_env_scores[env_idx], '--', color=env_color, 
                     linewidth=1.5, alpha=0.7, label=f'Environment {env_idx}')
    
    # Secondary y-axis: Total rewards (including penalties)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Total Reward (Last 1000 Steps)', fontsize=14, color=color2)
    
    # Secondary plot line for total rewards
    line2 = ax2.plot(steps, step_window_scores, 's--', color=color2, 
                     linewidth=2, markersize=6, label='Total Reward (All Envs)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add trend lines
    if len(step_window_bubble_scores) > 1:
        # Trend line for bubble scores (primary metric)
        z1 = np.polyfit(steps, step_window_bubble_scores, 1)
        p1 = np.poly1d(z1)
        ax1.plot(steps, p1(steps), color=color1, alpha=0.7, linewidth=3, 
                label=f'Bubble Score Trend (slope: {z1[0]:.1f})')
        
        # Trend line for total rewards
        z2 = np.polyfit(steps, step_window_scores, 1)
        p2 = np.poly1d(z2)
        ax2.plot(steps, p2(steps), color=color2, alpha=0.7, linewidth=3, 
                label=f'Total Reward Trend (slope: {z2[0]:.1f})')
    
    # Customize the plot
    plt.title('AI Learning Progress: Individual Environment Performance Over Training (Every 1k Steps)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, ncol=2)
    
    # Add annotations for key insights
    if len(step_window_bubble_scores) > 1:
        first_bubble_score = step_window_bubble_scores[0]
        last_bubble_score = step_window_bubble_scores[-1]
        bubble_improvement = last_bubble_score - first_bubble_score
        
        ax1.annotate(f'Start: {first_bubble_score:.0f}', 
                    xy=(steps[0], first_bubble_score), 
                    xytext=(steps[0] + 2000, first_bubble_score + max(step_window_bubble_scores) * 0.1),
                    arrowprops=dict(arrowstyle='->', color=color1),
                    fontsize=10, color=color1)
        
        ax1.annotate(f'End: {last_bubble_score:.0f}', 
                    xy=(steps[-1], last_bubble_score), 
                    xytext=(steps[-1] - 2000, last_bubble_score - max(step_window_bubble_scores) * 0.3),
                    arrowprops=dict(arrowstyle='->', color=color1),
                    fontsize=10, color=color1)
        
        if bubble_improvement > 0:
            ax1.text(0.02, 0.98, f'Bubble Score Improvement: +{bubble_improvement:.0f}', 
                    transform=ax1.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax1.text(0.02, 0.98, f'Bubble Score Change: {bubble_improvement:.0f}', 
                    transform=ax1.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Save the plot with step count in filename
    current_steps = len(step_window_scores) * 1000
    plot_filename = f'ai_learning_progress_{current_steps}steps_1k_intervals.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Learning progress plot saved as: {plot_filename}")
    
    # Show the plot
    #plt.show()
    
    # Print analysis
    print(f"\n📊 Learning Progress Analysis ({current_steps:,} Steps):")
    print(f"First 1000 steps bubble score: {step_window_bubble_scores[0]:.0f}")
    print(f"Last 1000 steps bubble score: {step_window_bubble_scores[-1]:.0f}")
    print(f"Bubble score improvement: {step_window_bubble_scores[-1] - step_window_bubble_scores[0]:.0f}")
    print(f"Average bubble score per 1000 steps: {np.mean(step_window_bubble_scores):.0f}")
    print(f"Total reward improvement: {step_window_scores[-1] - step_window_scores[0]:.0f}")
    print(f"Average total reward per 1000 steps: {np.mean(step_window_scores):.0f}")
    
    # Individual environment analysis
    print("\n🔍 Individual Environment Analysis:")
    for env_idx in range(len(step_window_env_scores)):
        if len(step_window_env_scores[env_idx]) > 0:
            env_scores = step_window_env_scores[env_idx]
            if len(env_scores) >= 2:
                improvement = env_scores[-1] - env_scores[0]
                print(f"Environment {env_idx}: Start={env_scores[0]:.0f}, End={env_scores[1]:.0f}, Change={improvement:+.0f}")
    
    if step_window_bubble_scores[-1] > step_window_bubble_scores[0]:
        print("\n🎉 POSITIVE TREND: The AI is learning to pop more bubbles!")
    else:
        print("\n⚠️  NO IMPROVEMENT: The AI may need training adjustments.")

if __name__ == '__main__':
    print("Starting single-player training for right-side AI bubble shooter...")
    train_target_dqn(num_steps=50_000, num_envs=1)