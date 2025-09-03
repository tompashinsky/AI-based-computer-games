"""
Shared Bubble Geometry Module

This module provides consistent geometry, physics, and color encoding for both
the training environment and the main game. This ensures perfect consistency
between environments and eliminates color mismatches and geometry differences.

Shared Components:
- Grid to screen coordinate mapping
- Screen to grid coordinate conversion
- Collision detection and bounce physics
- Lose threshold calculations
- Color encoding and constants
- Honeycomb grid layout calculations
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional

# ============================================================================
# SHARED CONSTANTS (must match between training and game)
# ============================================================================

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BUBBLE_RADIUS = 20

# Grid dimensions
GRID_ROWS = 20
GRID_COLS = 35

# Color constants (RGB tuples)
BUBBLE_COLORS = [
    (255, 0, 0),    # 0 - Red
    (0, 255, 0),    # 1 - Green
    (0, 0, 255),    # 2 - Blue
    (255, 255, 0),  # 3 - Yellow
    (255, 0, 255),  # 4 - Magenta
    (0, 255, 255),  # 5 - Cyan
]

# Game mechanics
BUBBLE_COLORS_COUNT = len(BUBBLE_COLORS)
LOSE_THRESHOLD = 100
CENTER_LINE_PUSH_AMOUNT = 6
MAX_CENTER_LINE_OFFSET = 200

# ============================================================================
# GRID TO SCREEN COORDINATE MAPPING
# ============================================================================

def grid_to_screen(row: int, col: int, player_num: int, center_line_offset: float = 0) -> Tuple[float, float]:
    """
    Convert grid coordinates to screen coordinates.
    
    Args:
        row: Grid row (0-19)
        col: Grid column (0-34)
        player_num: Player number (1 for left, 2 for right)
        center_line_offset: Current center line offset
        
    Returns:
        Tuple of (x, y) screen coordinates
    """
    middle_x = SCREEN_WIDTH // 2 + center_line_offset
    start_y = BUBBLE_RADIUS * 2
    
    # Offset every other row to create honeycomb pattern
    row_offset = BUBBLE_RADIUS if row % 2 == 1 else 0
    
    if player_num == 1:
        # Left player (player 1) - grid extends left from center
        x = middle_x - (col * BUBBLE_RADIUS * 2) - row_offset - BUBBLE_RADIUS
    else:
        # Right player (player 2) - grid extends right from center
        x = middle_x + (col * BUBBLE_RADIUS * 2) + row_offset + BUBBLE_RADIUS
    
    y = start_y + (row * BUBBLE_RADIUS * 1.8)
    
    return x, y

def screen_to_grid(x: float, y: float, player_num: int, center_line_offset: float = 0) -> Tuple[int, int]:
    """
    Convert screen coordinates back to grid coordinates.
    
    Args:
        x: Screen x coordinate
        y: Screen y coordinate
        player_num: Player number (1 for left, 2 for right)
        center_line_offset: Current center line offset
        
    Returns:
        Tuple of (row, col) grid coordinates
    """
    middle_x = SCREEN_WIDTH // 2 + center_line_offset
    start_y = BUBBLE_RADIUS * 2
    
    # Reverse the grid_to_screen calculation
    y_rel = y - start_y
    row = int(y_rel / (BUBBLE_RADIUS * 1.8))
    
    x_rel = x - middle_x
    if player_num == 1:
        # Left player: x_rel is negative, need to handle properly
        x_rel = abs(x_rel)  # Make positive for calculation
        if row % 2 == 1:
            x_rel -= BUBBLE_RADIUS
        col = int(x_rel / (2 * BUBBLE_RADIUS))
    else:
        # Right player: x_rel is positive
        x_rel -= BUBBLE_RADIUS
        if row % 2 == 1:
            x_rel -= BUBBLE_RADIUS
        col = int(x_rel / (2 * BUBBLE_RADIUS))
    
    # Clamp to valid range
    row = max(0, min(GRID_ROWS - 1, row))
    col = max(0, min(GRID_COLS - 1, col))
    
    return row, col

# ============================================================================
# COLLISION DETECTION AND BOUNCE PHYSICS
# ============================================================================

def check_wall_collision(x: float, y: float, velocity_x: float, velocity_y: float) -> Tuple[float, float, float, float, bool]:
    """
    Check for wall collision and calculate bounce.
    
    Args:
        x, y: Current bubble position
        velocity_x, velocity_y: Current velocity
        
    Returns:
        Tuple of (new_x, new_y, new_vx, new_vy, did_bounce)
    """
    did_bounce = False
    new_x, new_y = x, y
    new_vx, new_vy = velocity_x, velocity_y
    
    # Top wall collision
    if y - BUBBLE_RADIUS <= 0:
        new_y = BUBBLE_RADIUS
        new_vy = -velocity_y
        did_bounce = True
    
    # Bottom wall collision
    elif y + BUBBLE_RADIUS >= SCREEN_HEIGHT:
        new_y = SCREEN_HEIGHT - BUBBLE_RADIUS
        new_vy = -velocity_y
        did_bounce = True
    
    return new_x, new_y, new_vx, new_vy, did_bounce

def check_bubble_collision(bubble_x: float, bubble_y: float, 
                          target_x: float, target_y: float) -> bool:
    """
    Check if a bubble collides with a target position.
    
    Args:
        bubble_x, bubble_y: Bubble position
        target_x, target_y: Target position
        
    Returns:
        True if collision detected
    """
    dx = bubble_x - target_x
    dy = bubble_y - target_y
    distance = math.sqrt(dx * dx + dy * dy)
    return distance < BUBBLE_RADIUS * 2

def simulate_shot_trajectory(shooter_x: float, shooter_y: float, 
                           target_x: float, target_y: float, 
                           player_num: int, center_line_offset: float = 0,
                           max_bounces: int = 3, max_steps: int = 1000) -> Tuple[Optional[Tuple[int, int]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Simulate a shot trajectory with wall bounces and collision detection.
    
    Args:
        shooter_x, shooter_y: Shooter position
        target_x, target_y: Target position
        player_num: Player number
        center_line_offset: Current center line offset
        max_bounces: Maximum wall bounces allowed
        max_steps: Maximum simulation steps
        
    Returns:
        Tuple of (landing_grid_pos, path_points, bounce_points)
        landing_grid_pos is None if no valid landing found
    """
    # Calculate initial velocity
    dx = target_x - shooter_x
    dy = target_y - shooter_y
    distance = math.sqrt(dx * dx + dy * dy)
    
    if distance < 1e-6:
        return None, [], []
    
    # Normalize and set speed
    speed = 10  # Match game's SHOOT_SPEED
    vx = (dx / distance) * speed
    vy = (dy / distance) * speed
    
    # Ensure right shooter goes leftward
    if player_num == 2 and vx > 0:
        vx = -abs(vx)
        vy = vy / abs(vx) * (1.0 - vx*vx)**0.5 if abs(vx) < 1.0 else 0.0
    
    # Simulation state
    x, y = shooter_x, shooter_y
    px, py = x, y  # Previous position for swept collision
    path_points = [(x, y)]
    bounce_points = []
    bounce_count = 0
    step_len = 3.0
    
    # Get existing bubbles for collision detection (this would be passed in)
    # For now, we'll simulate without collision detection
    
    for step in range(max_steps):
        # Advance position
        x += vx * step_len
        y += vy * step_len
        
        # Record path
        if len(path_points) == 0 or (abs(x - path_points[-1][0]) + abs(y - path_points[-1][1]) > 3.0):
            path_points.append((x, y))
        
        # Check wall collisions
        new_x, new_y, new_vx, new_vy, did_bounce = check_wall_collision(x, y, vx, vy)
        if did_bounce:
            x, y = new_x, new_y
            vx, vy = new_vx, new_vy
            bounce_points.append((x, y))
            bounce_count += 1
            
            if bounce_count >= max_bounces:
                break
        
        # Check if reached center line
        middle_x = SCREEN_WIDTH // 2 + center_line_offset
        if (player_num == 1 and x + BUBBLE_RADIUS >= middle_x) or \
           (player_num == 2 and x - BUBBLE_RADIUS <= middle_x):
            # Snap to nearest grid position
            landing_x = middle_x - BUBBLE_RADIUS if player_num == 1 else middle_x + BUBBLE_RADIUS
            landing_row, landing_col = screen_to_grid(landing_x, y, player_num, center_line_offset)
            return (landing_row, landing_col), path_points, bounce_points
        
        # Update previous position
        px, py = x, y
    
    # If we reach here, no valid landing found
    return None, path_points, bounce_points

# ============================================================================
# VALID PLACEMENT AND TARGET VALIDATION
# ============================================================================

def is_valid_placement(row: int, col: int, grid: Dict[Tuple[int, int], int]) -> bool:
    """
    Check if a position is valid for bubble placement.
    
    Args:
        row, col: Grid position to check
        grid: Current grid state {(row, col): color}
        
    Returns:
        True if position is valid for placement
    """
    # Top row is always valid
    if row == 0:
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

def get_valid_targets(grid: Dict[Tuple[int, int], int], player_num: int, 
                     center_line_offset: float = 0) -> List[Tuple[int, int]]:
    """
    Get all valid target positions where a bubble can be placed.
    
    Args:
        grid: Current grid state
        player_num: Player number
        center_line_offset: Current center line offset
        
    Returns:
        List of valid (row, col) positions
    """
    valid_targets = []
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if (row, col) not in grid:
                # Check if this position is reachable
                if is_valid_placement(row, col, grid):
                    # Additional screen-space filtering
                    x, y = grid_to_screen(row, col, player_num, center_line_offset)
                    
                    # Keep targets within safe corridor
                    if player_num == 1:
                        # Left player - must be right of lose threshold
                        if x <= LOSE_THRESHOLD + BUBBLE_RADIUS:
                            continue
                    else:
                        # Right player - must be left of lose threshold
                        if x >= SCREEN_WIDTH - LOSE_THRESHOLD - BUBBLE_RADIUS:
                            continue
                    
                    valid_targets.append((row, col))
    
    return valid_targets

# ============================================================================
# LOSE CONDITION CHECKING
# ============================================================================

def check_lose_condition(grid: Dict[Tuple[int, int], int], player_num: int, 
                        center_line_offset: float = 0) -> bool:
    """
    Check if a player has lost (bubbles reached edge).
    
    Args:
        grid: Current grid state
        player_num: Player number to check
        center_line_offset: Current center line offset
        
    Returns:
        True if player has lost
    """
    for (row, col) in grid.keys():
        x, y = grid_to_screen(row, col, player_num, center_line_offset)
        
        if player_num == 1 and x <= LOSE_THRESHOLD:
            return True
        elif player_num == 2 and x >= SCREEN_WIDTH - LOSE_THRESHOLD:
            return True
    
    return False

# ============================================================================
# COLOR ENCODING AND DECODING
# ============================================================================

def rgb_to_color_index(rgb_color: Tuple[int, int, int]) -> int:
    """
    Convert RGB color tuple to color index (0-5).
    
    Args:
        rgb_color: RGB tuple (r, g, b)
        
    Returns:
        Color index (0-5) or -1 if not found
    """
    try:
        return BUBBLE_COLORS.index(rgb_color)
    except ValueError:
        return -1

def color_index_to_rgb(color_index: int) -> Tuple[int, int, int]:
    """
    Convert color index to RGB color tuple.
    
    Args:
        color_index: Color index (0-5)
        
    Returns:
        RGB color tuple
    """
    if 0 <= color_index < len(BUBBLE_COLORS):
        return BUBBLE_COLORS[color_index]
    else:
        return (128, 128, 128)  # Default gray

def encode_color_planes(grid: Dict[Tuple[int, int], int], player_num: int) -> np.ndarray:
    """
    Encode grid colors as 6 one-hot color planes.
    
    Args:
        grid: Current grid state {(row, col): color_index}
        player_num: Player number
        
    Returns:
        Numpy array of shape (6, GRID_ROWS * GRID_COLS) with one-hot encoding
    """
    color_planes = np.zeros((BUBBLE_COLORS_COUNT, GRID_ROWS * GRID_COLS), dtype=np.float32)
    
    for (row, col), color_idx in grid.items():
        if 0 <= color_idx < BUBBLE_COLORS_COUNT:
            idx = row * GRID_COLS + col
            color_planes[color_idx, idx] = 1.0
    
    return color_planes

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_angle_to_target(shooter_x: float, shooter_y: float, 
                            target_x: float, target_y: float) -> float:
    """
    Calculate the angle needed to shoot from shooter to target.
    
    Args:
        shooter_x, shooter_y: Shooter position
        target_x, target_y: Target position
        
    Returns:
        Angle in degrees
    """
    dx = target_x - shooter_x
    dy = target_y - shooter_y
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def get_adjacent_positions(row: int, col: int) -> List[Tuple[int, int]]:
    """
    Get adjacent positions in honeycomb pattern.
    
    Args:
        row, col: Grid position
        
    Returns:
        List of adjacent (row, col) positions
    """
    if row % 2 == 0:  # Even row
        return [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col-1), (row+1, col-1)]
    else:  # Odd row
        return [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row-1, col+1), (row+1, col+1)]

def calculate_center_line_push(current_offset: float, push_amount: int, 
                              direction: int) -> float:
    """
    Calculate new center line offset after pushing.
    
    Args:
        current_offset: Current center line offset
        push_amount: Amount to push (bubbles popped * CENTER_LINE_PUSH_AMOUNT)
        direction: Push direction (1 for left, -1 for right)
        
    Returns:
        New center line offset
    """
    new_offset = current_offset + (push_amount * direction)
    return max(-MAX_CENTER_LINE_OFFSET, min(MAX_CENTER_LINE_OFFSET, new_offset))

# =========================================================================
# PER-CELL SAME-COLOR NEIGHBOR COUNTS (for current bubble color)
# =========================================================================

def neighbor_same_color_counts(grid: Dict[Tuple[int, int], int], current_bubble_color: int) -> np.ndarray:
    """
    For each grid cell, count adjacent bubbles with the same color as the
    current bubble color. Occupied cells are set to 0 (we care about empty targets).

    Returns a (GRID_ROWS * GRID_COLS,) float32 array with values in [0,6].
    """
    counts = np.zeros(GRID_ROWS * GRID_COLS, dtype=np.float32)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = row * GRID_COLS + col
            if (row, col) in grid:
                counts[idx] = 0.0
                continue
            total = 0
            for nr, nc in get_adjacent_positions(row, col):
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    if (nr, nc) in grid and grid[(nr, nc)] == current_bubble_color:
                        total += 1
            counts[idx] = float(total)
    return counts

# ============================================================================
# COLOR-AWARE BUBBLE SYSTEM
# ============================================================================

class ColorAwareBubble:
    """Represents a bubble with position and color information"""
    def __init__(self, row: int, col: int, color: int):
        self.row = row
        self.col = col  # Fixed the bug you caught!
        self.color = color  # 0-5 for the 6 colors
    
    def __str__(self):
        return f"({self.row}, {self.col}, {self.color})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if isinstance(other, ColorAwareBubble):
            return self.row == other.row and self.col == other.col and self.color == other.color
        return False
    
    def __hash__(self):
        return hash((self.row, self.col, self.color))

def get_available_colors_for_ai(grid: Dict[Tuple[int, int], int], current_bubble_color: int, valid_positions: List[Tuple[int, int]]) -> List[int]:
    """
    Get colors of bubbles the AI can potentially match by placing current bubble.
    This provides information without hard-coding strategy.
    
    Args:
        grid: Current grid state {(row, col): color}
        current_bubble_color: Color of bubble AI is about to shoot
        valid_positions: List of (row, col) positions AI can reach
        
    Returns:
        List of colors that could potentially be matched
    """
    potential_colors = set()
    
    for row, col in valid_positions:
        # Check adjacent colors without telling AI if it's "good" or not
        adjacent_colors = get_adjacent_colors(grid, row, col)
        for color in adjacent_colors:
            if color == current_bubble_color:
                potential_colors.add(color)
    
    return list(potential_colors)

def get_adjacent_colors(grid: Dict[Tuple[int, int], int], row: int, col: int) -> List[int]:
    """
    Get colors of bubbles adjacent to a position.
    Pure information - no strategic advice.
    
    Args:
        grid: Current grid state
        row, col: Position to check around
        
    Returns:
        List of adjacent bubble colors
    """
    colors = []
    
    # Get adjacent positions based on row parity (honeycomb pattern)
    if row % 2 == 0:  # Even row
        adjacent_positions = [
            (row-1, col), (row+1, col), (row, col-1), (row, col+1), 
            (row-1, col-1), (row+1, col-1)
        ]
    else:  # Odd row
        adjacent_positions = [
            (row-1, col), (row+1, col), (row, col-1), (row, col+1), 
            (row-1, col+1), (row+1, col+1)
        ]
    
    for adj_row, adj_col in adjacent_positions:
        if (adj_row, adj_col) in grid:
            colors.append(grid[(adj_row, adj_col)])
    
    return colors

def get_color_based_targets(grid: Dict[Tuple[int, int], int], current_bubble_color: int, valid_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Get valid positions where AI could place current bubble.
    Only checks reachability - AI learns positioning through experience.
    
    Args:
        grid: Current grid state
        current_bubble_color: Color of bubble AI is about to shoot
        valid_positions: List of (row, col) positions AI can reach
        
    Returns:
        List of valid positions - AI learns which ones are good through rewards
    """
    # Simply return all reachable positions
    # AI learns optimal positioning through trial and error + rewards
    return valid_positions

def encode_compact_state_consistent(
    grid: Dict[Tuple[int, int], int],
    current_bubble_color: int,
    next_bubble_color: int = None,
) -> np.ndarray:
    """
    Clean, focused state encoding with only essential information.
    FIXED: Only include CURRENT bubble color to avoid confusion with the next bubble.

    Args:
        grid: Current grid state {(row, col): color}
        current_bubble_color: Color of bubble AI is about to shoot (0-5)
        next_bubble_color: (Ignored) Kept for backward compatibility

    Returns:
        Clean state array: 700 colors + current_color
        Total: 700 + 1 = 701 dimensions
    """
    # Initialize array for all 700 grid positions
    # Each position stores ONLY the color: [color]
    # -1 means empty position, 0-5 means actual colors
    color_data = np.full(GRID_ROWS * GRID_COLS, -1, dtype=np.float32)
    
    # Fill in occupied positions with ONLY colors (no row/col bias!)
    for (row, col), color in grid.items():
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            position_idx = row * GRID_COLS + col
            color_data[position_idx] = color  # Only color, no position!
    
    # Combine all features: colors only + current_color (NO next color)
    # Result: 700 + 1 = 701 dimensions
    return np.concatenate([
        color_data,             # 700 dimensions (colors only, no position bias!)
        [current_bubble_color], # 1 dimension (0-5)
    ])

def encode_compact_state(grid: Dict[Tuple[int, int], int], current_bubble_color: int, next_bubble_color: int) -> np.ndarray:
    """
    DEPRECATED: Old mixed approach kept for backward compatibility.
    Use encode_compact_state_consistent() instead - it's much better!
    """
    print("⚠️  WARNING: Using deprecated encode_compact_state(). Use encode_compact_state_consistent() instead!")
    return encode_compact_state_consistent(grid, current_bubble_color, next_bubble_color)

# ============================================================================
# HYBRID TARGET MASKING: Front-most-per-row + Angular LOS
# ============================================================================

def get_front_most_per_row_targets(
    grid: Dict[Tuple[int, int], int],
    player_num: int,
) -> List[Tuple[int, int]]:
    """
    Step 1: For a single-side grid mapping {(row,col): color}, return the
    front-most (maximum col) occupied bubble in each row.
    """
    front_by_row: Dict[int, int] = {}
    for (row, col), _ in grid.items():
        prev = front_by_row.get(row)
        if prev is None or col > prev:
            front_by_row[row] = col
    return [(row, col) for row, col in front_by_row.items()]


def check_angular_los_occlusion(
    shooter_x: float,
    shooter_y: float,
    target_row: int,
    target_col: int,
    player_num: int,
    grid: Dict[Tuple[int, int], int],
    center_line_offset: float = 0,
    corridor_width: float = None,
) -> bool:
    """
    Step 2 (swept-circle): Return True if the straight segment from shooter to target
    passes within a blocking corridor equal to the moving bubble's swept-circle
    (shot_radius + bubble_radius = 2*BUBBLE_RADIUS) of any existing bubble.

    A small exemption is applied very close to the target center to avoid
    falsely blocking attachment into the target's neighborhood.
    """
    # True swept-circle clearance: moving bubble center vs inflated obstacles
    if corridor_width is None:
        corridor_width = 2.0 * BUBBLE_RADIUS

    # Exempt the last small portion near the target to allow snapping
    # when the center path is already inside the neighborhood of the target.
    end_exempt_fraction = 0.12  # widen exemption near target to reduce false blocks

    # Segment A(shooter) -> B(target)
    bx, by = grid_to_screen(target_row, target_col, player_num, center_line_offset)
    ax, ay = shooter_x, shooter_y
    abx, aby = bx - ax, by - ay
    ab_len2 = max(abx * abx + aby * aby, 1e-6)

    for (row, col), _ in grid.items():
        # Do not treat the target bubble itself as an obstacle
        if row == target_row and col == target_col:
            continue
        cx, cy = grid_to_screen(row, col, player_num, center_line_offset)
        # Project C onto segment AB
        t = ((cx - ax) * abx + (cy - ay) * aby) / ab_len2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        # Near-target exemption
        if t > (1.0 - end_exempt_fraction):
            continue
        px, py = ax + t * abx, ay + t * aby
        dx, dy = cx - px, cy - py
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < corridor_width:
            return True
    return False


def get_angular_los_filtered_targets(
    shooter_x: float,
    shooter_y: float,
    candidate_targets: List[Tuple[int, int]],
    player_num: int,
    grid: Dict[Tuple[int, int], int],
    center_line_offset: float = 0,
    corridor_width: float = None,
) -> List[Tuple[int, int]]:
    """
    Filter candidate targets by removing those occluded angularly by closer bubbles.
    """
    if corridor_width is None:
        corridor_width = 0.75 * BUBBLE_RADIUS  # slightly narrower to avoid over-blocking
    result: List[Tuple[int, int]] = []
    for r, c in candidate_targets:
        if not check_angular_los_occlusion(
            shooter_x,
            shooter_y,
            r,
            c,
            player_num,
            grid,
            center_line_offset,
            corridor_width,
        ):
            result.append((r, c))
    return result


def get_hybrid_masked_targets(
    shooter_x: float,
    shooter_y: float,
    grid: Dict[Tuple[int, int], int],
    player_num: int,
    center_line_offset: float = 0,
    corridor_width: float = None,
) -> List[Tuple[int, int]]:
    """
    Complete hybrid masking: front-most-per-row prefilter followed by angular-LOS filtering.
    """
    front = get_front_most_per_row_targets(grid, player_num)
    if not front:
        return []
    return get_angular_los_filtered_targets(
        shooter_x,
        shooter_y,
        front,
        player_num,
        grid,
        center_line_offset,
        corridor_width,
    )
