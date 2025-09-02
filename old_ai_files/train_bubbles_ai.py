import numpy as np
import torch
import random
from bubbles_dqn import DQNAgent
import os
import copy

# --- Game Environment Abstraction ---
# We'll use a minimal environment for training. In practice, you would refactor the game logic for full RL compatibility.

GRID_ROWS = 20
GRID_COLS = 35
BUBBLE_COLORS = 6
STATE_SIZE = GRID_ROWS * GRID_COLS + 1 + 1 + BUBBLE_COLORS + BUBBLE_COLORS  # grid + shooter_y + shooter_angle + current_bubble_color + next_bubble_color
DQN_ACTION_SIZE = 42  # Reduced from 50 (removed first 4 and last 4 angles)

# --- State/Action Conversion Utilities ---
def encode_grid(grid, ai_player_num):
    """
    Encodes the grid as a flat array:
    0 = empty, 1 = AI bubble, 2 = opponent bubble
    """
    arr = np.zeros(GRID_ROWS * GRID_COLS, dtype=np.float32)
    for (row, col, player), bubble in grid.items():
        idx = row * GRID_COLS + col
        if player == ai_player_num:
            arr[idx] = 1.0
        else:
            arr[idx] = 2.0
    return arr

def encode_state(grid, shooter_y, shooter_angle, bubble_color, next_bubble_color, ai_player_num):
    """
    Returns a flat state vector for the DQN agent.
    """
    grid_vec = encode_grid(grid, ai_player_num)
    shooter_y_norm = shooter_y / 800.0  # Normalize y
    shooter_angle_norm = (shooter_angle - 90) / 180.0  # Normalize angle to [0,1]
    color_onehot = np.zeros(BUBBLE_COLORS, dtype=np.float32)
    color_onehot[bubble_color] = 1.0
    next_color_onehot = np.zeros(BUBBLE_COLORS, dtype=np.float32)
    next_color_onehot[next_bubble_color] = 1.0
    return np.concatenate([grid_vec, [shooter_y_norm], [shooter_angle_norm], color_onehot, next_color_onehot])

def decode_action(action_idx):
    """
    Converts a discrete action index to an angle in degrees.
    Skips first 4 and last 4 angles: 90-126° and 234-270° are removed.
    Valid range: 127° to 233° (107° total range)
    """
    # Map action_idx (0-41) to angles 127-233
    # 127° + (action_idx * 106° / 41)
    return 127 + action_idx * (106 / (DQN_ACTION_SIZE - 1))

# --- Reward Function ---
def compute_reward(prev_score, new_score, done, lost):
    """
    +1 for popping bubbles, -1 for losing, 0 otherwise.
    """
    if lost:
        return -1.0
    elif new_score > prev_score:
        return 1.0
    elif done:
        return 0.0
    else:
        return -0.01  # Small step penalty to encourage faster play

# --- Training Loop ---
def train_dqn(num_episodes=100_000, save_path='bubbles_dqn_model.pth', num_envs=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent1 = DQNAgent(STATE_SIZE, DQN_ACTION_SIZE, device)
    agent2 = DQNAgent(STATE_SIZE, DQN_ACTION_SIZE, device)
    # Continue training from existing model if present
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path} for continued training...")
        agent1.load(save_path)
        agent2.load(save_path)
    
    print(f"Training with {num_envs} parallel environments for better learning quality...")
    
    for episode in range(num_episodes):
        # Create multiple environments for parallel experience collection
        envs = [BubbleShooterEnv() for _ in range(num_envs)]
        states1 = [env.get_state(1) for env in envs]
        states2 = [env.get_state(2) for env in envs]
        
        # Track total rewards across all environments
        total_reward1 = 0
        total_reward2 = 0
        games_completed = 0
        
        # Add initial progress reporting
        if episode < 5:
            print(f"Starting episode {episode}...")
        
        # Run all environments until completion (one player loses)
        step_count = 0
        while not all(env.done for env in envs):
            step_count += 1
            
            # Add debugging for first few episodes
            if episode < 3 and step_count % 10 == 0:
                done_count = sum(1 for env in envs if env.done)
                print(f"  Episode {episode}, Step {step_count}: {done_count}/{num_envs} environments done")
            
            # Safety check to prevent infinite loops
            if step_count > 400:
                print(f"  Episode {episode} taking too long ({step_count} steps), forcing completion")
                for env in envs:
                    if not env.done:
                        env.done = True
                        env.lost[1] = True  # Force player 1 to lose as default
                break
            
            # Agent 1 actions for all environments
            actions1 = [agent1.select_action(state) for state in states1]
            
            # Step all environments for agent 1
            results1 = []
            for i, (env, action) in enumerate(zip(envs, actions1)):
                if not env.done:
                    result = env.step(1, action)
                    results1.append(result)
                    agent1.store_transition(states1[i], action, result[1], result[0], result[2])
                    total_reward1 += result[1]
                    if result[2]:  # If done
                        games_completed += 1
                else:
                    results1.append((states1[i], 0, True, False))
            
            # Update states for agent 1
            states1 = [result[0] for result in results1]
            
            # Agent 2 actions for all environments (if not done)
            actions2 = [agent2.select_action(state) for state in states2]
            
            # Step all environments for agent 2
            results2 = []
            for i, (env, action) in enumerate(zip(envs, actions2)):
                if not env.done:
                    result = env.step(2, action)
                    results2.append(result)
                    agent2.store_transition(states2[i], action, result[1], result[0], result[2])
                    total_reward2 += result[1]
                    if result[2]:  # If done
                        games_completed += 1
                else:
                    results2.append((states2[i], 0, True, False))
            
            # Update states for agent 2
            states2 = [result[0] for result in results2]
        
        # Update agents with larger, more diverse experience batches
        for _ in range(num_envs):  # Multiple updates to process all experiences
            agent1.update()
            agent2.update()
        
        # Update target networks periodically
        if episode % 100 == 0:
            agent1.update_target()
            agent2.update_target()
        
        # Save model and print progress
        if episode % 100 == 0:
            avg_reward1 = total_reward1 / num_envs if num_envs > 0 else 0
            avg_reward2 = total_reward2 / num_envs if num_envs > 0 else 0
            completion_rate = games_completed / num_envs * 100
            
            # Count wins for each agent
            agent1_wins = sum(1 for env in envs if env.lost[2])  # Agent 1 wins if Agent 2 lost
            agent2_wins = sum(1 for env in envs if env.lost[1])  # Agent 2 wins if Agent 1 lost
            
            # Calculate actual step counts and ending types
            total_steps = sum(env.step_count for env in envs)
            avg_steps = total_steps / num_envs if num_envs > 0 else 0
            natural_endings = sum(getattr(env, 'natural_endings', 0) for env in envs)
            artificial_endings = sum(getattr(env, 'artificial_endings', 0) for env in envs)
            
            print(f"Episode {episode}: Agent1 avg reward {avg_reward1:.2f}, Agent2 avg reward {avg_reward2:.2f}, Games completed: {games_completed}/{num_envs} ({completion_rate:.1f}%), Agent1 wins: {agent1_wins}, Agent2 wins: {agent2_wins}, Avg steps: {avg_steps:.1f}, Natural endings: {natural_endings}, Artificial endings: {artificial_endings}")
            agent1.save(save_path)
        
        # Add more frequent progress reporting for debugging
        if episode % 10 == 0:
            avg_reward1 = total_reward1 / num_envs if num_envs > 0 else 0
            avg_reward2 = total_reward2 / num_envs if num_envs > 0 else 0
            completion_rate = games_completed / num_envs * 100
            
            # Count wins for each agent
            agent1_wins = sum(1 for env in envs if env.lost[2])  # Agent 1 wins if Agent 2 lost
            agent2_wins = sum(1 for env in envs if env.lost[1])  # Agent 2 wins if Agent 1 lost
            
            # Calculate actual step counts and ending types
            total_steps = sum(env.step_count for env in envs)
            avg_steps = total_steps / num_envs if num_envs > 0 else 0
            natural_endings = sum(getattr(env, 'natural_endings', 0) for env in envs)
            artificial_endings = sum(getattr(env, 'artificial_endings', 0) for env in envs)
            
            print(f"Episode {episode}: Agent1 avg reward {avg_reward1:.2f}, Agent2 avg reward {avg_reward2:.2f}, Games completed: {games_completed}/{num_envs} ({completion_rate:.1f}%), Agent1 wins: {agent1_wins}, Agent2 wins: {agent2_wins}, Avg steps: {avg_steps:.1f}, Natural endings: {natural_endings}, Artificial endings: {artificial_endings}")
    
    print("Training complete. Model saved to", save_path)

# --- Environment Stub (to be implemented with real game logic) ---
class BubbleShooterEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize separate grids for each player
        self.grid_player1 = {}  # (row, col): color
        self.grid_player2 = {}  # (row, col): color
        self.center_line_offset = 0  # Current offset from center
        self.target_center_line_offset = 0
        self.last_shift_offset = 0  # Track the last offset when grids were shifted
        
        # Fill top few rows with random bubbles for both players
        # Start from column 3 to avoid immediate losing condition
        initial_cols = 7
        for row in range(5):
            for col in range(3, 3 + initial_cols):  # Start from column 3
                if random.random() < 0.7:  # 70% chance for more realistic density
                    color = random.randint(1, BUBBLE_COLORS)
                    self.grid_player1[(row, col)] = color
                    self.grid_player2[(row, col)] = color
        
        self.shooter_y = {1: 400, 2: 400}
        self.shooter_angle = {1: 180, 2: 180}
        self.current_bubble_color = {1: random.randint(0, BUBBLE_COLORS-1), 2: random.randint(0, BUBBLE_COLORS-1)}
        self.next_bubble_color = {1: random.randint(0, BUBBLE_COLORS-1), 2: random.randint(0, BUBBLE_COLORS-1)}
        self.scores = {1: 0, 2: 0}
        self.done = False
        self.lost = {1: False, 2: False}
        return self.get_state(1), self.get_state(2)

    def get_state(self, player_num):
        # Encode the player's own grid as a flat array
        flat_grid = np.zeros(GRID_ROWS * GRID_COLS, dtype=np.float32)
        
        # Get the player's grid
        player_grid = self.grid_player1 if player_num == 1 else self.grid_player2
        
        # Fill in the grid values
        for (row, col), color in player_grid.items():
            if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
                idx = row * GRID_COLS + col
                flat_grid[idx] = float(color)
        
        shooter_y_norm = self.shooter_y[player_num] / 800.0
        shooter_angle_norm = (self.shooter_angle[player_num] - 90) / 180.0
        color_onehot = np.zeros(BUBBLE_COLORS, dtype=np.float32)
        color_onehot[self.current_bubble_color[player_num]] = 1.0
        next_color_onehot = np.zeros(BUBBLE_COLORS, dtype=np.float32)
        next_color_onehot[self.next_bubble_color[player_num]] = 1.0
        
        return np.concatenate([flat_grid, [shooter_y_norm], [shooter_angle_norm], color_onehot, next_color_onehot])

    def step(self, player_num, action_idx):
        if self.done:
            return self.get_state(player_num), 0.0, True, self.lost[player_num]
        
        angle = decode_action(action_idx)
        self.shooter_angle[player_num] = angle
        prev_score = self.scores[player_num]
        
        # Simulate shooting with honeycomb pattern
        col = self._angle_to_col(angle, player_num)
        row = self._find_landing_row(col, player_num)
        color = self.current_bubble_color[player_num] + 1
        
        if row is not None:
            player_grid = self.grid_player1 if player_num == 1 else self.grid_player2
            player_grid[(row, col)] = color
            popped = self._pop_bubbles(row, col, color, player_num)
            self.scores[player_num] += popped
            
            # Push center line away from the player who popped bubbles
            if popped > 0:
                push_amount = popped * 6  # 6 pixels per bubble popped
                if player_num == 1:
                    self.target_center_line_offset += push_amount
                else:
                    self.target_center_line_offset -= push_amount
                # Clamp to reasonable bounds
                self.target_center_line_offset = max(-200, min(200, self.target_center_line_offset))
                
                # Apply the grid shift immediately
                self._shift_grids()
                

        else:
            popped = 0
        
        # Check for losing condition (bubbles too close to player's edge)
        if self._check_lose_condition(player_num):
            self.done = True
            self.lost[player_num] = True
            # Debug: Track natural game endings
            if not hasattr(self, 'natural_endings'):
                self.natural_endings = 0
            self.natural_endings += 1
        
        # Add maximum step limit to prevent infinite games
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        # Force game end after 200 steps to prevent infinite loops
        if self.step_count >= 200:
            self.done = True
            # Determine winner based on scores or random if tied
            if self.scores[1] > self.scores[2]:
                self.lost[2] = True
            elif self.scores[2] > self.scores[1]:
                self.lost[1] = True
            else:
                # Random winner if tied
                winner = random.choice([1, 2])
                self.lost[3 - winner] = True
            # Debug: Track artificial game endings
            if not hasattr(self, 'artificial_endings'):
                self.artificial_endings = 0
            self.artificial_endings += 1
        
        # Prepare next bubble
        self.current_bubble_color[player_num] = self.next_bubble_color[player_num]
        self.next_bubble_color[player_num] = random.randint(0, BUBBLE_COLORS-1)
        
        # Calculate reward
        reward = popped - 0.01  # +1 per bubble popped, small step penalty
        if self.lost[player_num]:
            reward -= 1.0
        
        return self.get_state(player_num), reward, self.done, self.lost[player_num]

    def _shift_grids(self):
        """Shift both grids based on the center line offset change"""
        # Calculate how much to shift (in grid columns)
        # The target_center_line_offset represents the total desired offset
        # We need to shift by the difference from the current position
        shift_amount = int(self.target_center_line_offset / 6)  # 6 pixels per bubble = 1 grid column shift
        
        if shift_amount != 0:  # Only shift if there's actually a change
            # Shift Player 1's grid (left player)
            new_grid1 = {}
            for (row, col), color in self.grid_player1.items():
                new_col = col + shift_amount
                if 0 <= new_col < GRID_COLS:  # Keep bubbles within grid bounds
                    new_grid1[(row, new_col)] = color
            self.grid_player1 = new_grid1
            
            # Shift Player 2's grid (right player) 
            new_grid2 = {}
            for (row, col), color in self.grid_player2.items():
                new_col = col + shift_amount
                if 0 <= new_col < GRID_COLS:  # Keep bubbles within grid bounds
                    new_grid2[(row, new_col)] = color
            self.grid_player2 = new_grid2
            
            # Reset the target offset after applying the shift
            self.target_center_line_offset = 0

    def _angle_to_col(self, angle, player_num):
        # Map angle to column considering player position and honeycomb pattern
        if player_num == 1:
            # Left player: angle 90-270 maps to columns 0-34
            rel = (angle - 90) / 180.0
            col = int(rel * (GRID_COLS - 1))
        else:
            # Right player: angle 90-270 maps to columns 0-34 (from right side)
            rel = (angle - 90) / 180.0
            col = int(rel * (GRID_COLS - 1))
        return min(max(col, 0), GRID_COLS - 1)

    def _find_landing_row(self, col, player_num):
        # Find landing position considering honeycomb pattern
        player_grid = self.grid_player1 if player_num == 1 else self.grid_player2
        
        # Start from bottom and work up
        for row in range(GRID_ROWS-1, -1, -1):
            if (row, col) not in player_grid:
                return row
        return None

    def _pop_bubbles(self, row, col, color, player_num):
        # BFS to find connected bubbles in honeycomb pattern
        player_grid = self.grid_player1 if player_num == 1 else self.grid_player2
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
            return len(visited)
        return 0

    def _check_lose_condition(self, player_num):
        # Check if any bubble is too close to the player's edge
        player_grid = self.grid_player1 if player_num == 1 else self.grid_player2
        
        if player_num == 1:
            # Check if any bubble is too far left (close to player 1's edge)
            for (row, col) in player_grid:
                if col <= 1:  # Less strict: only lose if bubble is in column 0 or 1
                    return True
        else:
            # Check if any bubble is too far right (close to player 2's edge)
            for (row, col) in player_grid:
                if col >= GRID_COLS - 2:  # Less strict: only lose if bubble is in last 2 columns
                    return True
        return False

if __name__ == '__main__':
    # You can adjust the number of parallel environments here
    # More environments = better diversity but more memory usage
    train_dqn(num_episodes=100_000, num_envs=16)  # 16 parallel environments for faster testing 