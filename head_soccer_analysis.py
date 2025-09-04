import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Environment Constants
WIDTH = 800
HEIGHT = 850
GRAVITY = 0.5

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DQN Training Viewer")
clock = pygame.time.Clock()

# Visualize the training environment for tracking AI's behavior
def draw_env(env):
    screen.fill((0, 100, 0))  # Green background

    # Draw goals
    pygame.draw.rect(screen, (200, 200, 200), (0, HEIGHT - 20 - 200, 80, 200))  # Left goal
    pygame.draw.rect(screen, (200, 200, 200), (WIDTH - 80, HEIGHT - 20 - 200, 80, 200))  # Right goal

    # Draw players (AI = blue, Opponent = red)
    pygame.draw.rect(screen, (0, 0, 255), (env.ai_x - 25, env.ai_y - 90, 50, 90))
    pygame.draw.rect(screen, (255, 0, 0), (env.opp_x - 25, env.opp_y - 90, 50, 90))

    # Draw ball
    pygame.draw.circle(screen, (255, 255, 255), (int(env.ball_x), int(env.ball_y)), 15)

    pygame.display.flip()


ACTIONS = ['left', 'right', 'jump', 'kick', 'stay']
ACTION_SIZE = len(ACTIONS)


# The game environment
class HeadSoccerEnv:
    def __init__(self):
        # Initialize state attributes
        self.ai_x = self.ai_y = self.ai_vy = None
        self.ai_on_ground = None
        self.opp_x = self.opp_y = self.opp_vy = None
        self.opp_on_ground = None
        self.ball_x = self.ball_y = self.ball_vx = self.ball_vy = None
        self.done = None
        self.ai_kick_timer = None
        self.opp_kick_timer = None
        self.reset()

    def reset(self):
        # Players
        self.ai_x = 650
        self.ai_y = HEIGHT - 20
        self.ai_vy = 0
        self.ai_on_ground = True
        self.ai_kick_timer = 0

        self.opp_x = 150
        self.opp_y = HEIGHT - 20
        self.opp_vy = 0
        self.opp_on_ground = True
        self.opp_kick_timer = 0

        # Ball
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.ball_vx = random.choice([-4, 4])
        self.ball_vy = -5

        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array([
            # AI Player position
            (self.ai_x - 40) / (WIDTH - 80),
            self.ai_y / HEIGHT,

            # Ball position and velocity
            (self.ball_x - 40) / (WIDTH - 80),
            self.ball_y / HEIGHT,
            self.ball_vx / 10,  # horizontal velocity scaled
            self.ball_vy / 10,  # vertical velocity scaled

            # Opponent position
            (self.opp_x - 40) / (WIDTH - 80),
            self.opp_y / HEIGHT,  # Vertical position

            # Ball relative to AI
            (self.ball_x - self.ai_x) / (WIDTH - 80),  # Ahead / behind AI player
            (self.ai_y - self.ball_y) / HEIGHT,  # Above / Under AI player
            (WIDTH - 80 - self.ball_x) / (WIDTH - 80),  # distance to AI's goal

            # Ball movement flags
            1.0 if self.ball_vx > 0 else 0.0,  # is ball moving toward AI goal
            1.0 if self.ball_x > WIDTH - 200 and self.ball_vx > 0 else 0.0,  # danger zone

            # AI status flags
            1.0 if self.ai_on_ground else 0.0,  # AI on ground

            # Opponent relative info (optional but helpful)
            (self.opp_x - self.ai_x) / (WIDTH - 80),  # horizontal distance to opponent
            (self.opp_y - self.ai_y) / HEIGHT,  # vertical distance to opponent
        ], dtype=np.float32)

    def get_opp_state(self):
        """Observation from the OPPONENT's perspective (mirror of get_state)."""
        return np.array([
            # Opponent (red) position, treated as "self"
            (self.opp_x - 40) / (WIDTH - 80),
            self.opp_y / HEIGHT,

            # Ball info
            (self.ball_x - 40) / (WIDTH - 80),
            self.ball_y / HEIGHT,
            self.ball_vx / 10,
            self.ball_vy / 10,

            # AI position (treated as 'opponent' from red's view)
            (self.ai_x - 40) / (WIDTH - 80),
            self.ai_y / HEIGHT,

            # Ball relative to opponent
            (self.ball_x - self.opp_x) / (WIDTH - 80),
            (self.opp_y - self.ball_y) / HEIGHT,
            (self.ball_x - 80) / (WIDTH - 80),  # distance to red's goal (left)

            # Ball movement flags (toward left goal is "positive danger" for red)
            1.0 if self.ball_vx < 0 else 0.0,
            1.0 if self.ball_x < 200 and self.ball_vx < 0 else 0.0,

            # Opponent status flags
            1.0 if self.opp_on_ground else 0.0,

            # AI relative info
            (self.ai_x - self.opp_x) / (WIDTH - 80),
            (self.ai_y - self.opp_y) / HEIGHT,
        ], dtype=np.float32)

    def apply_action(self, player, action):
        """Apply an action to 'ai' or 'opp' player entities."""
        if player == "ai":
            x, y, vy, on_ground, kick_timer = (
                self.ai_x, self.ai_y, self.ai_vy, self.ai_on_ground, self.ai_kick_timer
            )
        else:
            x, y, vy, on_ground, kick_timer = (
                self.opp_x, self.opp_y, self.opp_vy, self.opp_on_ground, self.opp_kick_timer
            )

        if action == 'left':
            x -= 5
        elif action == 'right':
            x += 5
        elif action == 'jump':
            if on_ground:
                vy = -12
                on_ground = False
        elif action == 'kick':
            # "kick" arms the jump/impact power for a few frames if close to ball
            if kick_timer == 0:
                if abs(x - self.ball_x) < 80 and abs(y - self.ball_y) < 100:
                    kick_timer = 5
        elif action == 'stay':
            pass

        # enforce movement limits
        x = np.clip(x, 40, WIDTH - 40)

        # write back
        if player == "ai":
            self.ai_x, self.ai_y, self.ai_vy, self.ai_on_ground, self.ai_kick_timer = x, y, vy, on_ground, kick_timer
        else:
            self.opp_x, self.opp_y, self.opp_vy, self.opp_on_ground, self.opp_kick_timer = x, y, vy, on_ground, kick_timer

    def rule_based_opponent(self):
        """Fallback: simple ball-chasing bot (your original logic)."""
        if self.opp_x < self.ball_x:
            self.opp_x += 5
        else:
            self.opp_x -= 5
        if self.opp_on_ground and abs(self.opp_x - self.ball_x) < 50 and self.ball_y < self.opp_y and self.ball_vy > 0:
            self.opp_vy = -12
            self.opp_on_ground = False
        if abs(self.opp_x - self.ball_x) < 40 and abs(self.opp_y - self.ball_y) < 50:
            self.opp_kick_timer = 5

    def step(self, action_idx, opponent_agent=None):
        prev_ai_x = self.ai_x
        prev_ball_x = self.ball_x
        action = ACTIONS[action_idx]
        reward = 0

        # --- Apply AI action ---
        self.apply_action("ai", action)

        # --- Opponent policy ---
        if opponent_agent is not None:
            # Frozen self-play opponent (greedy)
            opp_state = self.get_opp_state()
            opp_action_idx = opponent_agent.select_action(opp_state, epsilon=0.0)
            self.apply_action("opp", ACTIONS[opp_action_idx])
        else:
            # Fallback: rule-based bot
            self.rule_based_opponent()

        # --- Gravity & player physics ---
        self.ai_vy += GRAVITY
        self.ai_y += self.ai_vy
        if self.ai_y >= HEIGHT - 20:
            self.ai_y = HEIGHT - 20
            self.ai_vy = 0
            self.ai_on_ground = True
        if self.ai_kick_timer > 0:
            self.ai_kick_timer -= 1

        self.opp_vy += GRAVITY
        self.opp_y += self.opp_vy
        if self.opp_y >= HEIGHT - 20:
            self.opp_y = HEIGHT - 20
            self.opp_vy = 0
            self.opp_on_ground = True
        if self.opp_kick_timer > 0:
            self.opp_kick_timer -= 1

        self.ai_x = np.clip(self.ai_x, 40, WIDTH - 40)
        self.opp_x = np.clip(self.opp_x, 40, WIDTH - 40)

        # --- Ball physics ---
        self.ball_vy += GRAVITY
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        if self.ball_x - 15 < 0 or self.ball_x + 15 > WIDTH:
            self.ball_vx *= -1
        if self.ball_y - 15 < 0:
            self.ball_y = 15
            self.ball_vy *= -0.85
        if self.ball_y >= HEIGHT - 20:
            self.ball_y = HEIGHT - 20
            self.ball_vy *= -0.85

        # Player-player collision
        ai_rect = pygame.Rect(self.ai_x - 25, self.ai_y - 90, 50, 90)
        opp_rect = pygame.Rect(self.opp_x - 25, self.opp_y - 90, 50, 90)

        if ai_rect.colliderect(opp_rect):
            if self.ai_x < self.opp_x:
                overlap = (ai_rect.right - opp_rect.left) // 2
                self.ai_x -= overlap
                self.opp_x += overlap
            else:
                overlap = (opp_rect.right - ai_rect.left) // 2
                self.ai_x += overlap
                self.opp_x -= overlap

        # Crossbar collisions
        goal_width = 80
        goal_height = 200
        goal_y = HEIGHT - 20 - goal_height
        crossbar_height = 10
        if (
            self.ball_x - 15 < goal_width and self.ball_y - 15 < goal_y + crossbar_height and self.ball_y + 15 > goal_y
        ) or (
            self.ball_x + 15 > WIDTH - goal_width and self.ball_y - 15 < goal_y + crossbar_height and self.ball_y + 15 > goal_y
        ):
            self.ball_y = goal_y - 15
            self.ball_vy *= -0.85

        # --- Goal detection ---
        left_goal_rect = pygame.Rect(0, goal_y, goal_width, goal_height)
        right_goal_rect = pygame.Rect(WIDTH - goal_width, goal_y, goal_width, goal_height)
        ball_rect = pygame.Rect(self.ball_x - 15, self.ball_y - 15, 30, 30)

        if ball_rect.colliderect(left_goal_rect):
            reward += 5  # AI scored
            return self.reset(), reward, True
        elif ball_rect.colliderect(right_goal_rect):
            reward -= 5  # AI conceded
            return self.reset(), reward, True

        # --- Defensive encouragement: step out of goal when ball is coming in ---
        # Penalty box area in front of AI's goal
        penalty_box_rect = pygame.Rect(WIDTH // 2, goal_height - 20, WIDTH // 2, goal_height - 20)

        if penalty_box_rect.colliderect(ball_rect) and self.ball_vx > 0:
            if self.ai_x > self.ball_x:
                # AI is between ball and his goal
                reward += 0.05
            else:
                reward -= 0.05

        # --- Player-ball collisions ---
        ai_rect = pygame.Rect(self.ai_x - 25, self.ai_y - 90, 50, 90)
        opp_rect = pygame.Rect(self.opp_x - 25, self.opp_y - 90, 50, 90)
        if ball_rect.colliderect(ai_rect):
            if self.ball_x < self.ai_x:  # toward enemy
                reward += 0.3
                self.ball_vx = -abs(self.ball_vx)
            else:  # toward own goal
                reward -= 0.3
                self.ball_vx = abs(self.ball_vx)
            self.ball_vy = -12 if self.ai_kick_timer > 0 else -9

        if ball_rect.colliderect(opp_rect):
            self.ball_vy = -12 if self.opp_kick_timer > 0 else -9

        # Encourage jumping when ball is above
        if self.ball_y < self.ai_y - 50 and abs(self.ball_x - self.ai_x) < 60:
            if action == ACTIONS.index('jump'):
                reward += 0.3

        # Encourage kicking when ball is close
        if (self.ball_x - self.ai_x) < 50 and (self.ball_y - self.ai_y) < 50:
            if action == ACTIONS.index('kick'):
                reward += 0.3

        # --- Small incentive for closing distance to ball ---
        if abs(self.ai_x - self.ball_x) < abs(prev_ai_x - prev_ball_x):
            reward += 0.1
        else:
            reward -= 0.1

        return self.get_state(), reward, False


# Q-Network
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


# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity

    def push(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


# Minimal agent wrapper for opponent policy
class FrozenAgent:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.net(s)
            return int(q.argmax().item())


# Training DQN (with self-play vs frozen past self)
def train_dqn(episodes=1000, render=False, opponent_refresh=500):
    env = HeadSoccerEnv()

    device = torch.device("cpu")

    net = QNet().to(device)
    target_net = QNet().to(device)
    target_net.load_state_dict(net.state_dict())

    opponent_net = QNet().to(device)
    opponent_net.load_state_dict(net.state_dict())
    opponent_agent = FrozenAgent(opponent_net, device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    buffer = ReplayBuffer()
    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    epsilon_decay = 0.9975
    epsilon_min = 0.01
    sync_target_every = 1000

    # --- Tracking lists ---
    rewards_per_ep = []
    losses_per_ep = []
    qvals_per_ep = []
    winrates_per_ep = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        ep_losses = []
        ep_qvals = []
        ai_goals, opp_goals = 0, 0

        for t in range(300):
            # ε-greedy action
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SIZE - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = net(state_tensor)
                    action = int(q_values.argmax().item())
                    ep_qvals.append(q_values.max().item())

            next_state, reward, done = env.step(action, opponent_agent)

            # track goals for win rate
            if reward == 5:
                ai_goals += 1
            elif reward == -5:
                opp_goals += 1

            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                draw_env(env)
                clock.tick(60)

            buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Learn
            if len(buffer.buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                q_vals = net(states).gather(1, actions)
                with torch.no_grad():
                    next_q_vals = target_net(next_states).max(1)[0].unsqueeze(1)
                    targets = rewards + gamma * next_q_vals * (1.0 - dones)

                loss = nn.MSELoss()(q_vals, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                ep_losses.append(loss.item())

            if done:
                break

        # target network update
        if episode % sync_target_every == 0:
            target_net.load_state_dict(net.state_dict())

        # opponent refresh
        if episode % opponent_refresh == 0 and episode > 0:
            opponent_net.load_state_dict(net.state_dict())
            print(f"[Self-Play] Opponent updated at episode {episode}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # --- Logging per episode ---
        rewards_per_ep.append(total_reward)
        losses_per_ep.append(np.mean(ep_losses) if ep_losses else 0)
        qvals_per_ep.append(np.mean(ep_qvals) if ep_qvals else 0)
        winrates_per_ep.append(1 if ai_goals > opp_goals else 0)

        if episode % 100 == 0 and episode > 0:
            avg_last_100 = np.mean(rewards_per_ep[-100:])
            print(f"Episode {episode}: avg_reward(last 100) = {avg_last_100:.2f}")

    # --- Plotting with multiple y-axes ---
    def smooth(data, w=100):
        return np.convolve(data, np.ones(w)/w, mode="valid")

    rewards_s = smooth(rewards_per_ep)
    losses_s = smooth(losses_per_ep)
    qvals_s = smooth(qvals_per_ep)
    wins_s = smooth(winrates_per_ep)

    fig, ax1 = plt.subplots(figsize=(12,6))

    # Rewards
    ax1.plot(rewards_s, label="Reward (avg)", color="blue")
    ax1.plot(wins_s, label="Win Rate", color="green", linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward / Win Rate", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Loss
    ax2 = ax1.twinx()
    ax2.plot(losses_s, label="Loss", color="red", alpha=0.6)
    ax2.set_ylabel("Loss", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Q-values
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))  # shift right
    ax3.plot(qvals_s, label="Q-values", color="purple", alpha=0.6)
    ax3.set_ylabel("Q-values", color="purple")
    ax3.tick_params(axis="y", labelcolor="purple")

    fig.suptitle("DQN Training Metrics (100-episode moving average)", fontsize=14)

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.show()

    # save model
    torch.save(net.state_dict(), "analysis_nn_model.pth")


    # Save the trained model
    torch.save(net.state_dict(), "analysis_nn_model.pth")


# Run training
if __name__ == "__main__":
    train_dqn(episodes=5000, render=False, opponent_refresh=200)
#