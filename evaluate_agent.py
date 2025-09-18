import os
import csv
import math
import time
import random
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from train_target_bubbles_ai import TargetBubbleShooterEnv
from bubbles_target_dqn import TargetDQNAgent
from bubble_geometry import GRID_ROWS, GRID_COLS


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def neighbors_positions(row: int, col: int):
    if row % 2 == 0:
        return [
            (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
            (row - 1, col - 1), (row + 1, col - 1),
        ]
    else:
        return [
            (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
            (row - 1, col + 1), (row + 1, col + 1),
        ]


def count_neighbors_same_color(grid: dict, row: int, col: int, color: int) -> int:
    cnt = 0
    for nr, nc in neighbors_positions(row, col):
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            if (nr, nc) in grid and grid[(nr, nc)] == color:
                cnt += 1
    return cnt


def pop_count_if_place(grid: dict, row: int, col: int, color: int) -> int:
    # Simulate place
    if (row, col) in grid:
        return 0
    temp = dict(grid)
    temp[(row, col)] = color
    # BFS same-color component
    visited = set()
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for nr, nc in neighbors_positions(r, c):
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                if (nr, nc) in temp and temp[(nr, nc)] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
    return len(visited)


def find_popable_targets(grid: dict, color: int) -> list:
    out = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r, c) in grid:
                continue
            pops = pop_count_if_place(grid, r, c, color)
            if pops >= 3:
                out.append((r, c))
    return out


def evaluate_policy(agent: TargetDQNAgent, num_steps: int, output_dir: str, tag: str, use_random: bool = False,
                    seed: int = 42, reset_every: int = None):
    random.seed(seed)
    np.random.seed(seed)

    env = TargetBubbleShooterEnv(debug_render=False)
    state = env.reset()

    csv_path = os.path.join(output_dir, f"eval_log_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "episode", "had_pop_option", "chose_pop_option", "popped_count", "fell_count",
            "match_neighbors", "classification", "reward", "lost",
        ])

        total_steps = 0
        episode = 0
        # Aggregates for plotting windows
        window = 1000
        window_counts = []  # list of dicts
        current_w = {"good_pop": 0, "good_match": 0, "bad_missed_pop": 0, "bad_nomatch": 0, "steps": 0}

        good_total = 0
        overall_total = 0

        while total_steps < num_steps:
            if env.done:
                state = env.reset()
                episode += 1

            # Optional periodic reset independent of done
            if reset_every is not None and reset_every > 0 and (total_steps % reset_every == 0) and total_steps > 0:
                state = env.reset()
                episode += 1

            # Select action
            reachable = env.get_reachable_empty_targets(2)
            if not reachable:
                # let env handle no-op step
                action_idx = random.randrange(GRID_ROWS * GRID_COLS)
            else:
                if use_random:
                    r, c = random.choice(reachable)
                    action_idx = r * GRID_COLS + c
                else:
                    action_idx = agent.select_action(state, reachable, training_mode=False)

            # Build grid and current color for popable targets, constrained to reachable only
            grid = dict(env.grid_player2)
            current_color = env.current_bubble_color.get(2, 0)
            popable_pre_all = find_popable_targets(grid, current_color)
            reachable_set = set(reachable)
            popable_pre = [rc for rc in popable_pre_all if rc in reachable_set]
            popable_pre_set = set(popable_pre)

            prev_score = env.scores.get(2, 0)
            prev_grid = dict(env.grid_player2)

            next_state, reward, done, lost = env.step(2, action_idx)

            # Derive landing cell and stats
            new_grid = dict(env.grid_player2)
            popped_count = env.scores.get(2, 0) - prev_score
            fell_count = max(0, (len(prev_grid) + (1 if True else 0)) - len(new_grid) - popped_count)  # conservative

            # Landing cell: find new cell in new_grid not in prev_grid
            landing_rc = None
            for k in new_grid.keys():
                if k not in prev_grid:
                    landing_rc = k
                    break

            match_neighbors = 0
            chose_pop_option = False
            classification = "bad_nomatch"

            if landing_rc is not None:
                lr, lc = landing_rc
                match_neighbors = count_neighbors_same_color(new_grid, lr, lc, new_grid[landing_rc])
                chose_pop_option = (landing_rc in popable_pre_set)

            had_pop_option = len(popable_pre) > 0

            if popped_count > 0:
                classification = "good_pop"
                good_total += 1
            elif match_neighbors > 0 and not had_pop_option:
                classification = "good_match"
                good_total += 1
            elif match_neighbors > 0 and had_pop_option and not chose_pop_option:
                classification = "bad_missed_pop"
            else:
                classification = "bad_nomatch"

            overall_total += 1

            writer.writerow([
                total_steps + 1, episode, int(had_pop_option), int(chose_pop_option), popped_count, fell_count,
                match_neighbors, classification, float(reward), int(bool(lost)),
            ])

            # window agg
            current_w["steps"] += 1
            current_w[classification] += 1
            if current_w["steps"] >= window:
                window_counts.append(current_w)
                current_w = {"good_pop": 0, "good_match": 0, "bad_missed_pop": 0, "bad_nomatch": 0, "steps": 0}

            state = next_state
            total_steps += 1

        # Flush last partial window
        if current_w["steps"] > 0:
            window_counts.append(current_w)

    # Plot stacked bars of classification rates per window
    steps_axis = [i + 1 for i in range(len(window_counts))]
    gp = [w["good_pop"] for w in window_counts]
    gm = [w["good_match"] for w in window_counts]
    bmp = [w["bad_missed_pop"] for w in window_counts]
    bnm = [w["bad_nomatch"] for w in window_counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(steps_axis, gp, label="good_pop")
    ax.bar(steps_axis, gm, bottom=gp, label="good_match")
    ax.bar(steps_axis, bmp, bottom=[gp[i] + gm[i] for i in range(len(gp))], label="bad_missed_pop")
    ax.bar(steps_axis, bnm, bottom=[gp[i] + gm[i] + bmp[i] for i in range(len(gp))], label="bad_nomatch")
    ax.set_xlabel("Window (x1000 steps)")
    ax.set_ylabel("Count per window")
    ax.set_title(f"Action classification per 1000 steps ({tag})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"classification_stacked_{tag}.png"), dpi=200)
    plt.close(fig)

    # Good ratio
    ratio = (good_total / overall_total) if overall_total > 0 else 0.0
    with open(os.path.join(output_dir, f"summary_{tag}.txt"), "w") as sf:
        sf.write(f"Total steps: {overall_total}\n")
        sf.write(f"Good actions: {good_total}\n")
        sf.write(f"Good/Overall ratio: {ratio:.4f}\n")

    # Single overall bar split (stacked) across all steps
    overall_counts = {
        "good_pop": sum(w["good_pop"] for w in window_counts),
        "good_match": sum(w["good_match"] for w in window_counts),
        "bad_missed_pop": sum(w["bad_missed_pop"] for w in window_counts),
        "bad_nomatch": sum(w["bad_nomatch"] for w in window_counts),
    }
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    base = 0
    colors = {
        "good_pop": "tab:green",
        "good_match": "tab:blue",
        "bad_missed_pop": "tab:orange",
        "bad_nomatch": "tab:red",
    }
    for key in ["good_pop", "good_match", "bad_missed_pop", "bad_nomatch"]:
        ax2.bar([1], [overall_counts[key]], bottom=[base], color=colors[key], label=key)
        base += overall_counts[key]
    ax2.set_xticks([1])
    ax2.set_xticklabels(["All steps"])
    ax2.set_ylabel("Count")
    ax2.set_title(f"Overall action composition ({tag})")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"classification_singlebar_{tag}.png"), dpi=200)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="target_bubbles_dqn_model.pth")
    # Default to a quick 1k-step evaluation; no periodic resets by default
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--reset_every", type=int, default=50, help="Reset board every N steps (default: 50)")
    parser.add_argument("--out", type=str, default="eval_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ts_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(ts_dir)

    # Agent
    # Get STATE_SIZE and ACTION_SIZE from training setup
    from train_target_bubbles_ai import STATE_SIZE, TARGET_ACTION_SIZE
    agent = TargetDQNAgent(STATE_SIZE, TARGET_ACTION_SIZE, device='cpu')
    # Resolve model path robustly (relative to this file if not absolute)
    model_path = args.model
    if not os.path.isabs(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, model_path)
        if os.path.exists(candidate):
            model_path = candidate
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"Warning: failed to load model from {model_path}: {e}\nContinuing with uninitialized agent (random policy).")
    else:
        print(f"Warning: model file not found at {args.model} (resolved to: {model_path}). Continuing without loading.")
    agent.set_evaluation_mode()

    # Evaluate learned agent
    evaluate_policy(agent, args.steps, ts_dir, tag="learned", use_random=False, seed=args.seed, reset_every=args.reset_every)
    # Evaluate random baseline
    evaluate_policy(agent, args.steps, ts_dir, tag="random_baseline", use_random=True, seed=args.seed + 1, reset_every=args.reset_every)

    print(f"Evaluation complete. Results in: {ts_dir}")


if __name__ == "__main__":
    main()


