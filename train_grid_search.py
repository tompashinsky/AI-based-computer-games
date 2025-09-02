import os
import csv
import time
import math
import threading
import random
import numpy as np
import torch
from typing import List, Dict, Any

from bubbles_target_dqn import TargetDQNAgent
from train_target_bubbles_ai import TargetBubbleShooterEnv, STATE_SIZE, TARGET_ACTION_SIZE


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_hidden_layers(depth: int, width: int) -> List[int]:
    return [width] * depth


def evaluate_agent(agent: TargetDQNAgent, steps: int = 100) -> int:
    # Exploitation-only evaluation using bubbles popped as the score
    env = TargetBubbleShooterEnv(debug_render=False)
    state = env.reset()
    agent.set_evaluation_mode()
    total_popped = 0
    last_score = env.scores[2]
    for _ in range(steps):
        # Mask to front-most bubbles per row
        valid_targets = env.get_front_bubble_targets(2)
        action = agent.select_action(state, valid_targets, training_mode=False)
        next_state, reward, done, lost = env.step(2, action)
        # Count popped bubbles directly from score delta
        popped_now = env.scores[2] - last_score
        if popped_now > 0:
            total_popped += popped_now
        last_score = env.scores[2]
        state = next_state
        if done:
            # Reset and continue until we reach desired steps
            state = env.reset()
            last_score = env.scores[2]
    return total_popped


def train_one_trial(depth: int, width: int, combo_id: int, combo: Dict[str, Any], clip_norm: float, steps: int,
                    device: torch.device, results: List[Dict[str, Any]], thread_name: str):
    seed = hash((depth, width, combo_id, clip_norm)) % (2**32 - 1)
    set_seed(seed)

    hidden_layers = make_hidden_layers(depth, width)
    model_name = f"d{depth}_w{width}_c{combo_id}_clip{'on' if clip_norm else 'off'}.pth"
    save_path = os.path.join(os.getcwd(), model_name)

    overrides = dict(
        learning_rate=combo['lr'],
        gamma=combo['gamma'],
        batch_size=combo['batch_size'],
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=combo['eps_decay'],
        target_update_freq=combo['target_update'],
        grad_clip_norm=clip_norm,
        hidden_layers=hidden_layers,
    )

    # Build agent
    agent = TargetDQNAgent(STATE_SIZE, TARGET_ACTION_SIZE, device=device, **overrides)

    # Single env loop, streaming style for speed
    env = TargetBubbleShooterEnv(debug_render=False)
    state = env.reset()
    total_steps = 0
    WARMUP_STEPS = 2000
    warmup_done = False
    while total_steps < steps:
        if env.done:
            env.reset()
            state = env.get_state(2)
        valid_targets = env.get_front_bubble_targets(2)
        action = agent.select_action(state, valid_targets, training_mode=True)
        next_state, reward, done, lost = env.step(2, action)
        agent.store_transition(state, action, reward, next_state, done)
        if len(agent.replay_buffer) >= WARMUP_STEPS:
            if not warmup_done:
                print(f"[{thread_name}] Warmup complete at step {total_steps}. Starting updates.")
                warmup_done = True
            agent.update()
        if total_steps % agent.target_update_freq == 0:
            agent.update_target()
        state = next_state
        total_steps += 1
    agent.save(save_path)
    print(f"[{thread_name}] Trained d={depth}, w={width}, combo={combo_id}, clip={'on' if clip_norm else 'off'} -> saved {model_name}")

    # Evaluate
    eval_popped = evaluate_agent(agent, steps=100)
    print(f"[{thread_name}] Evaluated d={depth}, w={width}, combo={combo_id}, clip={'on' if clip_norm else 'off'}: bubbles_popped={eval_popped}")

    results.append(dict(
        depth=depth,
        hidden_width=width,
        combo_id=combo_id,
        lr=combo['lr'],
        gamma=combo['gamma'],
        batch_size=combo['batch_size'],
        eps_decay=combo['eps_decay'],
        target_update=combo['target_update'],
        clip_norm=clip_norm if clip_norm else None,
        eval_bubbles_popped=eval_popped,
        steps_trained=steps,
        seed=seed,
        model_path=model_name,
    ))


def run_thread(depth: int, widths: List[int], combos: List[Dict[str, Any]], steps: int, results_sink: List[Dict[str, Any]]):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    thread_name = f"depth-{depth}"
    total_trials = len(widths) * len(combos) * 2
    completed = 0
    for width in widths:
        for i, combo in enumerate(combos, start=1):
            for clip_norm in [None, 1.0]:
                train_one_trial(depth, width, i, combo, clip_norm, steps, device, results_sink, thread_name)
                completed += 1
                remaining = total_trials - completed
                print(f"[{thread_name}] Completed {completed}/{total_trials}. Remaining: {remaining}.")


def main():
    # 5 combos tailored for 10k steps
    combos = [
        dict(lr=1e-3, gamma=0.60, batch_size=64,  eps_decay=1500, target_update=750),
        dict(lr=5e-4, gamma=0.75, batch_size=64,  eps_decay=3000, target_update=1000),
        dict(lr=3e-4, gamma=0.80, batch_size=128, eps_decay=5000, target_update=1500),
        dict(lr=1e-4, gamma=0.90, batch_size=128, eps_decay=8000, target_update=2000),
        dict(lr=2e-4, gamma=0.70, batch_size=96,  eps_decay=4000, target_update=1200),
    ]
    widths = [128, 256, 512]
    steps = 4_000

    # Results collector across threads
    results: List[Dict[str, Any]] = []

    threads = []
    for depth in [1, 2, 3]:
        t = threading.Thread(target=run_thread, args=(depth, widths, combos, steps, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Sort by eval_bubbles_popped desc
    results.sort(key=lambda r: r['eval_bubbles_popped'], reverse=True)

    # Write CSV
    csv_path = os.path.join(os.getcwd(), 'grid_search_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'depth','hidden_width','combo_id','lr','gamma','batch_size','eps_decay','target_update',
            'clip_norm','eval_bubbles_popped','steps_trained','seed','model_path'
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved results CSV to {csv_path}")

    # Histogram plot (bar chart) of evaluation scores per combo, colored by depth
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        # Group by (combo_id, depth)
        groups = {}
        for r in results:
            key = (r['combo_id'], r['depth'])
            groups.setdefault(key, []).append(r['eval_bubbles_popped'])
        xs = []
        ys = []
        colors = []
        for (combo_id, depth), vals in groups.items():
            xs.append(f"C{combo_id}-D{depth}")
            ys.append(np.mean(vals))
            colors.append({1:'tab:blue',2:'tab:orange',3:'tab:green'}[depth])
        ax.bar(xs, ys, color=colors)
        ax.set_title('Evaluation (bubbles popped) by Combo and Depth (mean over widths & clip)')
        ax.set_ylabel('Mean bubbles popped (100-step eval)')
        ax.set_xlabel('Combo-Depth')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(os.getcwd(), 'grid_search_histogram.png')
        plt.savefig(plot_path, dpi=200)
        print(f"Saved histogram to {plot_path}")
    except Exception as e:
        print(f"Plot skipped due to error: {e}")


if __name__ == '__main__':
    main()


