# ---------------------------------------------------------*\
# Benchmarking MARL Models
# ---------------------------------------------------------*/

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tabulate import tabulate
import re

import torch
from stable_baselines3 import PPO, DQN
from src.testing import test_env, test_model, test_env_rule
from utils.genetic import genetic_action_search, simulate_candidate
from imitation.policies.base import FeedForward32Policy


# ---------------------------------------------------------------------
# Helper: Load model by type.
# ---------------------------------------------------------------------

def load_model_by_type(model_type, model_path):
    """
    Loads a model based on its type. For PPO, PPOBC, DQN, DQNBC the respective
    loader is used. For BC, the model is assumed to be saved via torch.save(model.state_dict(), ...).
    """
    if model_type in {"PPO", "PPOBC"}:
        model = PPO.load(model_path)
    elif model_type in {"DQN", "DQNRB"}:
        model = DQN.load(model_path)
    elif model_type == "BC":
        model = load_bc_policy(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def load_bc_policy(load_path):
    """Loads a BC-Policy"""
    saved_data = torch.load(load_path)
    policy = FeedForward32Policy(
        observation_space=saved_data["observation_space"],
        action_space=saved_data["action_space"],
        lr_schedule=lambda x: 1e-3
    )
    policy.load_state_dict(saved_data["state_dict"])
    return policy

# ---------------------------------------------------------------------
# Benchmark a Single Seed Across All Policies.
# ---------------------------------------------------------------------


def benchmark_seed_all(seed, models_dict, env_creator, steps_test, out_dir, population_size, generations):
    """
    For a given seed, benchmark:
      - Random actions (via test_env)
      - Rule-Based agent (via test_env_rule)
      - Each model in models_dict (loaded by load_model_by_type)
      - Genetic Algorithm (GA) agent.

    Returns a dictionary with keys:
      "seed", "Random", "Rule-Based", (model types…), "Genetic Algorithm"
    """
    results = {"seed": seed}
    # Create environment
    env = env_creator(env_type="Sorting", seed=seed, max_steps=steps_test, log=False)

    # Random Agent
    env.reset(seed=seed)
    cum_rew_random = test_env(
        env=env,
        tag=f"benchmark_env_seed_{seed}",
        save=False,
        show=False,
        title=f"Benchmark Env Seed {seed}",
        steps=steps_test,
        dir=out_dir,
        seed=seed,
        stats=False
    )
    results["R"] = cum_rew_random

    # Rule-Based Agent
    env.reset(seed=seed)
    cum_rew_rule = test_env_rule(
        env=env,
        tag=f"benchmark_rule_seed_{seed}",
        save=False,
        title=f"Benchmark Rule Seed {seed}",
        steps=steps_test,
        dir=out_dir,
        video=False,
        seed=seed,
        show=False,
        stats=False
    )
    results["RB"] = cum_rew_rule

    # Benchmark each model type from the models dictionary.
    for model_type, model_path in models_dict.items():
        env.reset(seed=seed)
        try:
            model = load_model_by_type(model_type, model_path)
            cum_rew_model = test_model(
                model=model,
                env=env,
                tag=f"benchmark_{model_type}_seed_{seed}",
                save=False,
                show=False,
                title=f"Benchmark {model_type} Seed {seed}",
                steps=steps_test,
                dir=out_dir,
                seed=seed,
                stats=False
            )
            results[model_type] = cum_rew_model
        except Exception as e:
            print(f"Error benchmarking {model_type} at seed {seed}: {e}")
            results[model_type] = None

    # Genetic Algorithm Agent
    env.reset(seed=seed)
    candidate = genetic_action_search(
        env=env,
        steps=steps_test,
        env_seed=seed,
        population_size=population_size,
        generations=generations,
        crossover_rate=0.7,
        mutation_rate=0.1,
        visualize=False,
        PRINT=False,
    )
    cum_rew_ga, _, _ = simulate_candidate(env, candidate, env_seed=seed)
    results["GA"] = cum_rew_ga

    env.close()

    # Print the results for this seed in one row (order: Random, Rule-Based, BC, DQN, DQNBC, PPO, PPOBC, Genetic Algorithm)
    order = ["R", "RB", "BC", "DQN", "DQNRB", "PPO", "PPOBC", "GA"]
    print(f"{seed}\t" + "\t".join(f"{results.get(key, 'N/A'):.2f}" if results.get(key)
                                  is not None else "N/A" for key in order))

    return results

# ---------------------------------------------------------------------
# Run Benchmark Across All Models in Folder.
# ---------------------------------------------------------------------


def run_model_benchmark(env_creator, num_seeds=10, steps_test=50, tag="all_models", out_dir="./img/figures/",
                        population_size=100, generations=10):
    """
    Scans the "./models/" folder (ignoring the "prev" subfolder) for models of type:
      BC, DQN, DQNRB, PPO, PPOBC.
    Then, benchmarks each model over multiple seeds.
    """
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} does not exist.")
        return

    # List all files in models_dir (ignoring directories)
    files = [f for f in os.listdir(models_dir)
             if os.path.isfile(os.path.join(models_dir, f)) and f.endswith(".zip")]

    # Define desired model types exactly.
    desired_types = ["BC", "DQN", "DQNRB", "PPO", "PPOBC"]
    models_dict = {}
    for model_type in desired_types:
        # Use a regex to match exactly: e.g. ^PPO_\d+\.zip$
        pattern = re.compile(f"^{model_type}_\\d+\\.zip$")
        for f in files:
            if pattern.match(f):
                models_dict[model_type] = os.path.join(models_dir, f)
                break  # take the first file that matches for this type

    print("\nModels found for benchmarking:")
    for model_type in desired_types:
        if model_type in models_dict:
            print(f"  {model_type}: {models_dict[model_type]}")
        else:
            print(f"  {model_type}: Not found")

    # Run benchmark across seeds in parallel.
    all_results = []
    cores = max(1, mp.cpu_count() - 2)
    print(f"\n⚙ Running benchmark with {cores} cores across {num_seeds} seeds...\n")
    print("Seed\tR\tRB\tBC\tDQN\tDQNRB\tPPO\tPPOBC\tGA")

    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [
            executor.submit(
                benchmark_seed_all,
                seed,
                models_dict,
                env_creator,
                steps_test,
                out_dir,
                population_size,
                generations
            )
            for seed in range(1, num_seeds + 1)
        ]
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)

    headings = ["Policy", "R", "RB", "DQN", "DQNRB", "PPO", "BC", "PPOBC", "GA"]
    means = ["Mean"]
    stds = ["Std"]
    for key in policy_keys:
        if summary[key]["mean"] is not None:
            means.append(f"{summary[key]['mean']:.2f}")
            stds.append(f"{summary[key]['std']:.2f}")
        else:
            means.append("N/A")
            stds.append("N/A")

    table = [headings, means, stds]
    print("\nSummary of Benchmark Results:")
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

    # Use consistent keys for summary
    policy_keys = ["R", "RB", "DQN", "DQNRB", "PPO", "PPOBC", "GA"]
    # policy_keys = ["R", "RB", "DQN", "DQNRB", "PPO", "BC", "PPOBC", "GA"]
    
    summary = {}
    for key in policy_keys:
        rewards = [res[key] for res in all_results if res.get(key) is not None]
        if rewards:
            summary[key] = {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "rewards": rewards
            }
        else:
            summary[key] = {"mean": None, "std": None, "rewards": []}
            
    # Create bar plot.
    labels = []
    mean_vals = []
    std_vals = []
    for key in policy_keys:
        if summary[key]["mean"] is not None:
            labels.append(key)
            mean_vals.append(summary[key]["mean"])
            std_vals.append(summary[key]["std"])

    if labels:
        x_pos = np.arange(len(labels))
        fig, ax = plt.subplots()
        bars = ax.bar(x_pos, mean_vals, yerr=std_vals, align='center', alpha=0.7, capsize=10)
        ax.set_ylabel('Cumulative Reward')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(f'Model Benchmark ({num_seeds} Seeds)', fontsize=14)
        ax.yaxis.grid(True)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=8)
        os.makedirs(out_dir, exist_ok=True)
        plot_filename_png = './img/figures/Model_Benchmark.png'
        plot_filename_svg = './img/figures/Model_Benchmark.svg'
        plt.tight_layout()
        plt.savefig(plot_filename_png, dpi=300)
        plt.savefig(plot_filename_svg)
        plt.show()
        plt.close()
        print(f"\nBenchmark bar plot saved to {plot_filename_png} and {plot_filename_svg}")
    else:
        print("No data available for plotting.")

    return summary


# -------------------------Notes-----------------------------------------------*\
# R = Random
# RB = Rule-Based
# BC = Behavioral Cloning
# DQN = Deep Q-Network
# DQN-BC = DQN with Behavioral Cloning
# PPO = Proximal Policy Optimization
# PPO-BC = PPO with Behavioral Cloning
# GA = Genetic Algorithm
# -----------------------------------------------------------------------------*/
