# ---------------------------------------------------------*\
# Genetic Algorithm
# ---------------------------------------------------------*/

import matplotlib.pyplot as plt
import random
import numpy as np
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import sys
from src.testing import test_env_rule, test_env
from IPython.display import display, clear_output  # <-- Required for Jupyter

# Check if running inside a Jupyter Notebook
try:
    IN_JUPYTER = hasattr(sys, 'ps1') and ("ipykernel" in sys.modules)
except ImportError:
    IN_JUPYTER = False

# ---------------------------------------------------------*/
# Simulate Candidate
# ---------------------------------------------------------*/

def simulate_candidate(env, candidate, env_seed=42):
    """
    Simulates a candidate sequence in the environment.
    The entire sequence is executed, and the cumulative reward is calculated.
    If the cumulative reward at the end is positive, the sequence is considered valid.

    Returns:
      - cumulative reward,
      - valid: Boolean indicating whether the cumulative reward is positive,
      - valid_length: Length of the sequence (since it is always fully simulated).
    """
    env.reset(seed=env_seed)
    cumulative_reward = 0.0
    for action in candidate:
        obs, reward, done, _, _ = env.step(action)
        cumulative_reward += reward
        if done:
            break
    valid = True  # Valid if cumulative reward is positive
    valid_length = len(candidate)
    return cumulative_reward, valid, valid_length

# ---------------------------------------------------------*/
# Genetic Action Search
# ---------------------------------------------------------*/


def genetic_action_search(env, steps=17, env_seed=None, population_size=100, generations=50,
                          crossover_rate=0.7, mutation_rate=0.1, visualize=True,
                          title="Action Agreement Comparison", PRINT=True):
    """
    Genetic Algorithm that searches for an action sequence with a positive cumulative reward.

    It continuously updates a plot showing:
      - The evolution (max, min, mean) of rewards in the population (primary y-axis)
      - A red 'X' marking the highest reward seen so far, with its value shown as text.

    After evolution, the function compares the GA best candidate with a rule-based agent.
    Finally, it prints the number of unique action sequences in the final population.

    Returns:
      The best candidate sequence (as a tuple) found by the GA if available;
      otherwise, a longest valid prefix.
    """
    # GA parameters and initialization
    actions = [0, 1]
    population = [tuple(random.choices(actions, k=steps)) for _ in range(population_size)]

    best_candidate = None   # Best candidate (highest cumulative reward)
    best_fitness = float('-inf')

    longest_valid_prefix = ()
    longest_valid_length = 0

    # Lists to store reward statistics per generation
    generation_max = []
    generation_min = []
    generation_mean = []

    # Track the highest reward ever seen (for the red "X" marker)
    global_max_reward = float('-inf')
    global_max_gen = None

    # Initialize plot
    if visualize:
        if not IN_JUPYTER:
            plt.ion()  # Use interactive mode only outside Jupyter
        fig, ax = plt.subplots(figsize=(7, 5))
        # Pre-create lines for non-Jupyter mode.
        if not IN_JUPYTER:
            line_max, = ax.plot([], [], label="Max Reward", color="blue")
            line_min, = ax.plot([], [], label="Min Reward", color="orange")
            line_mean, = ax.plot([], [], label="Mean Reward", linestyle="dashed", color="green")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Evolution in the Population")
        ax.set_xlim(0, generations)
        ax.legend(loc="lower right")

    for gen in range(generations):
        fitnesses = []
        for candidate in population:
            reward, valid, valid_length = simulate_candidate(env, candidate, env_seed)
            fitness = reward if valid else -1000
            fitnesses.append(fitness)

            if valid and fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate
            if valid_length > longest_valid_length:
                longest_valid_length = valid_length
                longest_valid_prefix = candidate[:valid_length]

        current_max = max(fitnesses)
        generation_max.append(current_max)
        generation_min.append(min(fitnesses))
        generation_mean.append(sum(fitnesses) / len(fitnesses))

        # Update the "global" max if a new higher reward is found.
        if current_max > global_max_reward:
            global_max_reward = current_max
            global_max_gen = gen

        if visualize:
            if IN_JUPYTER:
                clear_output(wait=True)
                ax.clear()
                # Plot reward statistics on primary y-axis.
                ax.plot(generation_max, label="Max Reward", color="blue")
                ax.plot(generation_min, label="Min Reward", color="orange")
                ax.plot(generation_mean, label="Mean Reward", linestyle="dashed", color="green")
                ax.set_xlabel("Generation")
                ax.set_ylabel("Reward")
                ax.set_title("Reward Evolution in the Population")
                ax.set_xlim(0, generations)
                # Mark the global highest reward with a red 'X' and text.
                if global_max_gen is not None:
                    ax.plot(global_max_gen, global_max_reward, marker='x', color='red', markersize=10)
                    ax.text(global_max_gen, global_max_reward - 5, f"{global_max_reward:.2f}",
                            color='red', ha='center', va='top')
                ax.legend(loc="lower right")
                display(fig)
            else:
                # Update lines for non-Jupyter mode.
                x_data = np.arange(len(generation_max))
                line_max.set_data(x_data, generation_max)
                line_min.set_data(x_data, generation_min)
                line_mean.set_data(x_data, generation_mean)
                ax.set_xlim(0, generations)
                ax.set_ylim(min(generation_min) - 10, max(generation_max) + 10)
                fig.canvas.draw()
                plt.pause(0.1)
                # Mark the global highest reward with a red 'X' and add text.
                if global_max_gen is not None:
                    ax.plot(global_max_gen, global_max_reward, marker='x', color='red', markersize=10)
                    ax.text(global_max_gen, global_max_reward - 5, f"{global_max_reward:.2f}",
                            color='red', ha='center', va='top')

        # Tournament selection: choose the better of two random candidates.
        def tournament_select():
            i, j = random.sample(range(population_size), 2)
            return population[i] if fitnesses[i] > fitnesses[j] else population[j]

        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_select()
            parent2 = tournament_select()
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, steps - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1, child2 = parent1, parent2

            def mutate(child):
                return tuple(action if random.random() > mutation_rate else random.choice(actions) for action in child)

            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))

        population = new_population

    # Rule-Based Agent Execution and comparison
    cumulative_rand_reward = test_env(env, seed=env_seed, steps=steps, title="Random Agent", show=False, stats=False)
    cumulative_rule_reward = test_env_rule(env, seed=env_seed, steps=steps,
                                           title="Rule-Based Agent", show=False, stats=False)
    if visualize:
        if not IN_JUPYTER:
            plt.ioff()
            plt.show()

    # Calculate number of unique action sequences in the final population.
    unique_sequences = len(set(population))

    if PRINT:
        print("--------------------------------------------------------")
        print(f"Genetic algorithm completed for Seed {env_seed} ✔️")
        print(f"Random Agent Cumulative Reward: {cumulative_rand_reward:.2f}")
        print(f"Rule-Based Agent Cumulative Reward: {cumulative_rule_reward:.2f}")
        print(f"Genetic Algorithm Best Cumulative Reward: {best_fitness:.2f}")
        print(f"Unique action sequences in the final population: {unique_sequences} out of {population_size}")
        if best_candidate is not None:
            print(f"Best GA sequence with cumulative reward: {best_fitness:.2f} -> {best_candidate}")
        else:
            print("No sequence with positive cumulative reward found.")
        print("--------------------------------------------------------")

        categories = ['Random Actions Reward', 'Rule-Based Reward', 'GA Reward']
        values = [cumulative_rand_reward, cumulative_rule_reward, best_fitness]
        # Create the bar plot
        plt.figure(figsize=(10, 5))
        plt.bar(categories, values, color=['blue', 'green', 'red'])

        # Add title and axis labels
        plt.title(f'Comparison of Rewards and Fitness (Seed {env_seed} with {generations} generations)')
        plt.xlabel('')
        plt.ylabel('Values')

        # Display the plot
        plt.savefig("img/figures/4_ga_development.png", dpi=300)
        plt.savefig("img/figures/4_ga_development.svg")
        plt.show()

    return best_candidate if best_candidate is not None else longest_valid_prefix


# ---------------------------------------------------------*/
# Genetic Versus Rule Based
# ---------------------------------------------------------*/

# Wrapper function to unpack arguments and call evaluate_seed.
def evaluate_seed_wrapper(args):
    return evaluate_seed(*args)

# Generates demonstrations by simulating the candidate in the environment.
def generate_demonstrations(env, candidate, env_seed=None):
    transitions = []
    obs, _ = env.reset(seed=env_seed)
    for action in candidate:
        next_obs, reward, done, _, _ = env.step(action)
        transition = {
            "state": [x for x in obs.tolist()] if isinstance(obs, np.ndarray) else obs,
            "action": action,
            "reward": reward,
            "next_state": [x for x in next_obs.tolist()] if isinstance(next_obs, np.ndarray) else next_obs,
            "done": done
        }
        transitions.append(transition)
        obs = next_obs
        if done:
            break
    return transitions

# Calculates the relative difference in a stable way.
def calculate_relative_difference(ga_reward, rba_reward):
    if rba_reward == 0:
        return float('inf') if ga_reward > 0 else float('-inf')
    return int(round((ga_reward - rba_reward) / abs(rba_reward) * 100))

def evaluate_seed(seed, steps, population_size, generations, env_creator, save_json=True):
    env = env_creator(env_type="Sorting", seed=seed, log=False)

    # Run Random Agent.
    env.reset(seed=seed)
    r_random = test_env(env, tag=f"Random_{seed}", save=False, title=f"Random Agent Seed {seed}",
                        steps=steps, dir="./img/", video=False, seed=seed, show=False, stats=False)

    # Run Rule-Based Agent.
    env.reset(seed=seed)
    rba_reward = test_env_rule(env, tag=f"RBA_{seed}", save=False, title=f"Rule-Based Seed {seed}",
                               steps=steps, dir="./img/", video=False, seed=seed, show=False, stats=False)

    # Run Genetic Algorithm.
    env.reset(seed=seed)
    candidate = genetic_action_search(env=env, steps=steps, env_seed=seed, population_size=population_size,
                                      generations=generations, crossover_rate=0.7, mutation_rate=0.1,
                                      visualize=False, PRINT=False)

    # Evaluate GA candidate.
    ga_reward, _, _ = simulate_candidate(env, candidate, env_seed=seed)
    difference = calculate_relative_difference(ga_reward, rba_reward)

    # Only return if GA outperforms RBA by >= 15%.
    if ga_reward > rba_reward and difference >= 15:
        print(f"{seed}\t{r_random:.2f}\t{rba_reward:.2f}\t{ga_reward:.2f}\t{(ga_reward - rba_reward):.2f}\t{difference:.0f}%\t✅")

        # Save demonstrations as JSON (optional).
        if save_json:
            demonstrations = generate_demonstrations(env, candidate, env_seed=seed)
            folder_path = os.path.dirname("./models/demonstrations/")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            filename = f"./models/demonstrations/demonstrations_seed_{seed}.json"
            with open(filename, "w") as f:
                json.dump(demonstrations, f, indent=4)

        return (seed, r_random, rba_reward, ga_reward)
    
    else:
        print(f"{seed}\t{r_random:.2f}\t{rba_reward:.2f}\t{ga_reward:.2f}\t{ga_reward - rba_reward:.2f}\t{difference:.0f}%\t❌")
        return None


def compare_agents_parallel(env_creator, start_seed, num_runs, steps=50, population_size=100, generations=50, save_json=True):
    print(f"Collecting {num_runs} valid results...")

    processes = max(1, mp.cpu_count() - 2)  # Use available cores but keep some free.
    print(f"Using {processes} cores for parallel processing...")

    results = []
    current_seed = start_seed
    print(f"\nSeed\tR\tRB\tGA\tDiff\t%\tSave")
    print("-" * 60)  # Adds a horizontal line for better readability
          
    while len(results) < num_runs:
        needed = num_runs - len(results)
        seeds_to_try = list(range(current_seed, current_seed + needed))
        args = [(seed, steps, population_size, generations, env_creator, save_json) for seed in seeds_to_try]

        with ProcessPoolExecutor(max_workers=processes) as executor:
            new_results = list(executor.map(evaluate_seed_wrapper, args))

        # Only keep valid results.
        valid_results = [(seed, result) for seed, result in zip(seeds_to_try, new_results) if result is not None]
        results.extend([r for _, r in valid_results])
        current_seed += needed

    print(f"\nCollected {len(results)} valid results (Target: {num_runs}).")

    # Sort and extract valid results.
    results.sort(key=lambda x: x[0])
    seeds = list(range(1, len(results) + 1))  # Numbering seeds from 1.
    random_rewards = [r[1] for r in results]
    rba_rewards = [r[2] for r in results]
    ga_rewards = [r[3] for r in results]

    # Create directory for figures if it doesn't exist.
    fig_dir = "img/figures"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle("Comparison of Cumulative Rewards", fontsize=18, fontweight="bold")
    
    # Plot 1: Line Plot (Cumulative Rewards per Seed)
    ax1.plot(seeds, random_rewards, marker="o", label="Random", color="blue")
    ax1.plot(seeds, rba_rewards, marker="o", label="Rule-Based", color="green")
    ax1.plot(seeds, ga_rewards, marker="o", label="GA", color="red")
    
    # Fill the area under each line
    ax1.fill_between(seeds, random_rewards, color="blue", alpha=0.1)
    ax1.fill_between(seeds, rba_rewards, color="green", alpha=0.1)
    ax1.fill_between(seeds, ga_rewards, color="red", alpha=0.1)
    
    ax1.set_title("Cumulative Reward per Seed", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Cumulative Reward", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    # Set font sizes for x-ticks and y-ticks
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: Boxplot (Cumulative Rewards per Agent)
    ax2.boxplot([random_rewards, rba_rewards, ga_rewards],
                labels=["Random", "Rule-Based", "GA"],
                showmeans=True,
                widths=0.6)  # Increase the width of the boxplots
    ax2.set_title("Cumulative Rewards per Agent", fontsize=16, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Set font sizes for x-ticks and y-ticks
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Remove the y-axis label from the second plot
    ax2.set_ylabel("")
    
    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for the suptitle
    plt.savefig(os.path.join(fig_dir, "combined_plot_rewards.png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "combined_plot_rewards.svg"))
    plt.show()
    
    return results


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
