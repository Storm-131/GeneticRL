#---------------------------------------------------------*\
# Genetic Algorithm vs. Bruteforce
#---------------------------------------------------------*/

import matplotlib.pyplot as plt
import random
import numpy as np
import multiprocessing as mp
from itertools import product
from src.testing import test_env_rule, test_env
from utils.bruteforce import process_batch
from utils.genetic import simulate_candidate
import sys
import IPython
import os

# Ensure consistent seeding for random and numpy
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------------------------------------#
# GA Function that returns best reward (cumulative reward) #
# ---------------------------------------------------------#
def genetic_action_search_return_reward(env, steps=15, env_seed=None, population_size=100, generations=50,
                                          crossover_rate=0.7, mutation_rate=0.1, visualize=False, PRINT=False):
    set_global_seed(env_seed)  # Set global seed for reproducibility
    actions = [0, 1]
    population = [tuple(random.choices(actions, k=steps)) for _ in range(population_size)]
    best_fitness = float('-inf')

    for gen in range(generations):
        fitnesses = []
        for candidate in population:
            env.reset(seed=env_seed)  # Ensure environment is reset with the correct seed
            reward, valid, _ = simulate_candidate(env, candidate, env_seed)
            fitness = reward if valid else -1000
            fitnesses.append(fitness)
            if valid and fitness > best_fitness:
                best_fitness = fitness

        # Tournament selection function
        def tournament_select():
            i, j = random.sample(range(population_size), 2)
            return population[i] if fitnesses[i] > fitnesses[j] else population[j]

        # Create new population via crossover and mutation
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

    return best_fitness

# ---------------------------------------------------------------------#
# Brute-Force Function that returns best reward for a given seed         #
# ---------------------------------------------------------------------#
def brute_force_action_search_return_reward(env, steps=15, env_seed=42, num_processes=None, plot=False):
    set_global_seed(env_seed)  # Ensure random and numpy use the correct seed
    
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 16)  # Limit to 16 cores max for efficiency
    
    all_sequences = list(product([0, 1], repeat=steps))
    total_sequences = len(all_sequences)

    # Ensure all sequences are divided properly across processes
    batch_size = (total_sequences + num_processes - 1) // num_processes  # Ensure all sequences are covered
    batches = [all_sequences[i * batch_size:(i + 1) * batch_size] for i in range(num_processes)]
    
    # If the last batch is empty (happens when batch_size is overestimated), remove it
    batches = [b for b in batches if b]

    print(f"\nBruteforcing for Seed {env_seed}, using {num_processes} cores for parallel processing.")
    print(f"Total sequences to test: {total_sequences}, Batch size: {batch_size}, Actual batches: {len(batches)}")

    pool = mp.Pool(processes=num_processes)
    results = []
    for idx, batch in enumerate(batches):
        results.append(pool.apply_async(process_batch, args=(batch, env, env_seed, steps, idx, num_processes)))

    pool.close()
    pool.join()

    best_reward_overall = float('-inf')
    for res in results:
        _, _, best_reward, _ = res.get()
        if best_reward > best_reward_overall:
            best_reward_overall = best_reward

    # Flush the output
    sys.stdout.flush()
    IPython.display.clear_output(wait=True)
 
    return best_reward_overall


# ---------------------------------------------------------------------------------------#
# Main Comparison Function: Runs each method over 50 seeds (steps=15) and plots Boxplots  #
# ---------------------------------------------------------------------------------------#
def compare_methods_rewards(env, steps=15, seeds=50, num_processes=None, generations=10):
    random_rewards = []
    rule_rewards = []
    ga_rewards = []
    bf_rewards = []

    # Print the heading once
    print("Seed\tRandom\tRule\tGA\tBF")

    for seed in range(seeds):
        set_global_seed(seed)  # Ensure reproducibility for each run

        # Get reward for Random Agent
        env.reset(seed=seed)
        r_random = test_env(env, seed=seed, steps=steps, title="Random Agent", show=False, stats=False)
        random_rewards.append(r_random)

        # Get reward for Rule-Based Agent
        env.reset(seed=seed)
        r_rule = test_env_rule(env, seed=seed, steps=steps, title="Rule-Based Agent", show=False, stats=False)
        rule_rewards.append(r_rule)

        # Get reward for Genetic Algorithm
        env.reset(seed=seed)
        r_ga = genetic_action_search_return_reward(env, steps=steps, env_seed=seed, population_size=100,
                                                     generations=generations, crossover_rate=0.7, mutation_rate=0.1,
                                                     visualize=False, PRINT=True)
        ga_rewards.append(r_ga)

        # Get reward for Brute-Force Search
        env.reset(seed=seed)
        r_bf = brute_force_action_search_return_reward(env, steps=steps, env_seed=seed, num_processes=num_processes, plot=False)
        bf_rewards.append(r_bf)
        
        print("\nSeed\tRandom\tRule\tGA\tBF")
        # Loop through the results and print each one
        for seed, (r_random, r_rule, r_ga, r_bf) in enumerate(zip(random_rewards, rule_rewards, ga_rewards, bf_rewards)):
            print(f"{seed}\t{r_random:.2f}\t{r_rule:.2f}\t{r_ga:.2f}\t{r_bf:.2f}")

    # Create Boxplots for each method
    data = [random_rewards, rule_rewards, ga_rewards, bf_rewards]
    labels = ['Random', 'Rule-Based', 'GA', 'Bruteforce']
    plt.figure(figsize=(5, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(f'Comparison of Reward and Fitness (Steps={steps}, Seeds={seeds}, Generations={generations})')
    plt.ylabel('Cumulative Reward')
    
    # Define the base filename and directory
    base_filename = "Gen_Brut_Comp"
    directory = "img/figures"

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Function to generate a unique filename if the file already exists
    def get_unique_filename(base_filename, extension):
        filename = f"{base_filename}.{extension}"
        counter = 1
        while os.path.exists(os.path.join(directory, filename)):
            filename = f"{base_filename}_{counter}.{extension}"
            counter += 1
        return filename

    # Get unique filenames for both PNG and SVG
    png_filename = get_unique_filename(base_filename, "png")
    svg_filename = get_unique_filename(base_filename, "svg")

    # Save the plot as PNG and SVG
    plt.savefig(os.path.join(directory, png_filename), dpi=300)
    plt.savefig(os.path.join(directory, svg_filename))
    
    plt.savefig("img/figures/5_Gen_Brut_Comp.png", dpi=300)
    plt.savefig("img/figures/5_Gen_Brut_Comp.svg",)
    plt.show()
    
    return {"Random": random_rewards, "Rule-Based": rule_rewards, "GA": ga_rewards, "Bruteforce": bf_rewards}


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\