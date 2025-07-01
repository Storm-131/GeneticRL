# ---------------------------------------------------------*\
# Main-Control
# ---------------------------------------------------------*/

# Environment & Parameters
from src.env import Env_Sorting

# RL: Trainer / Tester
from src.testing import test_env, test_env_rule, test_env_sequence
from src.training import RL_Trainer

# Utils
from utils.plot_env_analysis import run_env_analysis
from utils.simulation import env_simulation_video, interactive_simulation

# Benchmark
from utils.benchmark_models import run_model_benchmark
from utils.benchmark_testing import run_test_benchmark, plot_bale_purity
from utils.bruteforce import brute_force_action_search
from utils.genetic import genetic_action_search, compare_agents_parallel
from utils.gen_brute_comp import compare_methods_rewards

# ---------------------------------------------------------*/
# Parameters
# ---------------------------------------------------------*/

# 1. Choose Mode
# ---------------------------------------------------------*/
TEST = 1             # Test the Model
TRAIN = 0            # Train the Model

SIMULATION = 0       # For Simulation Mode (Interactive)
VIDEO = 0            # Record Video, for Test and Load Mode

GENETIC = 0          # Genetic Algorithm in comparison to Rule Based Agent
GENETIC_DEMO = 0     # Generate Demonstration-Data for Replay Buffer from Genetic Algorithm
GEN_BRUT_COMP = 0    # Compare Genetic Algorithm with Brute-Force Search
BRUTEFORCE = 0       # Brute-Force Action Search

ENV_ANALYSIS = 0     # Analyze the Environment
BENCHMARK_MODEL = 0  # Benchmark Models
BENCHMARK_TEST = 0   # Benchmark Tests

# 2. Environment Parameters
# ---------------------------------------------------------*/
NOISE_INPUT = 0.0      # Noise Range for Input Observation
NOISE_SORTING = 0.05    # Noise Range for Sorting Accuracy
BALESIZE = 200         # Standard Bale Size (in material-units)

# 3. Parameters
# ---------------------------------------------------------*/
MODEL = ["BC", "DQN", "DQNRB", "PPO", "PPOBC"]    # Select Model for Training
TIMESTEPS = 500_000     # Total Training Steps (Budget)
STEPS_TRAIN = 100       # Steps per Episode (Training)
STEPS_TEST = 100        # Steps per Episode (Testing)
SEED = 42               # Random Seed for Reproducibility

BENCH_SEEDS = 100       # Number of Seeds for Benchmarking

GENETIC_RUNS = 6        # Number of Genetic Algorithm Runs
GEN_SEED_START = 10108  # Starting Seed for Genetic Algorithm

SAVE = 1                # Save Images
DIR = "./img/figures/"  # Directory for Image-Logging

TAG = f""               # Specific Tag for Logging

# ---------------------------------------------------------*/
# Run Environment Simulation
# ---------------------------------------------------------*/

def create_environment(env_type, max_steps=STEPS_TEST, seed=SEED, SIMULATION=False, log=True):
    """Create a new environment based on the specified type and parameters"""
    if log:
        print(
            f"Creating environment of type {env_type} with: steps={max_steps}, noise_sorting={NOISE_SORTING}, "
            f"noise_input={NOISE_INPUT}, seed={seed}, simulation={SIMULATION}")

    if env_type == "Sorting":
        return Env_Sorting(max_steps=max_steps, noise_sorting=NOISE_SORTING, noise_input=NOISE_INPUT,
                             seed=seed, simulation=SIMULATION)

    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def run_sim(TEST=TEST, TRAIN=TRAIN, VIDEO=VIDEO, ENV_ANALYSIS=ENV_ANALYSIS,
            BENCHMARK_MODEL=BENCHMARK_MODEL, BENCHMARK_TEST=BENCHMARK_TEST, NOISE_SORTING=NOISE_SORTING, NOISE_INPUT=NOISE_INPUT,
            TIMESTEPS=TIMESTEPS, STEPS_TRAIN=STEPS_TRAIN, SIMULATION=SIMULATION, GENETIC=GENETIC, GEN_SEED_START=GEN_SEED_START,
            STEPS_TEST=STEPS_TEST, SEED=SEED, SAVE=SAVE, DIR=DIR, MODEL=MODEL, GENETIC_DEMO=GENETIC_DEMO, BENCH_SEEDS=BENCH_SEEDS,
            TAG=TAG, ENV="MARL", BRUTEFORCE=BRUTEFORCE, GEN_BRUT_COMP=GEN_BRUT_COMP):
    """Run the simulation based on the specified parameters """

    print("\n--------------------------------")
    print("Starting Simulation... 🚀")
    print("--------------------------------")
        
    def run_test(env_type=ENV, steps_test=STEPS_TEST, seed=SEED, tag=TAG, save=SAVE, dir=DIR):
        env = create_environment(env_type=env_type, max_steps=steps_test, seed=seed, log=False)

        random_reward = test_env(env=env, tag=tag, save=save,
                                 title=f"(Random Run {tag})", steps=steps_test, dir=dir, seed=seed)

        rule_based_reward = test_env_rule(env=env, tag=tag, save=save,
                                          title=f"(Rule-Based Run {tag})", steps=steps_test, dir=dir, seed=seed)

        return random_reward, rule_based_reward

    # ---------------------------------------------------------*/
    # Basic Functionality
    # ---------------------------------------------------------*/
    if TEST:
        run_test()

    if TRAIN:
        models = RL_Trainer(model_list=MODEL, total_timesteps=TIMESTEPS, max_steps=STEPS_TRAIN, noise_sorting=NOISE_SORTING,
                                      noise_input=NOISE_INPUT, tag=TAG, seed=SEED)        

    if SIMULATION:
        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED, SIMULATION=SIMULATION)
        interactive_simulation(env=env, steps=STEPS_TEST)

    if VIDEO:
        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED, SIMULATION=SIMULATION)
        env_simulation_video(env=env, tag=TAG, steps=STEPS_TEST)

    # ---------------------------------------------------------*/
    # Bruteforce + Genetic Algorithms
    # ---------------------------------------------------------*/
    if GENETIC:
        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED, log=False)

        # Genetic Algorithm Search
        best_seq = genetic_action_search(env=env, steps=STEPS_TEST, env_seed=SEED, population_size=100,
                                         generations=25, crossover_rate=0.7, mutation_rate=0.1)

        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED)
        test_env_sequence(env=env, sequence=best_seq, title="Genetic Algorithm")

    if GENETIC_DEMO:
        # Compare Agents: RBA with Genetic Algorithm
        results = compare_agents_parallel(create_environment, start_seed=GEN_SEED_START, save_json=False,
                                          num_runs=GENETIC_RUNS, steps=STEPS_TRAIN, generations=10)
        
    if GEN_BRUT_COMP:
        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED, log=False)
        compare_methods_rewards(env, steps=15, seeds=50, num_processes=None)
        
    if BRUTEFORCE:
        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED, log=False)
        brute_force_action_search(env, steps=15, env_seed=42, num_processes=None)

    # ---------------------------------------------------------*/
    # Analysis + Benchmarking
    # ---------------------------------------------------------*/
    if ENV_ANALYSIS:
        run_test()
        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED, SIMULATION=SIMULATION)
        run_env_analysis(env)
        BENCHMARK_TEST = 1
        
    if BENCHMARK_MODEL:
        print("\nStarting Benchmark...")
        run_model_benchmark(env_creator=create_environment, num_seeds=BENCH_SEEDS, steps_test=STEPS_TEST, tag=TAG,)

    if BENCHMARK_TEST:
        print("\nStarting Benchmark Test...")
        env = create_environment(env_type=ENV, max_steps=STEPS_TEST, seed=SEED)
        results_df = run_test_benchmark(env=env, steps_test=STEPS_TEST, num_benchmarks=48, num_runs=1, env_seed=42, seed=1,)
        plot_bale_purity(results_df)
        
    else:
        print("\n--------------------------------")
        print("Simulation Completed. 🌵")
        print("--------------------------------")


# ---------------------------------------------------------*/
# Main Function
# ---------------------------------------------------------*/
if __name__ == "__main__":

    # Run Simulation
    run_sim(TAG=TAG, ENV="Sorting")

# -------------------------Notes-----------------------------------------------*\

# -----------------------------------------------------------------------------
