# ---------------------------------------------------------*\
# Title: Run Test Benchmark with Detailed Purity Scores
# ---------------------------------------------------------*/

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt


# ---------------------------------------------------------*/
# Run Test Benchmark
# ---------------------------------------------------------*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_test_benchmark(
    env=None,
    steps_test=50,
    num_runs=1,
    num_benchmarks=3,
    env_seed=42,
    seed=1,
    save=False,
    dir="",
    tag="benchmark",
    title="Benchmark Results",
    show=False
):
    """
    Run multiple test benchmarks and store the results of final container purities per bale.
    """
    if env is None:
        raise ValueError("Environment must be provided")

    containers = ["A", "B", "C", "D", "E"]
    results = []
    reward_data = []

    def collect_purities(benchmark, run):
        """Helper function to collect bale purities."""
        for mat in containers:
            if hasattr(env, "bale_count") and mat in env.bale_count:
                if len(env.bale_count[mat]) > 0:
                    for bale_num, bale in enumerate(env.bale_count[mat], start=1):
                        size, quality = bale
                        results.append({
                            "Benchmark": benchmark,
                            "Run": run,
                            "Container": mat,
                            "Bale_Number": bale_num,
                            "Purity_Percentage": quality
                        })
                else:
                    results.append({
                        "Benchmark": benchmark,
                        "Run": run,
                        "Container": mat,
                        "Bale_Number": None,
                        "Purity_Percentage": np.nan
                    })

    static_actions = {
        "fixed_0": 0,
        "fixed_1": 1,
    }

    for benchmark, action_sort in static_actions.items():
        for run in range(1, num_runs + 1):
            obs, _ = env.reset(seed=env_seed)
            cumulative_reward = 0.0
            action_sequence = []

            for step in range(steps_test):
                obs, reward, done, _, _ = env.step(action_sort)
                action_sequence.append(action_sort)
                cumulative_reward += reward
                if done:
                    break
            
            # print(f"Benchmark: {benchmark}, Run: {run}, Seed: {env_seed}, Actions: {action_sequence}, Cumulative Reward: {cumulative_reward}")
            reward_data.append({"Benchmark": benchmark, "Run": run, "Cumulative Reward": cumulative_reward})
            
            if not hasattr(env, "bale_count") or not isinstance(env.bale_count, dict):
                raise ValueError("Environment does not have a valid `bale_count` structure.")

            collect_purities(benchmark, run)

    for benchmark in range(1, num_benchmarks + 1):
        for run in range(1, num_runs + 1):
            obs, _ = env.reset(seed=env_seed)
            action_seed = seed + (benchmark - 1)
            env.action_space.seed(action_seed)
            cumulative_reward = 0.0
            action_sequence = []

            for step in range(steps_test):
                action_sort = env.action_space.sample() % 2  # Ensure valid action
                obs, reward, done, _, _ = env.step(action_sort)
                action_sequence.append(action_sort)
                cumulative_reward += reward
                if done:
                    break
            
            # print(f"Benchmark: random_{benchmark}, Run: {run}, Seed: {action_seed}, Actions: {action_sequence}, Cumulative Reward: {cumulative_reward}")
            reward_data.append({"Benchmark": f"random_{benchmark}", "Run": run, "Cumulative Reward": cumulative_reward})
            
            if not hasattr(env, "bale_count") or not isinstance(env.bale_count, dict):
                raise ValueError("Environment does not have a valid `bale_count` structure.")

            collect_purities(f"random_{benchmark}", run)

    results_df = pd.DataFrame(results)
    reward_df = pd.DataFrame(reward_data)
    
    # Plot sorted barplot of cumulative rewards
    reward_df_sorted = reward_df.sort_values(by="Cumulative Reward", ascending=False)
    plt.figure(figsize=(10, 10))
    plt.barh(reward_df_sorted["Benchmark"].astype(str), reward_df_sorted["Cumulative Reward"], color='skyblue')
    plt.xlabel("Cumulative Reward")
    plt.ylabel("Benchmark - Run")
    plt.title("Sorted Cumulative Reward per Run")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
    
    return results_df



# ---------------------------------------------------------*/
# Plot Bale Purity
# ---------------------------------------------------------*/

def plot_bale_purity(results):
    """
    Generate a bar plot for mean purity and standard deviation per bale,
    dynamically scaling alpha by the number of appearances per bale.
    """

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Drop rows with missing purity values
    results_df = results_df.dropna(subset=["Purity_Percentage"])

    # Group by Container and Bale_Number to calculate mean, std deviation, and counts
    stats = results_df.groupby(["Container", "Bale_Number"])[
        "Purity_Percentage"].agg(["mean", "std", "count"]).reset_index()

    # Combine Container and Bale_Number for unique labels on X-axis (e.g., A1, B1)
    stats["Bale"] = stats["Container"] + stats["Bale_Number"].astype(int).astype(str)

    # Assign colors by container type
    container_colors = {
        "A": "blue",
        "B": "green",
        "C": "orange",
        "D": "red",
        "E": "purple",
    }
    stats["Color"] = stats["Container"].map(container_colors)

    # Normalize counts to scale alpha dynamically (between 0.4 and 1.0 for better visibility)
    min_count = stats["count"].min()
    max_count = stats["count"].max()
    stats["Alpha"] = 0.4 + 0.6 * (stats["count"] - min_count) / (max_count - min_count)

    # Calculate the mean standard deviation across all bales
    mean_std = stats["std"].mean()

    # Plot
    plt.figure(figsize=(10, 6))
    bars = []
    for _, row in stats.iterrows():
        bars.append(
            plt.bar(
                row["Bale"],
                row["mean"],
                yerr=row["std"],
                capsize=5,
                color=row["Color"],
                alpha=row["Alpha"]
            )
        )

    # Add count labels below the bars
    for bar, count in zip(bars, stats["count"]):
        plt.text(bar[0].get_x() + bar[0].get_width() / 2, 0, f"n={int(count)}", ha="center", va="bottom", fontsize=10)

    # Display the mean standard deviation as a text annotation
    plt.text(
        0.5, 90, f"Mean Std: {mean_std:.2f}", ha="center", va="top", fontsize=12, color="black"
    )

    plt.ylim(0, 100)
    plt.xlabel("Bales (e.g., A1, A2, B1, ...)")
    plt.ylabel("Mean Purity (%)")
    plt.title("Random Action Seed-Benchmark: Mean Purity and Standard Deviation per Bale Position")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# -------------------------Notes-----------------------------------------------*\
# The benchmark evaluates the variability of bale purity outcomes in a controlled 
# environment by consistently resetting it to the same initial state using a fixed seed. 
# It compares results from predefined static actions and dynamically sampled random actions,
# where randomness is controlled by varying seeds for the action space. 
# This approach isolates the impact of action choices on the results while maintaining identical 
# starting conditions, providing insight into the sensitivity of purity outcomes to different action strategies.
# -----------------------------------------------------------------------------*\
