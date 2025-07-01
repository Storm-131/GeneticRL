#---------------------------------------------------------*\
# Title: Bruteforce Action Search
#---------------------------------------------------------*/

from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import copy

#---------------------------------------------------------*/
# Bruteforce Action Search
#---------------------------------------------------------*/

class DummyProgressBar:
    def update(self, n):
        pass
    def close(self):
        pass

def process_batch(sequences, env, env_seed, steps, proc_idx, num_processes):
    """
    Processes a batch of sequences in a separate process.
    Displays a real tqdm progress bar only for the first process (proc_idx == 0).

    Returns:
    - Number of sequences in this batch with positive cumulative reward,
    - The sequence with the highest reward in this batch,
    - The highest reward in this batch,
    - The number of sequences evaluated.
    """
    if proc_idx == 0:
        pbar = tqdm(total=len(sequences), desc=f"Cores: {num_processes}", position=0, ncols=80, leave=True)
    else:
        pbar = DummyProgressBar()

    positive_count = 0
    best_seq = None
    best_reward = float('-inf')
    
    for sequence in sequences:
        local_env = copy.deepcopy(env)
        local_env.reset(seed=env_seed)
        cumulative_reward = 0.0

        for action in sequence:
            obs, reward, done, _, _ = local_env.step(action)
            cumulative_reward += reward
            if done:
                break

        if cumulative_reward > 0:
            positive_count += 1
        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            best_seq = sequence

        pbar.update(1)
    pbar.close()
    return positive_count, best_seq, best_reward, len(sequences)

def brute_force_action_search(env, steps=12, env_seed=42, num_processes=None):
    """
    Performs a parallel brute-force search over all possible action sequences (actions 0 and 1) for 'steps' steps.
    The given environment is duplicated in each process using copy.deepcopy.
    Only the first process displays a real tqdm progress bar; the other processes use a dummy bar.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Using {num_processes} cores for parallel processing.")
    
    all_sequences = list(product([0, 1], repeat=steps))
    total_sequences = len(all_sequences)
    print(f"Total sequences to test: {total_sequences}")

    batch_size = total_sequences // num_processes
    batches = [all_sequences[i * batch_size:(i + 1) * batch_size] for i in range(num_processes)]
    if total_sequences % num_processes:
        batches[-1].extend(all_sequences[num_processes * batch_size:])

    pool = mp.Pool(processes=num_processes)
    results = []
    for idx, batch in enumerate(batches):
        results.append(pool.apply_async(process_batch, args=(batch, env, env_seed, steps, idx, num_processes)))
    pool.close()
    pool.join()

    total_positive = 0
    best_seq_overall = None
    best_reward_overall = float('-inf')
    total_tested = 0

    for res in results:
        pos_count, best_seq, best_reward, tested = res.get()
        total_positive += pos_count
        total_tested += tested
        if best_reward > best_reward_overall:
            best_reward_overall = best_reward
            best_seq_overall = best_seq

    print("Brute-force search complete.")
    print(f"Number of sequences with positive reward: {total_positive} out of {total_sequences}")
    print(f"Highest final reward: {best_reward_overall:.2f} for action sequence: {best_seq_overall}")

    # Plotting results
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = ['Positive Reward', 'Non-Positive Reward']
    sizes = [total_positive, total_sequences - total_positive]
    colors = ['skyblue', 'lightcoral']
    explode = (0.1, 0)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        explode=explode, shadow=True, startangle=140, textprops={'fontsize': 12}
    )
    ax.set_title(f'Ratio of Sequences with Positive Reward: {steps} Steps', fontsize=14)
    plt.text(
        1, 1,
        f'Highest Final Reward: {best_reward_overall:.2f}\nBest sequence: {best_seq_overall}',
        fontsize=14, bbox=dict(facecolor='lightgreen', alpha=0.5, boxstyle="round,pad=0.3")
    )
    plt.show()

    return best_seq_overall


#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\