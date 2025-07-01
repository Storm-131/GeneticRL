# ---------------------------------------------------------*\
# Title: Simulations
# ---------------------------------------------------------*/

import os
import shutil
from tqdm import tqdm
from src.env import Env_Sorting
from utils.plotting import create_video

# ---------------------------------------------------------*/
# Interactive Simulation
# ---------------------------------------------------------*/

def interactive_simulation(env=None, steps=50):
    # Raise error if no environment is provided
    if env is None:
        raise ValueError("Environment must be provided")

    print("Starting interactive simulation...")

    # Display possible actions for the sorting agent
    env.print_possible_actions()

    obs, info = env.reset()

    # Split the observation vector according to the new observation space (33 dimensions)
    input_material = obs[0:4]              # Input Material (without noise)
    belt_material = obs[4:8]               # Belt Material
    accuracy_belt = obs[8:12]              # Accuracy for Belt Material
    sorting_material = obs[12:16]          # Sorting Machine Material
    accuracy_sorting = obs[16:20]          # Accuracy for Sorting Machine Material
    purity_diff = obs[20:24]               # Purity Differences for Containers A-D
    container_levels = obs[24:33]          # Container fill levels (8 values for A-D and 1 for E)

    print("\nInitial Observation:")
    print("Input Material:", input_material)
    print("Belt Material:", belt_material)
    print(f"Accuracy for Belt:", accuracy_belt)
    print("Sorting Machine Material:", sorting_material)
    print(f"Accuracy for Sorting Machine:", accuracy_sorting)
    print("Purity Differences:", purity_diff)
    print("Container Fill Levels:", container_levels)

    env.render(save=False, log_dir="./img/", filename='interactive_step_start',
               title='Step Start', format='png')

    for i in range(steps):
        # User input for action
        sorting_mode_action = int(input(f"Step {i+1} - Choose a sorting mode (0 or 1): "))
        if sorting_mode_action not in [0, 1]:
            print("Invalid sorting mode. Please choose 0 or 1.")
            continue

        action = sorting_mode_action

        obs, reward, done, _, info = env.step(action)

        # Split the observation vector for the current step
        input_material = obs[0:4]
        belt_material = obs[4:8]
        accuracy_belt = obs[8:12]
        sorting_material = obs[12:16]
        accuracy_sorting = obs[16:20]
        purity_diff = obs[20:24]
        container_levels = obs[24:33]

        print(f"Step {i+1} - Action: {action} -> Reward: {reward}")
        print("\n-----Next Observations-----")
        print("Input Material:", input_material)
        print("Belt Material:", belt_material)
        print("Accuracy for Belt:", accuracy_belt)
        print("Sorting Machine Material:", sorting_material)
        print("Accuracy for Sorting Machine:", accuracy_sorting)
        print("Purity Differences:", purity_diff)
        print("Container Fill Levels: [", ", ".join(f"{x:,.4f}" for x in container_levels), "]")
        
        # Display possible actions again
        env.print_possible_actions()

        env.render(save=False, log_dir="./img/", filename=f'interactive_step_{i+1}',
                   title=f'Step {i+1}', format='png')

        if done:
            print("Episode finished. Resetting environment.")
            obs, info = env.reset()
            break

# ---------------------------------------------------------*/
# Simulations with Video-Recordings
# ---------------------------------------------------------*/

def env_simulation_video(env=None, tag="", title="Environment Simulation", steps=50, dir="./img/"):

    if env is None:
        raise ValueError("Environment must be provided")

    temp_dir = f"{dir}temp/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    obs, _ = env.reset()
    print("Initial Observation:", obs)

    for i in tqdm(range(steps), disable=False):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        # print(f"Step {i+1} - Action: {action}, Observation: {obs}, Reward: {reward}")
        env.render(save=True, show=False, log_dir=temp_dir,
                   filename=f'{tag}_env_simulation_{i}', title=title, format='png')
        if done:
            env.render(save=True, show=False, log_dir=temp_dir,
                       filename=f'{tag}_env_simulation', title=title, format='png')
            obs, _ = env.reset()

    print("Creating video...")
    create_video(folder_path=temp_dir, output_path=f'{dir}{title}_{tag}.mp4')

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def model_simulation_video(model, env=None, tag="", title="Model Simulation", steps=50, dir="./img/"):

    if env is None:
        raise ValueError("Environment must be provided")

    temp_dir = f"{dir}temp/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    obs, _ = env.reset()
    print("Initial Observation:", obs)

    for i in tqdm(range(steps), disable=False):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        # Speichern des Bildes mit einem eindeutigen Namen
        env.render(save=True, show=False, log_dir=temp_dir,
                   filename=f'{tag}_env_simulation_{i}', title=title, format='png')
        if done:
            env.render(save=True, show=False, log_dir=temp_dir,
                       filename=f'{tag}_{model.__class__.__name__}_simulation_{i}', title=title, format='png')
            obs, _ = env.reset()

    print("Creating video...")
    create_video(folder_path=temp_dir, output_path=f'{dir}{title}_{tag}.mp4')

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


# ---------------------------------------------------------*/
# Run Environment Simulation
# ---------------------------------------------------------*/

if __name__ == "__main__":
    # Instantiate the environment
    env = Env_Sorting(max_steps=50, seed=42)
    # Run interactive simulation
    interactive_simulation(env=env, steps=50)


# -------------------------Notes-----------------------------------------------*\

# -----------------------------------------------------------------------------*\
