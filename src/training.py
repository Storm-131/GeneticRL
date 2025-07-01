# ---------------------------------------------------------*\
# Title: Training RL Agent
# ---------------------------------------------------------*/

import os
import copy
import time
import shutil
import numpy as np
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from typing import List

from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.data import rollout
import json

# Custom Files
from src.testing import test_model

# Import the environments
from src.env import Env_Sorting

# ---------------------------------------------------------*/
# Train a Behavioral Cloning Model
# ---------------------------------------------------------*/


def train_bc(demo_folder, env, n_epochs=100, batch_size=256):
    """
    Trains a policy using Behavioral Cloning (BC) from demonstration data.
    """
    print("-> Loading expert demonstrations...")
    transitions = load_demonstrations_bc(demo_folder, expected_obs_dim=env.observation_space.shape[0])
    if isinstance(transitions, list):
        transitions = list_to_transitions(transitions)

    print("\n----------------------------------------")
    print(f"🏋🏽‍♂️ Training BC policy from {transitions.acts.shape[0]} transitions...")
    print("----------------------------------------\n")

    device = 'cpu'
    print(f"Using device: {device}")
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        batch_size=batch_size,
        rng=np.random.RandomState(42),
        device=device
    )

    bc_trainer.train(n_epochs=n_epochs, log_interval=10000000)
    print("\n🐶 Behavioral Cloning training completed.\n")

    save_model(bc_trainer.policy, prefix="BC", timesteps=transitions.acts.shape[0])
    return bc_trainer.policy


# ---------------------------------------------------------*/
# Training the RL Agent with Optional BC Pretraining
# ---------------------------------------------------------*/
def Train_Agent(model_type, env, total_timesteps, bc_model=None, demo_folder="./models/demonstrations/", save_prefix=None, replay=False):
    """
    Trains an RL agent (PPO or DQN) on the given environment.

    If bc_model is provided, it will be used to warm-start the network weights.
    If replay is True (for DQNRB), demonstration data is loaded into the replay buffer.
    An optional save_prefix allows saving the model under a different name.
    """
    if env is None:
        raise ValueError("Environment must be provided")
    if save_prefix is None:
        save_prefix = model_type

    env = Monitor(env)
    check_env(env)

    tensorboard_log = "./log/tensorboard/"
    os.makedirs(tensorboard_log, exist_ok=True)
    time.sleep(1)

    device = 'cpu'
    print(f"Using device: {device}")

    if model_type == "PPO":
        policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0,
                    tensorboard_log=tensorboard_log, ent_coef=0.01, seed=42, device=device)
    elif model_type == "DQN":
        model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=42, device=device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if bc_model is not None:
        print("🧠 Initializing RL model with BC pre-trained weights...")
        bc_state_dict = bc_model.state_dict()
        rl_state_dict = model.policy.state_dict()

        # For debugging: print shapes for keys that exist in both dictionaries.
        for key in bc_state_dict:
            if key in rl_state_dict:
                if rl_state_dict[key].shape != bc_state_dict[key].shape:
                    print(f"Shape mismatch for {key}: BC={bc_state_dict[key].shape} vs RL={rl_state_dict[key].shape}")

        matched_keys = {k: v for k, v in bc_state_dict.items(
        ) if k in rl_state_dict and rl_state_dict[k].shape == v.shape}
        rl_state_dict.update(matched_keys)
        model.policy.load_state_dict(rl_state_dict, strict=False)

        # print("BC Model Layers:", bc_state_dict.keys())
        # print("RL Model Layers:", rl_state_dict.keys())
        print(f"✅ Loaded {len(matched_keys)} layers from BC into {model_type}.")

    if model_type == "DQN" and replay and os.path.exists(demo_folder) and os.listdir(demo_folder):
        print("📥 Loading demonstration data into replay buffer...")
        demos = load_demonstrations(demo_folder)
        for trans in demos:
            state = np.array(trans["state"])
            state = np.expand_dims(state, axis=0)
            action = np.array([trans["action"]])
            reward = np.array([float(trans["reward"])])
            next_state = np.array(trans["next_state"])
            next_state = np.expand_dims(next_state, axis=0)
            done = np.array([trans["done"]])
            model.replay_buffer.add(state, next_state, action, reward, done, infos=[{}])
        print(f"✅ Added {len(demos)} demonstration transitions to the replay buffer.")
    elif model_type == "DQN":
        print("⚠️ No demonstration data found for replay buffer. Training from scratch.")

    env_eval = copy.deepcopy(env)
    eval_env = Monitor(env_eval)
    eval_env.reset(seed=99)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./models/best_model/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=0
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    training_duration = time.time() - start_time
    print(f"✅ Training completed in {training_duration//60:.0f}m {training_duration%60:.0f}s ⏱️")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final Model Performance: {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_best_model_path = "./models/best_model/best_model.zip"
    best_model = None
    if os.path.exists(eval_best_model_path):
        best_model = model.__class__.load(eval_best_model_path, env=env)
        shutil.rmtree("./models/best_model/", ignore_errors=True)

    if best_model:
        mean_reward_best, std_reward_best = evaluate_policy(best_model, env, n_eval_episodes=10)
        print(f"Best Model Performance: {mean_reward_best:.2f} +/- {std_reward_best:.2f}")
        if mean_reward_best > mean_reward:
            print("🏅 Best model outperforms the fully trained model. Returning best model.")
            save_model(model=best_model, prefix=save_prefix, timesteps=total_timesteps)
            return best_model

    print("🏅 Returning fully trained model.")
    save_model(model=model, prefix=save_prefix, timesteps=total_timesteps)
    return model


# ---------------------------------------------------------*/
# Replay Buffer for Demonstrations
# ---------------------------------------------------------*/

def load_demonstrations(demo_folder):
    """
    """
    import os
    import json
    transitions = []
    for filename in os.listdir(demo_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(demo_folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for trans in data:
                        if isinstance(trans, dict):
                            transitions.append(trans)
                        elif isinstance(trans, list) and len(trans) >= 5:
                            transitions.append({
                                "state": trans[0],
                                "action": trans[1],
                                "reward": trans[2],
                                "next_state": trans[3],
                                "done": trans[4]
                            })
                elif isinstance(data, dict):
                    transitions.append(data)
    return transitions


def save_model(model, prefix, timesteps):
    """
    Saves the best model (or BC model) using the given prefix and timesteps.
    """
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)

    # Determine the new model file name and path
    new_filename = f"{prefix}_{timesteps}.zip"
    new_model_path = os.path.join(models_dir, new_filename)

    # Check if any file with the given prefix exists in the models_dir (excluding the "prev" folder)
    existing_models = [f for f in os.listdir(models_dir)
                       if os.path.isfile(os.path.join(models_dir, f)) and f.startswith(prefix)]

    if existing_models:
        # Create "prev" directory if it doesn't exist
        prev_dir = os.path.join(models_dir, "prev")
        os.makedirs(prev_dir, exist_ok=True)

        # Move all existing models with the given prefix to the "prev" folder
        for file in existing_models:
            old_path = os.path.join(models_dir, file)
            target_filename = file
            target_path = os.path.join(prev_dir, target_filename)
            suffix = 1
            # Append a numeric suffix if a file with the same name exists in "prev"
            while os.path.exists(target_path):
                base_name, ext = os.path.splitext(file)
                target_filename = f"{base_name}_{suffix}{ext}"
                target_path = os.path.join(prev_dir, target_filename)
                suffix += 1
            # print(f"Moving existing model {old_path} to {target_path}")
            shutil.move(old_path, target_path)

    # Save the new model
    try:
        if prefix == "BC":
            print("Saving BC policy...")
            save_bc_policy(model, new_model_path)
        elif hasattr(model, "save") and callable(getattr(model, "save")):
            model.save(new_model_path)
            print(f"Saved new {prefix} model at: {new_model_path}\n")
    except Exception as e:
        print(f"Error saving model: {e}")


def save_bc_policy(policy, save_path):
    """Saves the BC model including environment properties"""
    save_dict = {
        "state_dict": policy.state_dict(),  # Saves the model weights
        "observation_space": policy.observation_space,
        "action_space": policy.action_space
    }
    torch.save(save_dict, save_path)
    print(f"BC policy successfully saved at {save_path}")


def load_demonstrations_bc(demo_folder: str, expected_obs_dim: int = 33):
    """Load expert demonstrations from JSON files and return formatted transitions.

    Each JSON file is assumed to contain a list of dictionaries with keys "state", "action", "reward", "next_state", and "done".
    """
    trajectories: List[Trajectory] = []

    for filename in sorted(os.listdir(demo_folder)):
        if filename.endswith(".json"):
            file_path = os.path.join(demo_folder, filename)
            with open(file_path, "r") as f:
                data = json.load(f)

            obs = [step["state"] for step in data]
            acts = [step["action"] for step in data]

            # Append the final next_state as the last observation
            if "next_state" in data[-1]:
                obs.append(data[-1]["next_state"])
            else:
                raise ValueError(f"Missing 'next_state' field in the last entry of {filename}")

            obs = np.array(obs, dtype=np.float32).reshape(-1, expected_obs_dim)  # Ensure correct shape
            acts = np.array(acts, dtype=np.int64)  # Actions should be one less than observations

            if obs.shape[0] != acts.shape[0] + 1:
                raise ValueError(f"Mismatch: {obs.shape[0]} observations vs. {acts.shape[0]} actions in {filename}")

            traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=False)
            trajectories.append(traj)

    if not trajectories:
        raise ValueError("No valid trajectories found in the demonstration folder.")

    transitions = rollout.flatten_trajectories(trajectories)
    # print(f"Final Transitions: obs shape {transitions.obs.shape}, acts shape {transitions.acts.shape}")

    return transitions


def list_to_transitions(demos_list):
    """
    Converts a list of transitions (each a dict with "obs" and "acts") into a dictionary
    with keys "obs" and "acts" by stacking along axis 0.
    """
    obs = np.stack([d["obs"] for d in demos_list], axis=0)
    acts = np.stack([d["acts"] for d in demos_list], axis=0)
    return {"obs": obs, "acts": acts}


def test_agent(agent, env, name):
    """ Test the agent in the environment """
    env.reset(seed=42)
    test_model(agent, env, title=(name))

# ---------------------------------------------------------*/
# Training-Loop for the RL Agents
# ---------------------------------------------------------*/


def RL_Trainer(model_list, max_steps, total_timesteps, noise_sorting, noise_input, tag, seed,
               demo_folder="./models/demonstrations/", test_steps=None, test_dir="./img/figures/", test_save=False):
    """
    Trains models for the sorting environment based on a list of model types.

    Supported model types:
      - "BC": Behavioral Cloning policy.
      - "PPO": PPO trained from scratch.
      - "PPOBC": PPO initialized with BC pre-training.
      - "DQN": DQN trained from scratch.
      - "DQNRB": DQN trained with demonstration data replay buffer.

    The trained models are saved in the models folder using the corresponding prefix.
    """

    env = Env_Sorting(max_steps=max_steps, noise_input=noise_input, noise_sorting=noise_sorting)

    if test_steps is None:
        test_steps = max_steps

    if os.path.exists(demo_folder) and os.listdir(demo_folder):
        print("Demonstration folder found, using it for BC and RB ✅")
    else:
        print("No demonstration folder found, skipping BC and RB training ❌")

    trained_models = {}

    for algo in model_list:

        if algo == "BC":
            env.reset(seed=None)
            bc_policy = train_bc(demo_folder=demo_folder, env=env, n_epochs=100)
            test_agent(bc_policy, env, "BC")
            trained_models["BC"] = bc_policy

        elif algo == "DQN":
            print("\n----------------------------------------")
            print(f"🏋🏽‍♂️ Training {algo} from scratch...")
            print("----------------------------------------")
            env.reset(seed=None)
            agent = Train_Agent(model_type="DQN", env=env, total_timesteps=total_timesteps,
                                bc_model=None, save_prefix="DQN")
            test_agent(agent, env, "DQN")
            trained_models["DQN"] = agent

        elif algo == "DQNRB":
            print("\n----------------------------------------")
            print(f"🏋🏽‍♂️ Training {algo} with demonstration replay buffer...")
            print("----------------------------------------")
            env.reset(seed=None)
            agent = Train_Agent(model_type="DQN", env=env, total_timesteps=total_timesteps,
                                bc_model=None, save_prefix="DQNRB", replay=True)
            test_agent(agent, env, "DQNRB")
            trained_models["DQNRB"] = agent

        elif algo == "PPO":
            print("\n----------------------------------------")
            print(f"🏋🏽‍♂️ Training {algo} from scratch...")
            print("----------------------------------------")
            env.reset(seed=None)
            agent = Train_Agent(model_type="PPO", env=env, total_timesteps=total_timesteps,
                                bc_model=None, save_prefix="PPO")
            test_agent(agent, env, "PPO")
            trained_models["PPO"] = agent

        elif algo == "PPOBC":
            print("\n----------------------------------------")
            print(f"🏋🏽‍♂️ Training {algo} (initialized with BC pre-training)...")
            print("----------------------------------------")
            env.reset(seed=None)
            agent = Train_Agent(model_type="PPO", env=env, total_timesteps=total_timesteps,
                                bc_model=bc_policy, save_prefix="PPOBC")
            test_agent(agent, env, "PPOBC")
            trained_models["PPOBC"] = agent

        else:
            print(f"Unsupported model type: {algo}")

        # Remove the folder with the best model
        shutil.rmtree("./models/best_model/", ignore_errors=True)

    return trained_models


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
