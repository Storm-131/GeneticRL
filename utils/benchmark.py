# ---------------------------------------------------------*\
# Title: Benchmark
# ---------------------------------------------------------*/

import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


class RLBenchmark:
    def __init__(self, models, total_timesteps=100_000, n_eval_episodes=10, seed=42, train_env=None,
                 eval_env=None, tag="", agent_model=None):
        self.models = models
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.results = {}
        self.tag = tag
        self.agent_model = agent_model

        if train_env is not None and eval_env is not None:
            self.train_env = train_env
            self.eval_env = eval_env
        else:
            raise ValueError("Environment must be provided")

    def run_benchmark(self, dir="./img/"):
        """Run benchmark for all models.
        1. Test the environment
        2. Train the models
        3. Evaluate the models
        4. Plot the benchmark results
        """

        # 1) Random Agent (Test)
        train_env = self.wrap_env(self.train_env)
        mean_reward, std_reward = self.evaluate_model(None)

        self.results['Test'] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward
        }

        for model_type in self.models:
            # 2) RBA (Rule-Based Agent)
            if model_type == "RBA":
                mean_reward, std_reward = self.evaluate_rule_based_agent()
                self.results['RBA'] = {
                    'mean_reward': mean_reward,
                    'std_reward': std_reward
                }
            # 3) RL Models
            else:
                print(f"Training {model_type}...")
                model = self.train_model(model_type)
                print(f"Evaluating {model_type}...")
                mean_reward, std_reward = self.evaluate_model(model)
                self.results[model_type] = {
                    'mean_reward': mean_reward,
                    'std_reward': std_reward
                }

        # Save results to a file
        with open(f"./log/results_bm_{self.tag}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(['Model Type', 'Mean Reward', 'Standard Deviation Reward'])
            for key, value in self.results.items():
                writer.writerow([key, value['mean_reward'], value['std_reward']])

        self.plot_benchmark(dir=dir)

    def wrap_env(self, env):
        """Wrap environment with a Monitor and check it."""
        env.reset(seed=100)
        env = Monitor(env)
        check_env(env)
        return env

    def get_model(self, model_type, env, tensorboard_log):
        """Return a model based on the model type."""
        if model_type == "PPO":
            return PPO("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed,
                       gamma=0.85, learning_rate=0.0007, ent_coef=0.03)
        elif model_type == "DQN":
            return DQN("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, gamma=0.95)
        elif model_type == "A2C":
            return A2C("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log, seed=self.seed, ent_coef=0.06, 
                       gamma=0.9, learning_rate=0.0005)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train_model(self, model_type):
        """Train a model on the stochastic Sorting Environment ("no seed")"""
        env = self.wrap_env(self.train_env)

        tensorboard_log = f"./log/tensorboard/{model_type}/"
        os.makedirs(tensorboard_log, exist_ok=True)

        model = self.get_model(model_type, env, tensorboard_log)
        model.learn(total_timesteps=self.total_timesteps, progress_bar=True)

        model_path = f"./models/{model_type.lower()}_sorting_env_{self.tag}"
        model.save(model_path)
        model = model.__class__.load(model_path, env=env)

        return model

    def evaluate_model(self, model):
        """Evaluate a model over n episodes and give back the mean and std of the rewards."""
        rewards = []

        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.seed + episode)

            total_reward = 0
            done = False

            while not done:
                if model is None:
                    action = self.train_env.action_space.sample()
                else:
                    action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, _, _ = self.eval_env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        return np.mean(rewards), np.std(rewards)

    def evaluate_rule_based_agent(self):
        """Evaluate the Rule-Based Agent over n episodes and give back the mean and std of the rewards."""
        rewards = []
        env = self.train_env
        env.reset(seed=100)
        agent = self.agent_model(env)
        agent.run_analysis()

        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.seed + episode)

            total_reward = 0
            done = False

            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.eval_env.step(action)
                total_reward += reward

            rewards.append(total_reward)

        return np.mean(rewards), np.std(rewards)

    def plot_benchmark(self, dir="./img/"):
        """Plot and save the benchmark results."""
        models = list(self.results.keys())
        total_rewards_mean = [self.results[model]['mean_reward'] for model in models]
        total_rewards_std = [self.results[model]['std_reward'] for model in models]

        len_models = np.arange(len(models))

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Model')
        ax1.set_ylabel(f'Mean Reward over {self.n_eval_episodes} episodes')
        bars = ax1.bar(len_models, total_rewards_mean, yerr=total_rewards_std, alpha=0.8)
        ax1.set_xticks(len_models)
        ax1.set_xticklabels(models)
        ax1.set_title(f'Model Performance: Mean Reward ({self.tag} env)')

        # Add actual values at the bottom of the bars
        for bar, mean_reward in zip(bars, total_rewards_mean):
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, 3, round(mean_reward, 2), ha='center', va='bottom')

        os.makedirs(dir, exist_ok=True)
        plt.savefig(f'{dir}model_performance_{self.tag}.svg', format='svg', bbox_inches='tight')
        plt.show()


# -------------------------Notes-----------------------------------------------*\
# Perform a benchmark on the Sorting Environment using Stable Baselines3.
# -----------------------------------------------------------------------------*\
