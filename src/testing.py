# ---------------------------------------------------------*\
# Title: Testing MARL
# ---------------------------------------------------------*/

# ---------------------------------------------------------*/
# Test the Environment with Random Actions
# ---------------------------------------------------------*/

def test_env(env=None, tag="", save=False, title="", steps=50, dir="./img/", video=False, seed=None, show=True, stats=True):
    """Test the environment by running a random simulation for a given number of steps.

    Returns:
        cumulative_reward (float): Total cumulative reward obtained during the test.
    """
    if env is None:
        raise ValueError("Environment must be provided")

    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0

    for i in range(steps):
        action_sort = env.action_space.sample()  # Sample a random action
        obs, reward, done, _, _ = env.step(action_sort)
        cumulative_reward += reward
        if done:
            if stats:
                print(f"🏆 Cumulative reward: {cumulative_reward:.2f}")
                env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show)
            else:
                env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show, checksum=False)
            obs, _ = env.reset(seed=seed)

    return cumulative_reward

# ---------------------------------------------------------*/
# Test the Environment with Rule-Based Actions
# ---------------------------------------------------------*/

def test_env_rule(env=None, tag="", save=False, title="", steps=50, dir="./img/", video=False, seed=None, show=True, stats=True):
    """Test the environment by running a random simulation for a given number of steps.

    Returns:
        cumulative_reward (float): Total cumulative reward obtained during the test.
    """
    if env is None:
        raise ValueError("Environment must be provided")

    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0

    for i in range(steps):
        action_sort = env.choose_action(obs)  # Choose a rule-based action
        obs, reward, done, _, _ = env.step(action_sort)
        cumulative_reward += reward
        if done:
            if stats:
                print(f"🏆 Cumulative reward: {cumulative_reward:.2f}")
                env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show)
            else:
                env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show, checksum=False)
            obs, _ = env.reset(seed=seed)

    return cumulative_reward

# ---------------------------------------------------------*/
# Test Single Model in the Environment
# ---------------------------------------------------------*/


def test_model(model, env=None, tag="", save=False, title="", steps=100, dir="./img/", seed=None, show=True, stats=True):
    """Test the model in the environment by running a simulation for a given number of steps.

    Returns:
        cumulative_reward (float): Total cumulative reward obtained during the test.
    """
    if env is None:
        raise ValueError("Environment must be provided")

    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0

    for i in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        cumulative_reward += reward
        if done:
            if stats:
                print(f"🏆 Cumulative reward: {cumulative_reward:.2f}")
                env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show)
            else:
                env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show, checksum=False)
            obs, _ = env.reset(seed=seed)

    return cumulative_reward


#---------------------------------------------------------*/
# Testing a sequence of actions in the environment
#---------------------------------------------------------*/

def test_env_sequence(env=None, sequence=None, tag="", save=False, title="", dir="./img/", seed=None, show=True, stats=True):
    """
    """
    if env is None:
        raise ValueError("Environment must be provided")
    if sequence is None:
        raise ValueError("Action sequence must be provided")
    
    # Reset Environments with Seed
    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0

    for action in sequence:
        obs, reward, done, _, _ = env.step(action)
        cumulative_reward += reward
        
    if stats:
        print(f"🏆 Cumulative reward: {cumulative_reward:.2f}")
        env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show)
    else:
        env.render(save=save, log_dir=dir, filename=f'{tag}_env_simulation', title=title,show=show, checksum=False)
        
    obs, _ = env.reset(seed=seed)
    
    return cumulative_reward


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------
