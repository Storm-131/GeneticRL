# ---------------------------------------------------------*\
# Title: Environment (Master)
# ---------------------------------------------------------*/

import gymnasium as gym
import numpy as np
import random
from utils.input_generator import SeasonalInputGenerator
from utils.plotting import plot_env
from collections import Counter, deque
from gymnasium import spaces


# ---------------------------------------------------------*/
# Environment Super
# ---------------------------------------------------------*/
class Env_Sorting(gym.Env):
    """Custom Gym environment for a simple sorting system with 4 materials (A, B, C, D)
       and 5 containers. For active stations (A-D) the true sorted material is stored under the material key
       and the false-sorted material under the key "<material>_False".
       Any leftover false material (after sequential redistribution) is collected in container E.
       Container E is also considered for pressing.
    """

    def __init__(self, max_steps=50, seed=None, noise_input=0.00, noise_sorting=0.05, balesize=200, simulation=0):

        # For Training: Temporarily store the first agent
        self.train_mode = 0
        self.sort_agent = None

        # Active materials: A, B, C, D
        self.material_names = ["A", "B", "C", "D"]

        # ---------------------------------------------------------*/
        # Define the Action-Space and Observation-Space
        # ---------------------------------------------------------*/
        self._initialize_spaces()

        # Set the seed
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.set_seed(seed)

        # ---------------------------------------------------------*/
        # Sorting Machine: Agent (1)
        # ---------------------------------------------------------*/
        # Initial material compositions (equal distribution for 4 materials)
        self.current_material_input = [25, 25, 25, 25]       # Materials: A, B, C, D
        self.current_material_belt = [25, 25, 25, 25]        # On the belt
        self.current_material_sorting = [25, 25, 25, 25]     # In the sorting machine

        # Base accuracy for each material (for 4 materials)
        self.baseline_accuracy = (0.8, 0.8, 0.8, 0.8)
        self.boost = 0.5
        self.accuracy_belt = list(self.baseline_accuracy)
        self.accuracy_sorter = list(self.baseline_accuracy)
        
        # Sensor settings: two settings – 0 boosts A & C; 1 boosts B & D.
        self.sensor_current_setting = 0
        self.sensor_all_settings = [0, 1]

        self.input_occupancy = sum(self.current_material_input) / 100
        self.belt_occupancy = sum(self.current_material_belt) / 100

        # Noise levels for input and sorting accuracy
        self.noise_input = noise_input
        self.noise_accuracy = noise_sorting

        # Initialize containers:
        # For each active material, maintain two keys: one for true sorted material and one for false.
        self.container_materials = {name: 0 for name in self.material_names}
        self.container_materials.update({f"{name}_False": 0 for name in self.material_names})
        # Container E collects leftover false material.
        self.container_materials["E"] = 0

        # ---------------------------------------------------------*/
        # Presses and Bales: Agent (2)
        # ---------------------------------------------------------*/
        self.press_state = {
            "press_1": 0, "material_1": 0, "n_1": 0, "q_1": 0,
            "press_2": 0, "material_2": 0, "n_2": 0, "q_2": 0
        }
        # Include container E in bale_count so it can be pressed.
        self.bale_count = {material: [] for material in self.material_names + ["E"]}

        self.press_penalty_flag = 0
        self.bale_standard_size = balesize

        # ---------------------------------------------------------*/
        # Input Control: Agent (3)
        # ---------------------------------------------------------*/
        self.input_generator = SeasonalInputGenerator(seed=seed)

        # ---------------------------------------------------------*/
        # Reward
        # ---------------------------------------------------------*/
        self.PRESS_REWARD = 0

        # Quality thresholds for active containers (A-D)
        self.quality_thresholds = {
            "A": 0.85,
            "B": 0.80,
            "C": 0.75,
            "D": 0.70
        }
        self.container_max = {
            "A": 1000,
            "B": 1000,
            "C": 1000,
            "D": 1000
        }

        self.reward_data = {
            'Accuracy': [],
            'Setting': [],
            'Belt_Occupancy': [],
            'Reward': [],
            'Belt_Proportions': [],
        }

        self.max_steps = max_steps
        self.previous_setting = None
        self.current_step = 0

    # ---------------------------------------------------------*/
    # Helper Functions
    # ---------------------------------------------------------*/
    def _initialize_spaces(self):
        """
        Defines the extended Observation Space as a 33-dimensional vector, consisting of:
        - 4 values: Input Material (without noise)
        - 4 values: Belt Material
        - 4 values: Accuracy for the Belt Material
        - 4 values: Sorting Machine Material
        - 4 values: Accuracy for the Sorting Machine
        - 4 values: Purity Differences for Containers A-D
        - 9 values: Container fill levels
            • For Containers A-D: 2 values each (true and false) -> 8 values
            • For Container E: 1 value (alone, as only false material)
        """
        low = np.concatenate((
            np.zeros(4),           # Input Material
            np.zeros(4),           # Belt Material
            np.zeros(4),           # Accuracy for Belt Material (0 to 1)
            np.zeros(4),           # Sorting Machine Material
            np.zeros(4),           # Accuracy for Sorting Machine (0 to 1)
            np.full(4, -1.0),        # Purity Differences (in [-1,1])
            np.zeros(8),           # Containers A-D (true and false)
            np.zeros(1)            # Container E
        ))
        high = np.concatenate((
            np.ones(4),     # Input Material (assumed maximum value)
            np.ones(4),     # Belt Material
            np.ones(4),     # Accuracy for Belt Material (0 to 1)
            np.ones(4),     # Sorting Machine Material
            np.ones(4),     # Accuracy for Sorting Machine (0 to 1)
            np.ones(4),     # Purity Differences (in [-1,1])
            np.ones(8),     # Containers A-D (true and false), assumed maximum values
            np.ones(1),     # Container E
        ))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def set_seed(self, seed):
        """Set seed for the environment and its components."""
        self.seed = seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        else:
            self.rng = np.random.default_rng()
            np.random.seed()
            random.seed()
            random_seed = np.random.randint(0, 10000)
            self.action_space.seed(random_seed)
            self.observation_space.seed(random_seed)

    def compute_input_proportions(self):
        """Compute the proportions of the current input materials (A-D)."""
        current_input = self.current_material_input
        total_input_amount = sum(current_input)
        proportions = {}
        if total_input_amount > 0:
            for material, amount in zip(self.material_names, current_input):
                proportions[material] = amount / total_input_amount
        else:
            for material in self.material_names:
                proportions[material] = 0
        return proportions

    def compute_belt_proportions(self):
        """Compute the proportions of materials on the belt (A-D)."""
        current_belt = self.current_material_belt
        total_belt_amount = sum(current_belt)
        proportions = {}
        if total_belt_amount > 0:
            for material, amount in zip(self.material_names, current_belt):
                proportions[material] = amount / total_belt_amount
        else:
            for material in self.material_names:
                proportions[material] = 0
        return proportions

    def print_possible_actions(self):
        """Print possible actions for the sorting agent."""
        print("\nPossible actions for the Sorting Agent:")
        print("Sensor Settings: [0, 1]")
        print(" - 0: Boost (+25%) Accuracy for Materials A and C")
        print(" - 1: Boost (+25%) Accuracy for Materials B and D")
        print("\nSorting Process: Materials are sequentially processed at stations A, B, C, and D.")
        print("  The true sorted material is stored in container (A, B, C, D),")
        print("  while the false sorted material is stored in the respective False container.")
        print("  Any leftover false material is collected into container E.")

    def compute_purity_differences(self, scaling_factor=1):
        """
        Compute purity deviations from quality thresholds for active containers (A-D).
        """
        container_purities, _ = self.get_container_purity()
        purity_differences = []

        for mat in self.material_names:
            purity = container_purities.get(mat, 0)
            threshold = self.quality_thresholds[mat]
            diff = purity - threshold
            if diff < 0:
                diff *= scaling_factor
            purity_differences.append(round(diff, 2))

        return purity_differences

    # ---------------------------------------------------------*/
    # Gym Environment Functions
    # ---------------------------------------------------------*/
    def reset(self, seed=None):

        if seed is not None:
            self.set_seed(seed)
            self.input_generator.set_seed(seed)

        self.current_material_input = [25, 25, 25, 25]
        self.current_material_belt = [25, 25, 25, 25]
        self.current_material_sorting = [25, 25, 25, 25]
        self.container_materials = {name: 0 for name in self.material_names}
        self.container_materials.update({f"{name}_False": 0 for name in self.material_names})
        self.container_materials["E"] = 0

        self.press_state = {
            "press_1": 0, "material_1": 0, "n_1": 0, "q_1": 0,
            "press_2": 0, "material_2": 0, "n_2": 0, "q_2": 0
        }
        self.last_two_actions = deque(maxlen=2)
        self.input_history = {material: deque(maxlen=10) for material in self.material_names}
        self.accuracy_belt = list(self.baseline_accuracy)
        self.accuracy_sorter = list(self.baseline_accuracy)
        self.sensor_current_setting = 0
        self.input_occupancy = sum(self.current_material_input) / 100
        self.belt_occupancy = sum(self.current_material_belt) / 100
        self.bale_count = {material: [] for material in self.material_names + ["E"]}
        self.current_step = 0
        self.previous_setting = None
        self.reward_data = {
            'Accuracy': [],
            'Setting': [],
            'Belt_Occupancy': [],
            'Reward': [],
            'Belt_Proportions': [],
        }
        obs_sort = self.get_obs()
        return obs_sort, {}

    def step(self, action):
        """
        Execute one step:
          1. Sort current material batch.
          2. Update environment.
          3. Set sensor mode.
          4. Update accuracy, check press status, and press if needed.
          5. Calculate reward and return new observation.
        """
        self.sort_material()
        self.update_environment()
        self.set_multisensor_mode(action)
        self.update_accuracy()

        self.check_press_status()       # Press Bale if timer is done
        self.check_container_level()    # Use press if container is full and press is empty

        rew_total, _, _ = self.calculate_reward()
        obs_sort = self.get_obs()
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return obs_sort, rew_total, done, False, {}

    def update_environment(self):
        """Update environment by processing a batch of materials."""
        self.current_material_sorting = self.current_material_belt.copy()
        self.current_material_belt = self.current_material_input.copy()
        self.belt_occupancy = self.input_occupancy
        input_batch = self.input_generator.generate_input()
        material_counts = Counter(input_batch)
        a = material_counts.get('A', 0)
        b = material_counts.get('B', 0)
        c = material_counts.get('C', 0)
        d = material_counts.get('D', 0)
        total_units = len(input_batch)
        self.current_material_input = [a, b, c, d]
        self.input_occupancy = round(total_units / 100, 2)
        self.accuracy_sorter = self.accuracy_belt.copy()
        for i, material in enumerate(self.material_names):
            self.input_history[material].append(self.current_material_input[i])

    def get_obs(self):
        """
        Returns an observation vector (dimension: 33) containing the following components:
        1. Input Material (without noise) for A-D (4 values)
        2. Belt Material for A-D (4 values)
        3. Accuracy for the Belt Material for A-D (4 values)
        4. Sorting Machine Material for A-D (4 values)
        5. Accuracy for the Sorting Machine for A-D (4 values)
        6. Purity Differences for Containers A-D (4 values)
        7. Container fill levels:
            - For Containers A-D: true and false separately (total 8 values)
            - For Container E: one value (1 value)
        """
        # 1. Input Material (without noise)
        input_material = np.array(self.current_material_input, dtype=np.float32) / 1000  # 4 values

        # 2. Belt Material
        belt_material = np.array(self.current_material_belt, dtype=np.float32) / 1000  # 4 values

        # 3. Accuracy for Belt Material
        accuracy_belt = np.array(self.accuracy_belt, dtype=np.float32)            # 4 values

        # 4. Sorting Machine Material
        sorting_material = np.array(self.current_material_sorting, dtype=np.float32) / 1000  # 4 values

        # 5. Accuracy for Sorting Machine
        accuracy_sorting = np.array(self.accuracy_sorter, dtype=np.float32)         # 4 values

        # 6. Purity Differences for Containers A-D
        purity_diff = np.array(self.compute_purity_differences(scaling_factor=1.0), dtype=np.float32)  # 4 values

        # 7. Container fill levels:
        # For Containers A-D: true and false separately
        container_levels = []
        for material in self.material_names:
            container_levels.append(self.container_materials[material] / 1000)             # true
            container_levels.append(self.container_materials[f"{material}_False"] / 1000)  # false
        # For Container E: only one value (since only false material is collected)
        container_levels.append(self.container_materials["E"] / 1000)
        container_levels = np.array(container_levels, dtype=np.float32)  # 8+1 = 9 values

        # Concatenating the complete observation vector (4+4+4+4+4+4+9 = 33 dimensions)
        observation = np.concatenate((
            input_material,
            belt_material,
            accuracy_belt,
            sorting_material,
            accuracy_sorting,
            purity_diff,
            container_levels
        ))
        return observation

    def render(self, mode='human', save=False, show=True, log_dir='./img/log', filename='plot', title='', format="svg", checksum=True):
        """Render the environment."""
        plot_env(self.current_material_input, self.current_material_belt, self.current_material_sorting,
                 self.container_materials, self.accuracy_belt, self.accuracy_sorter, self.sensor_current_setting, self.reward_data,
                 self.belt_occupancy, self.press_state, self.bale_count, self.bale_standard_size,
                 save=save, show=show, log_dir=log_dir, filename=filename, title=title, format=format,
                 quality_thresholds=self.quality_thresholds, checksum=checksum)

    def choose_action(self, obs):
        """
        Rule-based action: compare totals for groups A+C and B+D.
        """
        inputs = obs[:4]
        sum_AC = inputs[0] + inputs[2]
        sum_BD = inputs[1] + inputs[3]
        mode = 0 if sum_AC >= sum_BD else 1
        return mode

    # ---------------------------------------------------------*/
    # Sorting Agent (1)
    # ---------------------------------------------------------*/
    def set_multisensor_mode(self, new_sensor_setting):
        """Set sensor mode (0 or 1)."""
        if new_sensor_setting in [0, 1]:
            self.previous_setting = self.sensor_current_setting
            self.sensor_current_setting = new_sensor_setting
        else:
            raise ValueError(f"Invalid sensor setting: {new_sensor_setting}. Must be 0 or 1.")

    def update_accuracy(self):
        """Update accuracy based on sensor mode and input amounts."""
        accuracy = list(self.baseline_accuracy)
        total_input_amount = sum(self.current_material_input)
        reduction_factor = ((total_input_amount / 100) ** 2) * 0.2

        for i in range(len(accuracy)):
            accuracy[i] -= reduction_factor
        if self.sensor_current_setting == 0:
            for i, material in enumerate(self.material_names):
                if material in ["A", "C"]:
                    accuracy[i] += self.boost
        elif self.sensor_current_setting == 1:
            for i, material in enumerate(self.material_names):
                if material in ["B", "D"]:
                    accuracy[i] += self.boost
        accuracy = np.clip(np.array(accuracy), 0.0, 1.0)

        if self.noise_accuracy != 0:
            noise = self.rng.uniform(-self.noise_accuracy, 0, size=len(accuracy))
            accuracy = [acc * (1 + n) for acc, n in zip(accuracy, noise)]
        self.accuracy_belt = accuracy

    def sort_material(self):
        """
        Process each active station (A-D) sequentially.
        For each station i:
        - target_amount = current input.
        - true_material = int(round(target_amount * accuracy)).
        - false_material = target_amount - true_material.
        - Remove the input from station i.
        - Redistribute false_material from other stations using an iterative probability-based approach.
        - Store computed true_material and false_material in the corresponding containers.
        After processing, any remaining discrepancy is added to container E.
        """

        # Store initial material count before sorting
        total_input = sum(self.current_material_sorting)

        # Initialize tracking arrays
        leftover = np.array(self.current_material_sorting, dtype=int)
        true_arr = np.zeros_like(leftover)
        false_arr = np.zeros_like(leftover)

        for i in range(len(self.material_names)):
            target_amount = leftover[i]
            acc = self.accuracy_sorter[i]

            # Compute sorted (true) and mis-sorted (false) materials
            true_val = int(round(target_amount * acc))
            false_val = target_amount - true_val

            true_arr[i] = true_val
            false_arr[i] = false_val

            # Remove input from the current station
            leftover[i] = false_val

            # Identify other indices for redistribution
            other_indices = list(range(len(self.material_names)))
            distributed = np.zeros(len(other_indices), dtype=int)

            # Iteratively distribute false material unit by unit
            for _ in range(false_val):
                available = leftover[other_indices]  # Current available material in each container
                total_avail = available.sum()

                if total_avail == 0:
                    # No more material available for redistribution
                    break

                # Compute probability distribution for redistribution
                pvals = available / total_avail
                selected = self.rng.choice(len(other_indices), p=pvals)

                # Ensure that we do not remove more than available
                if leftover[other_indices[selected]] > 0:
                    leftover[other_indices[selected]] -= 1
                    distributed[selected] += 1
                else:
                    raise ValueError(f"Container {other_indices[selected]} has insufficient material "
                                     f"(available: {leftover[other_indices[selected]]}).")

            # Validate that all false materials have been redistributed
            if distributed.sum() != false_val:
                diff = false_val - distributed.sum()
                print(f"Warning: Iterative redistribution failed to distribute {diff} units.")

        # Remaining leftover material is moved to container E
        e_input = sum(leftover)

        # Ensure material conservation: Adjust discrepancies if necessary
        total_output = sum(true_arr) + sum(false_arr) + e_input
        discrepancy = total_input - total_output

        if discrepancy != 0:
            if discrepancy == -1:
                e_input -= 1  # Minor correction for rounding issues
            else:
                print(f"DISCREPANCY: {discrepancy}")
                print(f"Total Input: {total_input}")
                print(f"Total Output: True: {sum(true_arr)}, False: {sum(false_arr)}, E: {e_input}")
                raise ValueError(f"Detected material loss or gain! Input: {total_input}, Output: {total_output}")

        # Store leftover material in container E
        self.container_materials["E"] += e_input

        # Update material containers with sorted and mis-sorted materials
        for i, material in enumerate(self.material_names):
            self.container_materials[material] += true_arr[i]
            self.container_materials[f"{material}_False"] += false_arr[i]

        # Compute and return mean purity
        mean_purity = round(1 - ((total_input - sum(true_arr)) / total_input), 2) if total_input > 0 else 0
        return mean_purity

    def get_belt_purity(self):
        """Calculate belt error rate based on true vs. false sorting."""
        materials = self.current_material_belt
        accuracies = self.accuracy_belt
        true_sorted = [materials[i] * accuracies[i] for i in range(len(materials))]
        false_sorted = [materials[i] - true_sorted[i] for i in range(len(materials))]
        total_material = sum(materials)
        total_false = sum(false_sorted)
        error_rate = total_false / total_material if total_material > 0 else 0
        return error_rate

    # ---------------------------------------------------------*/
    # Container Agent (2)
    # ---------------------------------------------------------*/

    def check_press_status(self):
        """Update press state and check timer -> produce bales when a press finishes."""
        for i in [1, 2]:  # Beide Pressen unabhängig prüfen
            if self.press_state[f"press_{i}"] > 0:
                self.press_state[f"press_{i}"] -= 1  # Zeit reduzieren

                if self.press_state[f"press_{i}"] == 0:
                    material = self.press_state[f"material_{i}"]
                    n = self.press_state[f"n_{i}"]
                    q = self.press_state[f"q_{i}"]

                    self.press_bale(material, n, q)

                    # **Fix: Presse korrekt zurücksetzen**
                    self.press_state[f"press_{i}"] = 0
                    self.press_state[f"material_{i}"] = 0
                    self.press_state[f"n_{i}"] = 0
                    self.press_state[f"q_{i}"] = 0

    def press_bale(self, material, n, q):
        """Press a bale for the given container."""
        q = int(q * 100)  # Qualität speichern
        bales = self.bale_count[material]

        full_bales = n // self.bale_standard_size
        remaining_material = n % self.bale_standard_size

        # **Fix: Unterscheide zwischen voller und Restmenge**
        for _ in range(full_bales):
            bales.append((self.bale_standard_size, q))

        if remaining_material > 0:
            if remaining_material > self.bale_standard_size / 2:
                bales.append((remaining_material, q))  # Neuer Ballen
            else:
                if bales:
                    last_bale_size, last_bale_quality = bales[-1]
                    bales[-1] = (last_bale_size + remaining_material, last_bale_quality)  # Rest addieren
                else:
                    bales.append((remaining_material, q))

        self.bale_count[material] = bales
        return self.bale_count

    def check_container_level(self):
        """
        Check if any container (active containers A-D and container E) exceeds the bale standard size.
        For active containers, volume = true + false; for container E, volume = its content.
        Returns an available press and the container key.
        """
        total_volumes = {}
        for material in self.material_names:
            vol = self.container_materials[material] + self.container_materials[f"{material}_False"]
            total_volumes[material] = vol
        total_volumes["E"] = self.container_materials["E"]
        highest_material = max(total_volumes, key=total_volumes.get)
        highest_volume = total_volumes[highest_material]

        if highest_volume > self.bale_standard_size:
            for press in [1, 2]:
                press_key = f"press_{press}"
                if self.press_state[press_key] == 0:
                    self.use_press(press, highest_material)
                    return press, highest_material

        return None, None

    def use_press(self, press, material):
        """Use a press on the specified container."""
        if self.press_state[f"press_{press}"] > 0:
            self.press_penalty_flag = 1  # Block if the press is still running
            return

        # Calculate the total material (True + False)
        total_material = self.container_materials[material] + self.container_materials.get(f"{material}_False", 0)

        # **Fix: Set quality for container E**
        if material in self.material_names:
            true_material = self.container_materials[material]
            false_material = self.container_materials[f"{material}_False"]
            total = true_material + false_material
            quality = round(true_material / total, 2) if total > 0 else 0
        else:
            # **Container "E" has a fixed quality**
            total = total_material
            quality = 0

        # **Empty the container**
        self.container_materials[material] = 0
        if material in self.material_names:
            self.container_materials[f"{material}_False"] = 0

        # Calculation of press times
        press_time = 10  # Uniform calculation

        # **Fix: Keep distinction between presses**
        self.press_state[f"press_{press}"] = press_time
        self.press_state[f"material_{press}"] = material
        self.press_state[f"n_{press}"] = total_material  # No more accumulation!
        self.press_state[f"q_{press}"] = quality

    def get_container_purity(self):
        """
        Calculates the purity for each active container (A-D) as the ratio:
        true_material / (true_material + false_material)
        If a container is empty, its purity is set to the corresponding threshold.
        Returns a dictionary of purities and the global average purity.
        """
        container_purities = {}
        for mat in self.material_names:
            true_val = self.container_materials.get(mat, 0)
            false_val = self.container_materials.get(f"{mat}_False", 0)
            total = true_val + false_val
            if total > 0:
                purity = true_val / total
            else:
                purity = self.quality_thresholds[mat]  # Set purity to threshold if empty
            container_purities[mat] = round(purity, 2)
        global_purity = sum(container_purities.values()) / len(container_purities) if container_purities else 0
        return container_purities, round(global_purity, 2)

    def set_container_purity(self, material, purity):
        """For analysis: set purity for a given container."""
        total_volume = self.bale_standard_size
        correct_amount = purity * total_volume
        self.container_materials[material] = correct_amount

    # ---------------------------------------------------------*/
    # Control Agent (3) - "Throughput"
    # ---------------------------------------------------------*/
    def set_occupation_level(self, occupation_level, distribution=None):
        """Set input materials based on desired occupation level (0-100)."""
        if distribution is None:
            raise ValueError("Distribution has to be given.")
        if not np.isclose(sum(distribution.values()), 1.0):
            raise ValueError("The sum of the distribution must be 1 (100%).")
        fraction = occupation_level / 100.0
        total_material = fraction * 100
        self.current_material_input = [total_material * distribution[material] for material in self.material_names]
        self.input_occupancy = sum(self.current_material_input) / 100
        self.current_distribution = distribution

    # ---------------------------------------------------------*/
    # Reward for Agent
    # ---------------------------------------------------------*/
    def calculate_reward(self):
        """Calculate total reward from sorting, pressing, and input control."""
        avg_accuracy = sum(self.accuracy_belt) / len(self.accuracy_belt)
        rew_sort, rew_sort_list = self.calculate_sorting_reward()

        rew_press = self.calculate_press_reward()
        rew_input = self.calculate_input_reward()

        rew_total = round(rew_sort + rew_press + rew_input, 2)

        reward_tuple = (rew_sort, rew_press, rew_input)

        proportions = self.compute_belt_proportions()
        self.reward_data['Reward'].append(reward_tuple)
        self.reward_data['Accuracy'].append(avg_accuracy)
        self.reward_data['Setting'].append(self.sensor_current_setting)
        self.reward_data['Belt_Occupancy'].append(self.belt_occupancy)
        self.reward_data['Belt_Proportions'].append(proportions)
        return rew_total, reward_tuple, rew_sort_list

    def calculate_sorting_reward(self):
        """
        Sorting reward is the sum of purity differences for active containers (A-D), with negative differences weighted.
        """
        reward_list = self.compute_purity_differences(scaling_factor=5)
        total_reward = sum(reward_list)
        return total_reward, reward_list

    def calculate_press_reward(self):
        """Calculate reward for the press agent."""
        rew_press = 0
        return round(rew_press, 2)

    def calculate_input_reward(self):
        """Calculate reward for the control agent."""
        rew_input = 0
        return round(rew_input, 2)


# -------------------------Notes-----------------------------------------------*\

# -----------------------------------------------------------------------------*/