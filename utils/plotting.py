# ---------------------------------------------------------*\
# Title: Plotting
# ---------------------------------------------------------*/

import cv2
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import re

# ---------------------------------------------------------*/
# Global font increase setting
# ---------------------------------------------------------*/
font_increase = 0  # Adjust this value to increase or decrease the font size globally

# ---------------------------------------------------------*/
# Plot the current state of the sorting environment
# ---------------------------------------------------------*/


def plot_env(material_composition, current_material_belt, current_material_sorting, container_materials, accuracy,
             prev_accuracy, sensor_setting, reward_data, belt_occupancy, press_state, bale_count, bale_standard_size,
             save=True, show=True, log_dir='./img/log/', filename='marl_system_diagram', sorting_mode=False, title="",
             format="svg", quality_thresholds=None, checksum=True):

    sns.set_theme(style="whitegrid")

    # Define colors for the bar segments (Container E represents the "Rest")
    colors = {
        'A': 'lightblue',
        'A_False': 'lightgray',
        'B': 'lightgreen',
        'B_False': 'lightgray',
        'C': 'lightcoral',
        'C_False': 'lightgray',
        'D': 'lightpink',
        'D_False': 'lightgray',
        'E': 'lightsalmon',       # Displayed as "Rest" in the container plot
        'E_False': 'lightgray',
        'Other': 'grey'
    }

    # Generate the figure and axes
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"MARL System {title}", fontsize=20, fontweight='bold')

    # Generate the subplots
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((3, 4), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((3, 4), (0, 2), colspan=1)
    ax4 = plt.subplot2grid((3, 4), (0, 3), colspan=1)
    ax5 = plt.subplot2grid((3, 4), (1, 0), colspan=3)
    ax6 = plt.subplot2grid((3, 4), (1, 3), colspan=1)
    ax7 = plt.subplot2grid((3, 4), (2, 0), colspan=1)
    ax8 = plt.subplot2grid((3, 4), (2, 1), colspan=1)
    ax9 = plt.subplot2grid((3, 4), (2, 2), colspan=1)
    ax10 = plt.subplot2grid((3, 4), (2, 3), colspan=1)

    # ---------------------------------------------------------*/
    # 1) Input Material Composition (4 Materials: A, B, C, D)
    # ---------------------------------------------------------*/
    ax1.set_title('Input', fontweight='bold', fontsize=12)
    total = sum(material_composition)
    ax1.pie(material_composition, labels=['A', 'B', 'C', 'D'],
            autopct=lambda p: f'{round(p * total / 100)}',
            colors=[colors['A'], colors['B'], colors['C'], colors['D']],
            textprops={'fontsize': 10})
    ax1.text(0.02, 0.98, f'Total: {total}', transform=ax1.transAxes, ha='left',
             va='top', fontweight='bold', fontsize=10)

    # ---------------------------------------------------------*/
    # 2) Conveyor Belt Material (4 Materials)
    # ---------------------------------------------------------*/
    ax2.set_title('Conveyor Belt', fontweight='bold', fontsize=12)
    bars = ax2.bar(['A', 'B', 'C', 'D'], current_material_belt,
                   color=[colors['A'], colors['B'], colors['C'], colors['D']])
    ax2.text(0.02, 0.98, f'Total: {sum(current_material_belt)}',
             transform=ax2.transAxes, ha='left', va='top', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Quantity', fontsize=10)
    ax2.set_ylim([0, 100])
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    for bar, value in zip(bars, current_material_belt):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, str(value),
                 ha='center', va='bottom', color='black', fontsize=10)

    # ---------------------------------------------------------*/
    # 3) Next Sorting (Belt Status)
    # ---------------------------------------------------------*/
    ax3.set_title('Current Accuracies, Occupation and Setting', fontweight='bold', fontsize=12)
    light_purple = '#BF40BF'  # RGB

    # Define hatch patterns for materials
    hatch_pattern = ['..', '/']

    # Erwartet werden 4 Genauigkeitswerte, gefolgt von Occupancy und Sensor Setting
    accuracy = np.array(accuracy)
    belt_occupancy = np.array([belt_occupancy])  # (1,)
    sensor_setting = np.array([sensor_setting])  # (1,)
    combined_data = np.concatenate((accuracy, belt_occupancy, sensor_setting))

    # Labels: 4 Materialien, Occupancy und Setting
    bars = ax3.bar(['A', 'B', 'C', 'D', 'Occ', 'Set'],
                   combined_data, color=[colors['A'], colors['B'], colors['C'], colors['D'], 'red', light_purple])

    # Apply hatch patterns
    for i, bar in enumerate(bars[:4]):
        bar.set_hatch(hatch_pattern[0])
    for i, bar in enumerate(bars[4:]):
        bar.set_hatch(hatch_pattern[1])

    ax3.set_ylim([0, 1.1])
    ax3.set_yticks(np.arange(0, 1.1, 0.2))
    ax3.set_ylabel('Percentage (/100)', fontsize=10)
    ax3.tick_params(axis='x', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)

    for i, bar in enumerate(bars):
        if i < len(bars) - 1:
            formatted_value = f"{combined_data[i]:.2f}"
            ax3.text(bar.get_x() + bar.get_width() / 2, combined_data[i], formatted_value,
                     ha='center', va='bottom', fontsize=10)

    # ---------------------------------------------------------*/
    # 4) Sorting Machine Material (4 Materials)
    # ---------------------------------------------------------*/
    ax4.set_title('Sorting Machine', fontweight='bold', fontsize=12)
    bars = ax4.bar(['A', 'B', 'C', 'D'], current_material_sorting,
                   color=[colors['A'], colors['B'], colors['C'], colors['D']])
    ax4.text(0.02, 0.98, f'Total: {sum(current_material_sorting)}',
             transform=ax4.transAxes, ha='left', va='top', fontweight='bold', fontsize=10)
    ax4.set_ylim([0, 100])
    ax4.tick_params(axis='x', labelsize=10)
    ax4.tick_params(axis='y', labelsize=10)

    for i, bar in enumerate(bars):
        accuracy_val = round(prev_accuracy[i], 2)
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'acc: {accuracy_val}', ha='center', va='bottom', fontsize=10)
        ax4.text(bar.get_x() + bar.get_width() / 2, 0, str(current_material_sorting[i]),
                 ha='center', va='bottom', color='black', fontsize=10)

    # ---------------------------------------------------------*/
    # 5) Relative Material Proportions and Sorting Mode
    # ---------------------------------------------------------*/

    # Switch for alternative representation (True = squares, False = dots)
    use_square_representation = True

    # Colors for standard mode representation
    mode_colors = {
        0: ('lightblue', 'lightcoral'),
        1: ('lightgreen', 'lightpink')
    }

    # Colors for alternative square representation
    square_colors = {
        0: ('green', 's'),  # Green, square
        1: ('red', 's')     # Red, square
    }

    ax5.set_title('Sorting Reward Metrics (Belt Status)', fontweight='bold', fontsize=12)
    x_axe = range(len(reward_data['Reward']))

    # Plot Accuracy and Occupancy
    ax5.plot(x_axe, reward_data['Accuracy'], label='Accuracy (avg.)', color='blue', linewidth=2, alpha=0.6)
    ax5.plot(x_axe, reward_data['Belt_Occupancy'], label='Occupancy', color='red', linewidth=2, alpha=0.6)

    # Proportions from reward_data (only A, B, C, D)
    materials = ['A', 'B', 'C', 'D']
    proportions = {material: [] for material in materials}
    for timestep_proportions in reward_data['Belt_Proportions']:
        for material in materials:
            proportions[material].append(timestep_proportions.get(material, 0))

    for material in materials:
        ax5.plot(x_axe, proportions[material], label=material, color=colors[material], linewidth=1)

    # Legend elements for the alternative representation
    legend_elements = []

    # Plot sorting modes as dots or squares
    for i, mode in enumerate(reward_data['Setting']):
        mode = int(mode)  # Convert mode to integer
        
        if use_square_representation:
            # Alternative representation with squares
            color, marker = square_colors[mode]
            ax5.scatter(i, -0.06, color=color, marker=marker, s=40, edgecolor='black', linewidth=0.5)

            # Add to legend only once
            if not any(l.get_label() == f"Action {mode}" for l in legend_elements):
                legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', label=f"Action {mode}",
                                                markerfacecolor=color, markersize=8, markeredgecolor='black'))
        else:
            # Standard representation with two dots and number
            if mode in mode_colors:
                top_color, bottom_color = mode_colors[mode]
                ax5.scatter(i, -0.04, color=top_color, marker='o', s=40, edgecolor='black', linewidth=0.5)
                ax5.scatter(i, -0.08, color=bottom_color, marker='o', s=40, edgecolor='black', linewidth=0.5)
                ax5.text(i, -0.12, str(mode), ha='center', va='top', fontsize=8, fontweight='bold', color='black')

    # Retrieve existing legend handles and labels
    handles, labels = ax5.get_legend_handles_labels()

    # Add alternative representation (squares) to the legend if enabled
    if use_square_representation:
        handles.extend(legend_elements)
        labels.extend([l.get_label() for l in legend_elements])

    # Apply updated legend
    ax5.legend(handles, labels, loc='upper left', fontsize=8)

    ax5.set_xlim([0, None])
    ax5.set_ylim([-0.1, 1.1])
    ax5.set_xlabel('Timesteps', fontsize=10)
    ax5.set_ylabel('Proportion', fontsize=10)
    ax5.tick_params(axis='x', labelsize=10)
    ax5.tick_params(axis='y', labelsize=10)
    ax5.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%d' % (x)))
    ax5.yaxis.set_ticks(np.arange(-0.2, 1.201, 0.2))
    ax5.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax5.set_aspect('auto')

    mean_accuracy = np.mean(reward_data['Accuracy']) * 100
    mean_occupancy = np.mean(reward_data['Belt_Occupancy']) * 100
    ax5.text(0.98, 0.98, f'Mean Accuracy: {mean_accuracy:.0f}%\nMean Occupancy: {mean_occupancy:.0f}%',
            horizontalalignment='right', verticalalignment='top', transform=ax5.transAxes, fontweight='bold',
            fontsize=8, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    # ---------------------------------------------------------*/
    # 6) Current Rewards Plot
    # ---------------------------------------------------------*/
    ax6.set_title('Current Rewards', fontweight='bold', fontsize=12)
    sorting_rewards = [reward[0] for reward in reward_data['Reward']]
    pressing_rewards = [reward[1] for reward in reward_data['Reward']]
    input_rewards = [reward[2] for reward in reward_data['Reward']]
    x_axe = range(len(sorting_rewards))
    ax6.plot(x_axe, sorting_rewards, color='blue', label='Sorting Reward')
    ax6.plot(x_axe, pressing_rewards, color='darkgreen', label='Pressing Reward')
    ax6.plot(x_axe, input_rewards, color='orange', label='Input Reward')
    ax6.legend(loc='lower right', fontsize=8)
    ax6.set_xlim([0, None])
    ax6.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%d' % (x)))
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax6.tick_params(axis='x', labelsize=10)
    ax6.tick_params(axis='y', labelsize=10)
    mean_sorting_reward = np.mean(sorting_rewards) if sorting_rewards else 0
    ax6.text(0.02, 0.95, f'Mean: {mean_sorting_reward:.2f}',
             horizontalalignment='left', verticalalignment='top', transform=ax6.transAxes,
             color='black', fontsize=10, fontweight='bold')
    try:
        min_reward = min(min(sorting_rewards), min(pressing_rewards), min(input_rewards))
        max_reward = max(max(sorting_rewards), max(pressing_rewards), max(input_rewards))
        if min_reward == max_reward:
            ax6.set_ylim([min_reward - 1, max_reward + 1])
        else:
            ax6.set_ylim([min_reward * 1.1, max_reward * 1.1])
    except ValueError:
        ax6.set_ylim([0, 1])

    # ---------------------------------------------------------*/
    # 7) Container Contents (5 Container: A, B, C, D und Rest)
    # ---------------------------------------------------------*/

    ax7.set_title('Container Contents', fontweight='bold', fontsize=12)
    grouped_keys = [
        ['A', 'A_False'],
        ['B', 'B_False'],
        ['C', 'C_False'],
        ['D', 'D_False'],
        ['E',]
    ]

    ax7.set_xticks(np.arange(len(grouped_keys)))
    ax7.set_xticklabels(['A', 'B', 'C', 'D', 'Rest'], fontsize=10)
    ax7.set_ylabel('Quantity', fontsize=10)
    ax7.tick_params(axis='x', labelsize=10)
    ax7.tick_params(axis='y', labelsize=10)
    total_container_contents = sum(container_materials.values())

    for index, group in enumerate(grouped_keys):
        bottoms = 0
        total = 0
        for key in group:
            value = container_materials.get(key, 0)
            total += value
            ax7.bar(index, value, bottom=bottoms, color=colors[key], label=key if bottoms == 0 else "")

            ax7.text(index, bottoms, str(value), ha='center', va='bottom', color='black', fontsize=10)
            bottoms += value

        if group[0] != 'E':
            ratio = (container_materials.get(group[0], 0) / total * 100) if total > 0 else 0
            ax7.text(index, total / 2, f"{ratio:.0f}%", ha='center', va='center', color='white',
                     fontweight='bold', fontsize=14)

    ax7.text(0.02, 0.98, f"Total: {total_container_contents}", ha='left',
             va='top', transform=ax7.transAxes, fontweight='bold', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    for index, key in enumerate(['A', 'B', 'C', 'D', 'E']):
        if key == 'E':
            label = ""  # Exclude percentage label for E
        else:
            label = f"{quality_thresholds[key] * 100:.0f}%" if quality_thresholds is not None else ""

        ax7.text(index, -0.10, label, ha='center', va='top',
                 color='black', fontsize=10, transform=ax7.get_xaxis_transform())

    # ---------------------------------------------------------*/
    # 8) Press Waiting Time
    # ---------------------------------------------------------*/

    # Ensure all press state values exist and are valid
    countdown_value_1 = press_state.get("press_1", 0) or 0
    max_value_1 = max(1, press_state.get("n_1", 1) // 20)  # Prevent division by zero
    elapsed_time_1 = max_value_1 - countdown_value_1

    countdown_value_2 = press_state.get("press_2", 0) or 0
    max_value_2 = max(1, press_state.get("n_2", 1) // 25)  # Prevent division by zero
    elapsed_time_2 = max_value_2 - countdown_value_2

    # Ensure no NaN or None values exist in pie chart sizes
    sizes_1 = np.nan_to_num([countdown_value_1, elapsed_time_1], nan=0.0)
    colors_1 = ['lightgray', 'skyblue']
    sizes_2 = np.nan_to_num([countdown_value_2, elapsed_time_2], nan=0.0)
    colors_2 = ['lightgray', 'lightcoral']

    # Ensure pie chart has valid sizes (avoid empty values)
    if sum(sizes_1) == 0 or np.isnan(sum(sizes_1)):
        sizes_1 = [1]
        colors_1 = ['white']
    if sum(sizes_2) == 0 or np.isnan(sum(sizes_2)):
        sizes_2 = [1]
        colors_2 = ['white']

    # Draw first pie chart
    ax8.pie(sizes_1, colors=colors_1, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.3), radius=1.8, center=(-1.8, 0))

    # Display press 1 info
    if countdown_value_1 > 0:
        material_1 = press_state.get("material_1", "Unknown")
        quantity_1 = press_state.get("n_1", 0) or 0
        quality_1 = int((press_state.get("q_1", 0) or 0) * 100)
        ax8.text(-1.8, 0, f'Material: {material_1}\nQuantity: {quantity_1}\nQuality: {quality_1}%',
                 ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        ax8.text(-1.8, -2.2, 'Ready!', ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw second pie chart
    if any(x < 0 for x in sizes_2):
        # print("Negative Value detected in sizes_2:", sizes_2)  # Debugging line
        sizes_2 = [max(0, x) for x in sizes_2]  # Replace negative values with 0

    ax8.pie(sizes_2, colors=colors_2, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.3), radius=1.8, center=(1.8, 0))

    # Display press 2 info
    if countdown_value_2 > 0:
        material_2 = press_state.get("material_2", "Unknown")
        quantity_2 = press_state.get("n_2", 0) or 0
        quality_2 = int((press_state.get("q_2", 0) or 0) * 100)
        ax8.text(1.8, 0, f'Material: {material_2}\nQuantity: {quantity_2}\nQuality: {quality_2}%',
                 ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        ax8.text(1.8, -2.2, 'Ready!', ha='center', va='center', fontsize=10, fontweight='bold')

    # Add labels
    ax8.text(-1.8, 2.2, 'Press 1', ha='center', va='center', fontweight='bold', fontsize=12)
    ax8.text(1.8, 2.2, 'Press 2', ha='center', va='center', fontweight='bold', fontsize=12)
    ax8.axis('equal')
    ax8.axis('off')

    # ---------------------------------------------------------*/
    # 9) Bales Produced
    # ---------------------------------------------------------*/
    bales = bale_count
    categories = ['A', 'B', 'C', 'D', 'E']
    # Hier verwenden wir 'E' als Rest-Container, die x-Achsen-Beschriftung wird angepasst
    cmap = LinearSegmentedColormap.from_list('deviation_cmap', ['red', 'green', 'red'], N=10)
    norm = BoundaryNorm(np.linspace(0.5, 1.5, 11), cmap.N)

    def get_color(deviation):
        return cmap(norm(deviation))
    bottoms = {key: 0 for key in categories}
    for idx, key in enumerate(categories):
        if key in bales and len(bales[key]) > 0:
            for i in range(len(bales[key])):
                size, quality = bales[key][i]
                deviation = size / bale_standard_size
                bar = ax9.bar(idx, deviation, bottom=bottoms[key], color=get_color(deviation))
                ax9.text(idx, bottoms[key] + deviation / 2, f"{int(quality)}%",
                         ha='center', va='center', fontsize=8, color='black')
                bottoms[key] += deviation
    ax9.set_xticks(range(len(categories)))
    ax9.set_xticklabels(['A', 'B', 'C', 'D', 'Rest'])
    ax9.set_title('Bales Produced', fontweight='bold', fontsize=12)
    ax9.set_ylabel('Relative Bale Size', fontsize=10)
    if max(bottoms.values()) > 0:
        ax9.set_ylim([0, max(bottoms.values()) * 1.1])
    else:
        ax9.set_ylim([0, 1])
    ax9.tick_params(axis='x', labelsize=10)
    ax9.tick_params(axis='y', labelsize=10)
    ax9.yaxis.set_major_locator(MaxNLocator(integer=True))
    for idx, key in enumerate(categories):
        if key in bales and len(bales[key]) > 0:
            ax9.text(idx, 0, f"n={len(bales[key])}", ha='center', va='bottom',
                     fontsize=14, color='white')
    total_material = sum(size for bales_list in bale_count.values() for size, quality in bales_list)
    ax9.text(0.02, 0.98, f'Total Material = {total_material}', transform=ax9.transAxes,
             ha='left', va='top', fontweight='bold', fontsize=10)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax9, orientation='vertical')
    cbar.set_label('Deviation from Bale Size', fontsize=10)
    cbar.set_ticks(np.linspace(0.5, 1.5, 11))
    cbar.set_ticklabels([f'{t:.1f}' for t in np.linspace(0.5, 1.5, 11)])

    # ---------------------------------------------------------*/
    # 10) Cumulative Total Reward Plot
    # ---------------------------------------------------------*/
    ax10.set_title('Total Rewards', fontweight='bold', fontsize=12)
    total_rewards = [sum(reward) for reward in reward_data['Reward']]
    cumulative_total_reward = np.cumsum(total_rewards)
    ax10.set_xlabel('Timesteps', fontsize=10)
    x_axe = range(len(total_rewards))
    ax10.plot(x_axe, total_rewards, color='red', label='Total Reward')
    ax10.plot(x_axe, cumulative_total_reward, color='purple', label='Cumulative Total Reward')
    ax10.legend(loc='upper left', fontsize=8)
    ax10.set_xlim([0, None])
    ax10.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%d' % (x)))
    ax10.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax10.tick_params(axis='x', labelsize=10)
    ax10.tick_params(axis='y', labelsize=10)
    try:
        min_reward = min(min(total_rewards), min(cumulative_total_reward))
        max_reward = max(max(total_rewards), max(cumulative_total_reward))
        if min_reward == max_reward:
            ax10.set_ylim([min_reward - 1, max_reward + 1])
        else:
            ax10.set_ylim([min_reward * 1.1, max_reward * 1.1])
    except ValueError:
        ax10.set_ylim([0, 1])
    final_total_reward = round(cumulative_total_reward[-1] if cumulative_total_reward.size > 0 else 0, 2)
    ax10.text(0.02, 0.8, f'Cumulative Total: {final_total_reward}', transform=ax10.transAxes,
              ha='left', va='bottom', fontweight='bold', fontsize=10,
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # ---------------------------------------------------------*/
    # Calculate the checksum
    # ---------------------------------------------------------*/
    if checksum:
        total_material_in_containers = sum(container_materials.values())
        total_material_in_presses = sum([press_state["n_1"], press_state["n_2"]])
        total_material_in_bales = sum(value[0] for values in bale_count.values() for value in values)

        # Calculate the checksum
        checksum = total_material_in_containers + total_material_in_presses + total_material_in_bales
        print(f"🔍 Checksum: {checksum} ({total_material_in_containers} + {total_material_in_presses} + {total_material_in_bales})")

    # ---------------------------------------------------------*/
    # Generate the plot ✍🏼
    # ---------------------------------------------------------*/
    if save or show:
        plt.tight_layout()
        if save:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            plt.savefig(f"{log_dir}/{filename}.{format}", format=format, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    plt.close()
    fig.clf()


# ---------------------------------------------------------*/
# Save the plot to a file
# ---------------------------------------------------------*/


def save_plot(fig, dir, base_filename, extension, dpi=300):
    """Saves the plot to a file with a unique filename in the specified directory."""
    os.makedirs(dir, exist_ok=True)  # Ensure the directory exists

    # Generate unique filename
    i=0
    filename=f"{base_filename}_{i}.{extension}"
    while os.path.exists(os.path.join(dir, filename)):
        i += 1
        filename=f"{base_filename}_{i}.{extension}"

    # Save the figure
    if extension == 'svg':
        fig.savefig(os.path.join(dir, filename), format=extension)
    else:
        fig.savefig(os.path.join(dir, filename), format=extension, dpi=dpi, bbox_inches='tight')

# ---------------------------------------------------------*/
# Create a Video from a folder of images
# ---------------------------------------------------------*/


def create_video(folder_path, output_path, display_duration=1):
    """ Creates a video from a folder of images.
    - folder_path: The path to the folder containing the images.
    - output_path: The path to save the output video.
    - display_duration: The duration (in seconds) to display each image.
    """
    sample_img=cv2.imread(os.path.join(folder_path, os.listdir(folder_path)[0]))
    height, width, layers=sample_img.shape
    size=(width, height)

    # Frame-Rate basierend auf der gewünschten Anzeigedauer jedes Bildes berechnen
    frame_rate=1 / display_duration

    out=cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, size)

    # Funktion zum Extrahieren der Nummern aus dem Dateinamen für die korrekte Sortierung
    def sort_key(filename):
        numbers=re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    # Dateien numerisch sortieren und Video schreiben
    for filename in sorted(os.listdir(folder_path), key=sort_key):
        if filename.endswith('.png'):
            # print("Added ", filename)
            file_path=os.path.join(folder_path, filename)
            img=cv2.imread(file_path)
            img=cv2.resize(img, size)  # Resize the image to the target size
            out.write(img)

    print("Video created. 🎥🍿")
    out.release()

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*/
