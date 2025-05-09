import matplotlib.pyplot as plt
import numpy as np
import pickle


def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

file2 = 'scripts/data/nina_measurements_31_1-1.pkl'
data2 = load_pickle_data(file2)

# Extract the time interval, starting phases, and end phases
time_interval = data2[0]  # Time interval between samples (in seconds)
start_phases = data2[1]   # Starting phases
end_phases = data2[2]     # End phases
measurements = data2[3]   # Core measurement series

# Initialize a list to store transition times
transition_times = []

phase_data_dic = {}

# Loop through each of the 20 measurements
for measurement in range(20):
    start_phase = start_phases[measurement]
    end_phase = end_phases[measurement]
    
    # Extract the 60 time samples for the current measurement
    phase_data = measurements[60 * measurement: 60 * (measurement + 1)]
        
    phase_data_dic[measurement] = phase_data
    
# delete the first measurement
# Split the phase_data_dic into two parts
# phase_data_dic_part1 = {k: v for k, v in list(phase_data_dic.items())[:10]}  # First 10 measurements
# phase_data_dic_part2 = {k: v for k, v in list(phase_data_dic.items())[10:]}  # Last 10 measurements


def analyze_phase_shifts(data_dict, timestep=0.001, figure_title="Phase Shifts"):
    summary = {}
    num_measurements = len(data_dict)
    cols = 5
    rows = (num_measurements + cols - 1) // cols  # round up

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2), squeeze=False)
    threshold = 0.0001  # Smaller threshold for sensitive detection

    for idx, (label, phase_data) in enumerate(data_dict.items()):
        phase_data = np.array(phase_data)
        time = np.arange(len(phase_data)) * timestep
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        avg_pre = np.mean(phase_data[(time >= 0.00) & (time <= 0.02)])
        avg_post = np.mean(phase_data[(time >= 0.03) & (time <= 0.06)])

        try:
            start_index = next(j for j in range(len(time)) if time[j] > 0.02 and phase_data[j] < avg_pre - threshold)
        except StopIteration:
            print(f"Warning: No start index found for measurement {label}. Skipping.")
            continue
        
        end_index = next(j for j in range(start_index + 1, len(phase_data)) if phase_data[j] <= avg_post)
        duration = (end_index - start_index) * timestep
        summary[label] = duration

        ax.plot(time, phase_data, label='Phase')
        ax.axhline(avg_pre, color='blue', linestyle='--', label='Avg Pre')
        ax.axhline(avg_post, color='purple', linestyle='--', label='Avg Post')
        ax.axvline(time[start_index], color='green', linestyle='--', label='Start')
        ax.axvline(time[end_index], color='orange', linestyle='--', label='End')
        ax.axvspan(time[start_index], time[end_index], color='yellow', alpha=0.3)

        ax.set_title(f"{label} (Δt={duration:.3f}s)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Phase", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True)
        ax.legend(loc='lower left', fontsize=6)

    # Turn off unused subplots
    for idx in range(len(data_dict), rows * cols):
        row, col = divmod(idx, cols)
        fig.delaxes(axes[row][col])

    plt.suptitle(figure_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig('scripts/plots/phase_shifts_measurements.png')
    plt.show()
    

    # Print summary
    print("Phase Shift Duration Summary:")
    for label, dur in summary.items():
        print(f"{label}: {dur:.4f} seconds")

    return summary


summary_part = analyze_phase_shifts(phase_data_dic, figure_title="Phase Shifts (First 10 Measurements)")

all_durations = list(summary_part.values())

# Calculate the average shift duration
average_duration = np.mean(all_durations)

# Print the average shift duration
print(f"\nAverage Phase Shift Duration Over All Measurements: {average_duration:.4f} seconds")