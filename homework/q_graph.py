import pandas as pd
import matplotlib.pyplot as plt
import os

# Create the 'graphs' folder if it doesn't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Load the CSV data
file_path = "rollout_results.csv"  # File name for the CSV data
data = pd.read_csv(file_path)

# Group by track and plot each track separately
tracks = data['Track'].unique()

for track in tracks:
    track_data = data[data['Track'] == track]
    plt.figure(figsize=(8, 5))
    plt.plot(track_data['Epoch'], track_data['Distance'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Completion (Distance)')
    plt.title(f'Epoch vs Completion for {track}')
    plt.grid(True)
    plt.tight_layout()
    # Save each plot as a file in the 'graphs' folder
    plt.savefig(f"graphs/{track}_epoch_vs_completion.png")
    plt.close()

# Perform overall trend analysis and plot
overall_trend = data.groupby('Epoch')['Distance'].mean()
plt.figure(figsize=(8, 5))
plt.plot(overall_trend.index, overall_trend.values, marker='o', label='Overall Trend')
plt.xlabel('Epoch')
plt.ylabel('Average Completion (Distance)')
plt.title('Overall Epoch vs Average Completion')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/overall_epoch_vs_completion.png")
plt.close()