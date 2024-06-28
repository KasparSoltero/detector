import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

def draw_loss_curves(train_numbers, save_path=None):
    # Read the results from results.csv
    data = {}
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'magenta', 'cyan', 'yellow', 'brown']
    
    for i, train_number in enumerate(train_numbers):
        file_path = f'runs/detect/train{train_number}/results.csv'
        data[train_number] = pd.read_csv(file_path)
        data[train_number].columns = data[train_number].columns.str.strip()

    # Define the metrics to plot with losses grouped together
    metrics = ['train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss', 
               'train/dfl_loss', 'val/dfl_loss', 'metrics/precision(B)', 
               'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    # Create a 3x4 grid for subplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()  # Flatten the 3x4 array to easily iterate over it

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for i, train_number in enumerate(train_numbers):
            color = colors[i % len(colors)]  # Get color based on index
            ax.plot(data[train_number]['epoch'], data[train_number][metric], label=f'train{train_number}', color=color)

        # Set plot titles and labels
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.split('/')[-1])
        ax.legend()

    # Remove any empty subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_gt_detection_metrics(train_numbers, plot=True, save_name=None):
    # Define a list of colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # check if string or int
    if isinstance(train_numbers, str):
        train_numbers = [train_numbers]
    elif isinstance(train_numbers, int):
        train_numbers = [str(train_numbers)]
    train_numbers = [str(tn) for tn in train_numbers]

    # get the csv files with the train numbers
    csv_files = [f for f in os.listdir('results') if f.endswith('.csv') and f.split('_')[1] in train_numbers]
    if len(csv_files) == 0:
        print(f'No CSV files found for train numbers {train_numbers}')
        return

    # Create two figures
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for i, result_file in enumerate(csv_files):
        df = pd.read_csv(os.path.join('results', result_file))
        # Extract global average row
        global_avg = df[df['file'] == 'GLOBAL_AVERAGE'].iloc[0]
        # Extract padding values
        padding_values = [int(col.split('_')[-1]) for col in df.columns if col.startswith('percent_time_intersection_pad_')]

        # Plot 1: Time percentage values for global average
        color = colors[i % len(colors)]
        ax1.plot(padding_values, [global_avg[f'percent_time_intersection_pad_{pad}'] for pad in padding_values], 
                 label=f"{result_file.split('_')[-2]} - True Positive", marker='o', color=color)
        ax1.plot(padding_values, [global_avg[f'percent_time_fp_pad_{pad}'] for pad in padding_values], 
                 linestyle='--', color=color, label=f"{result_file.split('_')[-1][:-4]} - False Positive")
        ax1.set_yticks(range(0, 101, 10))

        # Plot 2: Percentage of correct predictions vs number of boxes
        file_data = df[df['file'] != 'GLOBAL_AVERAGE'].copy()  # Create an explicit copy

        # Convert percent_centers to float and handle any potential string values
        file_data.loc[:, 'percent_centers'] = pd.to_numeric(file_data['percent_centers'], errors='coerce')

        # Handle cases where ground truth is 0
        file_data.loc[:, 'adjusted_percent'] = np.where(file_data['number_gt_boxes'] == 0, 100, file_data['percent_centers'])

        # Plot scatter points
        ax2.scatter(file_data['number_gt_boxes'], file_data['adjusted_percent'], 
                    label=f"{result_file.split('_')[-1][:-4]}", marker='o', color=color)

    # Finalize Plot 1
    ax1.set_xlabel('Prediction Padding (seconds)')
    ax1.set_ylabel('Percentage Total Time')
    ax1.legend()
    ax1.grid(True)

    # Finalize Plot 2
    ax2.set_xlabel('Number of Boxes')
    ax2.set_ylabel('Percentage of Correct Predictions')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_name:
        fig1.savefig(save_name+'_time_percentages.png')
        fig2.savefig(save_name+'_percent_centers.png')

    if plot:
        plt.show()

# List of training numbers to plot
train_numbers = [2602]

# draw_loss_curves(train_numbers)