import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_metrics(train_numbers):
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

# List of training numbers to plot
train_numbers = [20, 21, 22, 23, 24, 2401]

# Call the function to plot the metrics
plot_metrics(train_numbers)