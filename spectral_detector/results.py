import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_box_loss(train_numbers):
    # Read the results from results.csv
    data = {}
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'magenta']  # Add more colors if needed
    for i, train_number in enumerate(train_numbers):
        file_path = f'runs/detect/train{train_number}/results.csv'
        data[train_number] = pd.read_csv(file_path)
        data[train_number].columns = data[train_number].columns.str.strip()

    # Plot the results
    for i, train_number in enumerate(train_numbers):
        color = colors[i % len(colors)]  # Get color based on index
        plt.plot(data[train_number]['epoch'], data[train_number]['train/box_loss'], label=f'train{train_number}', color=color)
        plt.plot(data[train_number]['epoch'], data[train_number]['val/box_loss'], label=f'train{train_number}', linestyle='dashed', color=color)

    # x axis label
    plt.xlabel('Epoch')

    # y axis label
    plt.ylabel('Box Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

plot_box_loss([9,18,20,21,22])