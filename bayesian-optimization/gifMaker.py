import json
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load data from JSON
with open('progress.json', 'r') as f:
    data = json.load(f)

# Function to plot and save an image
def plot_iteration(iteration, data, save_path):
    plt.figure()
    target_values = [d['target'] for d in data[:iteration+1]]
    plt.plot(target_values, marker='o')
    plt.title(f'Optimization Progress up to Iteration {iteration}')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.savefig(save_path)
    plt.close()

# Create a folder for the images
os.makedirs('plots', exist_ok=True)

# Generate and save plots
for i in range(len(data)):
    plot_iteration(i, data, f'plots/iteration_{i}.png')

# Create a GIF
images = []
for i in range(len(data)):
    images.append(Image.open(f'plots/iteration_{i}.png'))

images[0].save('optimization_progress.gif', save_all=True, append_images=images[1:], duration=300, loop=0)

# Clean up (optional)
for image_file in os.listdir('plots'):
    os.remove(f'plots/{image_file}')
os.rmdir('plots')
