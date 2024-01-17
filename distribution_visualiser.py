import torch
from helper import symmetric_kl_divergence_2d, compute_euclidian_distance
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.widgets import Slider


loaded_mus = torch.load('saved_mus_k3.pt', map_location=torch.device('cpu'))
loaded_logvars = torch.load('saved_logvars_k3.pt', map_location=torch.device('cpu'))
loaded_weights = torch.load('saved_weights_k3.pt', map_location=torch.device('cpu'))
loaded_indices = torch.load('saved_indices_k3.pt', map_location=torch.device('cpu'))


def plot_kl_divergence_distances():
    distances = []
    for op_step in tqdm(range(loaded_mus.shape[0] - 100, loaded_mus.shape[0])):
        step_mu = loaded_mus[op_step]
        step_logvar = loaded_logvars[op_step]
        distance = symmetric_kl_divergence_2d(step_mu, step_logvar)
        distances.append(float(distance))

    x_values = range(len(distances))

    # Creating the plot
    plt.plot(x_values, distances)

    # Adding title and labels
    plt.title("Average distance of distributions during training")
    plt.xlabel("Training Step")
    plt.ylabel("Average Distance Between Distributions")

    # Showing the plot
    plt.show()


def plot_euclidian_distances():
    distances = []
    for op_step in tqdm(range(loaded_mus.shape[0] - 100, loaded_mus.shape[0])):
        step_mu = loaded_mus[op_step]
        step_logvar = loaded_logvars[op_step]
        distance = compute_euclidian_distance(step_mu)
        distances.append(float(distance))

    x_values = range(len(distances))

    # Creating the plot
    plt.plot(x_values, distances)

    # Adding title and labels
    plt.title("Average distance of distributions during training")
    plt.xlabel("Training Step")
    plt.ylabel("Average Distance Between Distributions")

    # Showing the plot
    plt.show()


def plot_distribution_timesteps():
    timestep_dict = {}
    batch_index = 0
    num_steps = 1000
    for op_step in tqdm(range(num_steps)):
        step_mu = loaded_mus[op_step]
        step_logvar = loaded_logvars[op_step]

        timestep_dict[op_step] = np.array([step_mu[batch_index][i].detach().numpy() for i in range(step_mu.shape[1])])

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Scatter plot for the initial time step
    scat = ax.scatter(timestep_dict[0][:, 0], timestep_dict[0][:, 1])

    # Adjust the axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time Step', 0, num_steps - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        step = int(slider.val)
        ax.set_title(f"Time step: {step}")
        scat.set_offsets(timestep_dict[step])

    slider.on_changed(update)

    plt.show()


def plot_distribution_weights():
    num_steps = 1000
    weights = []
    for op_step in tqdm(range(0, 1000)):
        batch_weights = loaded_weights[op_step]
        column_means = batch_weights.mean(dim=0)
        weights.append(column_means.detach().numpy())
    weights = np.array(weights)

    # Time steps
    time_steps = np.arange(len(weights))

    # Plot each dimension
    plt.figure(figsize=(10, 6))
    for i in range(weights.shape[1]):
        plt.plot(time_steps, weights[:, i], label=f'Head {i + 1}')

    # Adding labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Weight')
    plt.title('Development of Each Head Weight Over Time')
    plt.legend()

    # Show the plot
    plt.show()


def plot_chosen_indices():
    overall_counts = []
    for op_step in tqdm(range(0, 15000)):
        indices = loaded_indices[op_step]
        unique, counts = torch.unique(indices, return_counts=True)
        overall_counts.append(counts.detach().numpy())
    overall_counts = np.array(overall_counts)
    batches = np.arange(len(overall_counts))

    # Plot each dimension
    plt.figure(figsize=(10, 6))
    for i in range(overall_counts.shape[1]):
        plt.plot(batches, overall_counts[:, i], label=f'Head {i + 1}')

    # Adding labels and title
    plt.xlabel('Batch')
    plt.ylabel('Frequency')
    plt.title('Frequency of choosing each head')
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    plot_kl_divergence_distances()
