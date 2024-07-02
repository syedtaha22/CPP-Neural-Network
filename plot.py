import matplotlib.pyplot as plt
import os
import pandas as pd
import tqdm


# Define folder name for plots
folder_name = "plots"

# Create directory for plots if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Read the data from the file
data = pd.read_csv("predictions_epoch.csv")

# Define axis limits
x_min, x_max = -6, 6
y_min, y_max = -2, 2

# Extract unique epochs
unique_epochs = sorted(set(data["Epoch"]))

for epoch in tqdm.tqdm(unique_epochs):
    # Filter data for the current epoch
    epoch_data = data[data["Epoch"] == epoch]

    # Prepare data for plotting
    inputs = epoch_data["Input"]
    predictions = epoch_data["Prediction"]
    actual_values = epoch_data["Actual"]

    # Clip values to fit within the defined range
    inputs_clipped = inputs.clip(lower=x_min, upper=x_max)
    predictions_clipped = predictions.clip(lower=y_min, upper=y_max)
    actual_values_clipped = actual_values.clip(lower=y_min, upper=y_max)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        inputs_clipped,
        actual_values_clipped,
        label="Actual Values",
        color="blue",
    )
    plt.plot(
        inputs_clipped,
        predictions_clipped,
        label="Model Predictions",
        color="red",
        marker="x",
        linestyle="--",
        markersize=5,
    )

    plt.xlabel("Input")
    plt.ylabel("Value")
    plt.title(f"Neural Network Predictions vs. Actual Values (Epoch {epoch})")
    plt.legend()
    plt.grid(True)

    # Set axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Save plot
    plot_filename = os.path.join(folder_name, f"epoch_{epoch}.png")
    plt.savefig(plot_filename)
    plt.close()


print(f"All plots have been saved in the '{folder_name}' folder.")
