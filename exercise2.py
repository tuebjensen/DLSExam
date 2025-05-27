import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Make a bayesian neural network that predics the mean and log of the variance
class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


def generate_training_data(device, num_samples=100):
    # Generate synthetic training data with a sine wave and gaussian noise with variance of 0.1
    x = np.linspace(0, 1, num_samples)
    sine_wave = np.sin(2 * np.pi * x)
    noised_sine_wave = sine_wave + np.random.normal(0, 0.1, num_samples)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(device)
    noised_sine_wave = (
        torch.tensor(noised_sine_wave, dtype=torch.float32).unsqueeze(-1).to(device)
    )
    return x, noised_sine_wave, sine_wave


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate training data
    x_train, y_train, y_clean = generate_training_data(device)

    # Plot the training data
    # plt.figure(figsize=(10, 5))
    # plt.plot(x.cpu().numpy(), y.cpu().numpy(), "o", label="Training Data")
    # plt.title("Training Data")
    # plt.xlabel("Input")
    # plt.ylabel("Output")
    # plt.legend()
    # plt.grid()
    # plt.savefig("plots/exercise2_training_data.png", bbox_inches="tight")
    # plt.show()

    # Define the Bayesian Neural Network
    input_dim = 1
    output_dim = 1
    model = BNN(input_dim, output_dim).to(device)
    criterion = nn.GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 1000
    losses = []
    for epoch in range(num_epochs):
        # Forward pass
        mean, log_var = model(x_train)
        loss = criterion(mean, y_train, torch.exp(log_var))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("plots/exercise2_training_loss.png", bbox_inches="tight")

    model.eval()
    with torch.no_grad():
        mean, log_var = model(x_train)

    mean = mean.squeeze().cpu().numpy()
    log_var = log_var.squeeze().cpu().numpy()
    x_train = x_train.squeeze().cpu().numpy()
    y_train = y_train.squeeze().cpu().numpy()
    var_pred = np.exp(log_var)
    std_pred = np.sqrt(var_pred)
    lower_bound = mean - 2 * std_pred
    upper_bound = mean + 2 * std_pred

    # Plot the predictions with uncertainty bounds
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_train,
        mean,
        label="Predicted Mean",
        color="orange",
    )
    plt.fill_between(
        x_train,
        lower_bound,
        upper_bound,
        color="lightblue",
        alpha=0.5,
        label="Uncertainty Bounds (2 std)",
    )
    plt.plot(x_train, y_clean, label="True Function", color="green")
    plt.plot(x_train, y_train, "o", label="Training Data")
    plt.title("Bayesian Neural Network Predictions with Uncertainty")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.savefig("plots/exercise2_predictions_with_uncertainty.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
