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
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, torch.exp(log_var)


def generate_data(device="cpu", lower=0.0, upper=1.0, num_samples=100):
    # Generate synthetic training data with a sine wave and gaussian noise with variance of 0.1
    x = np.linspace(lower, upper, num_samples)
    sine_wave = np.sin(2 * np.pi * x)
    noised_sine_wave = sine_wave + np.random.normal(0, np.sqrt(0.1), num_samples)

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(device)
    noised_sine_wave = (
        torch.tensor(noised_sine_wave, dtype=torch.float32).unsqueeze(-1).to(device)
    )
    return x, noised_sine_wave, sine_wave


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Generate training data
    x_train, y_train, y_clean = generate_data(device)

    # Define the Bayesian Neural Network an optimizer
    input_dim = 1
    output_dim = 1
    model = BNN(input_dim, output_dim).to(device)
    criterion = nn.GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 1000
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        mean_pred, var_pred = model(x_train)
        loss = criterion(mean_pred, y_train, var_pred)
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

    x_test, _, y_clean_test = generate_data(device, num_samples=1000)
    model.eval()
    with torch.no_grad():
        mean_pred, var_pred = model(x_test)

    if device == "cuda":
        mean_pred = mean_pred.cpu()
        var_pred = var_pred.cpu()
        x_train = x_train.cpu()
        y_train = y_train.cpu()
        x_test = x_test.cpu()
    mean_pred = mean_pred.squeeze().numpy()
    var_pred = var_pred.squeeze().numpy()
    x_train = x_train.squeeze().numpy()
    y_train = y_train.squeeze().numpy()
    x_test = x_test.squeeze().numpy()
    std_pred = np.sqrt(var_pred)
    lower_bound = mean_pred - 2 * std_pred
    upper_bound = mean_pred + 2 * std_pred
    std_real = np.sqrt(0.1)  # The noise variance is 0.1, so the std is sqrt(0.1)
    lower_bound_real = y_clean_test - 2 * std_real
    upper_bound_real = y_clean_test + 2 * std_real
    # Plot the predictions with uncertainty bounds
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_test,
        mean_pred,
        label="Predicted Mean",
        color="black",
    )
    plt.fill_between(
        x_test,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.2,
        label="Uncertainty Bounds (2 std)",
    )
    plt.plot(x_test, y_clean_test, label="True Function", color="green")
    plt.fill_between(
        x_test,
        lower_bound_real,
        upper_bound_real,
        color="red",
        alpha=0.2,
        label="True Uncertainty Bounds (2 std)",
    )
    plt.plot(x_train, y_train, "o", label="Training Data")
    plt.title("Bayesian Neural Network Predictions with Uncertainty")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.savefig("plots/exercise2_predictions.png", bbox_inches="tight")

    x_test, _, y_test_clean = generate_data(
        device, lower=-0.5, upper=1.5, num_samples=1000
    )
    with torch.no_grad():
        mean_pred, var_pred = model(x_test)

    if device == "cuda":
        mean_pred = mean_pred.cpu()
        var_pred = var_pred.cpu()
        x_test = x_test.cpu()

    mean_pred = mean_pred.squeeze().numpy()
    var_pred = var_pred.squeeze().numpy()
    x_test = x_test.squeeze().numpy()
    std_pred = np.sqrt(var_pred)
    lower_bound = mean_pred - 2 * std_pred
    upper_bound = mean_pred + 2 * std_pred
    lower_bound_real = y_test_clean - 2 * std_real
    upper_bound_real = y_test_clean + 2 * std_real
    # Plot the predictions with uncertainty bounds
    plt.figure(figsize=(10, 5))
    plt.plot(
        x_test,
        mean_pred,
        label="Predicted Mean",
        color="black",
    )
    plt.fill_between(
        x_test,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.2,
        label="Uncertainty Bounds (2 std)",
    )
    plt.plot(x_test, y_test_clean, label="True Function", color="green")
    plt.fill_between(
        x_test,
        lower_bound_real,
        upper_bound_real,
        color="red",
        alpha=0.2,
        label="True Uncertainty Bounds (2 std)",
    )
    plt.plot(x_train, y_train, "o", label="Training Data")
    plt.title("Bayesian Neural Network Predictions with Uncertainty")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.savefig("plots/exercise2_test_predictions.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
