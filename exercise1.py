import os
from math import log2

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.nn.functional import relu
from torch_geometric.datasets import Airports, KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx


def connectivity_entropy(graph: nx.Graph) -> float:
    sum = 0
    for node in graph.nodes():
        chi = graph.degree[node] / (2 * graph.number_of_edges())
        chi = chi if chi > 0 else 0.01 / (2 * graph.number_of_nodes())
        sum += chi * log2(chi)
    return -sum


def connectivity(graph: nx.Graph) -> dict[int, float]:
    """Calculate the connectivity of a graph."""
    conn = connectivity_entropy(graph)
    subraph_conn: dict[int, float] = {}
    for node in graph.nodes():
        subgraph = graph.copy()
        subgraph.remove_node(node)
        subraph_conn[node] = conn - connectivity_entropy(subgraph)
    return subraph_conn


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 16):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = relu(x)
        x = self.conv2(x, edge_index)
        return x


def main():
    # Load the Karate Club dataset
    dataset = KarateClub()
    graph: nx.Graph = to_networkx(dataset[0])
    train_status_print_per_ep = 20
    train_eps = 500

    # Calculate centrality measures
    closeness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    connectivity_ent = connectivity(graph)
    eigenvector_scores = nx.eigenvector_centrality(graph)

    # Create feature matrix with centrality measures as the features
    num_nodes = graph.number_of_nodes()
    features = np.zeros((num_nodes, 4))  # 4 centrality measures
    for i, node in enumerate(graph.nodes()):
        features[i, 0] = closeness[node]
        features[i, 1] = betweenness[node]
        features[i, 2] = connectivity_ent[node]
        features[i, 3] = eigenvector_scores[node]

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    print(
        "Feature Matrix: Closeness, Betweenness, Connectivity Entropy, Eigenvector Centrality\n",
        features,
    )

    # Create a GCN encoder and train it
    encoder = GCNEncoder(in_channels=4, out_channels=1)
    optimizer = optim.AdamW(encoder.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    input_features = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(np.array(graph.edges()).T, dtype=torch.int)

    # Target is the mean of the features for each node
    target = np.mean(features, axis=1, keepdims=True)
    target = torch.tensor(target, dtype=torch.float)

    print("Target values (mean of features for each node):\n", target)

    losses = []
    for epoch in range(train_eps):
        optimizer.zero_grad()

        output = encoder(input_features, edge_index)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % train_status_print_per_ep == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Plot the loss curve
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("plots/exercise1_training_loss_curve.png", bbox_inches="tight")

    # Get predictions
    with torch.no_grad():
        output = encoder(input_features, edge_index)

    # Make a plot where it's score vs node index, and both the GCN output and target
    plt.figure()
    plt.scatter(
        range(num_nodes), output.detach().numpy(), label="GCN Output", marker="x"
    )
    plt.scatter(range(num_nodes), target.numpy(), label="Target", marker="x")
    for i in range(num_nodes):
        plt.axvline(x=i, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    plt.title("GCN Output vs Target")
    plt.xlabel("Node Index")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(
        "plots/exercise1_gcn_output_vs_target_node_index.png", bbox_inches="tight"
    )

    # Make a plot where it's the GCN output - target vs node index, with the largest 5 differences highlighted
    differences = target.numpy() - output.detach().numpy()
    plt.figure()
    plt.scatter(range(num_nodes), differences, label="GCN Output - Target", marker="x")
    plt.title("Target minus GCN Output vs Node Index")
    plt.xlabel("Node Index")
    plt.ylabel("Target - GCN Output")
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    for i in range(num_nodes):
        plt.axvline(x=i, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    # Highlight the largest 5 differences
    largest_diffs_indices = np.argsort(np.abs(differences.flatten()))[-5:]
    for idx in largest_diffs_indices:
        diff_val = differences[idx][0]
        # If difference is positive, put annotation below; if negative, put above
        y_offset = 10 if diff_val < 0 else -15
        va = "bottom" if diff_val < 0 else "top"
        plt.annotate(
            f"{idx}\n{diff_val:.4f}",
            (idx, diff_val),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            va=va,
            fontsize=8,
            color="red",
        )
    plt.scatter(
        largest_diffs_indices,
        differences[largest_diffs_indices],
        facecolors="none",
        edgecolors="red",
        label="Largest 5 Differences",
        marker="o",
        s=80,
        linewidths=1.5,
    )
    plt.legend()
    plt.savefig(
        "plots/exercise1_gcn_output_minus_target_node_index_largest_diffs.png",
        bbox_inches="tight",
    )

    # Compute the Pearson correlation between true and GNN-fused scores.
    pearson_corr, _ = pearsonr(
        output.detach().numpy().flatten(), target.numpy().flatten()
    )
    print(f"Pearson correlation (scipy): {pearson_corr:.4f}")

    # print GCN and target output side by side (zip)
    print("GCN Output vs Target:")
    for o, t in zip(output.numpy(), target.numpy()):
        print(f"Output: {o[0]:.4f}, Target: {t[0]:.4f}")
    plt.figure()
    plt.scatter(output.numpy(), target.numpy(), label="GCN Output vs Target")
    min_val = min(output.min().item(), target.min().item())
    max_val = max(output.max().item(), target.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
    plt.title("GCN Output vs Target")
    plt.xlabel("GCN Output")
    plt.ylabel("Target")
    plt.legend(title="Pearson Correlation: {:.4f}".format(pearson_corr))
    plt.grid()
    plt.savefig("plots/exercise1_gcn_output_vs_target.png", bbox_inches="tight")

    print("Training complete.")


if __name__ == "__main__":
    main()
