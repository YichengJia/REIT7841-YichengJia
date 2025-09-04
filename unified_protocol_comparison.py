"""
unified_protocol_comparison.py
Unified comparison of multiple federated learning protocols
"""

import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple
import copy

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.unicode_minus'] = False

from federated_protocol_framework import (
    create_protocol, ClientUpdate, FederatedProtocol
)
from optimized_protocol_config import generate_all_configs


# -----------------------------
# Simple Model Definition
# -----------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# -----------------------------
# Data Functions
# -----------------------------
def generate_federated_data(num_clients: int, samples_per_client: int,
                           input_dim: int, num_classes: int,
                           heterogeneity: float = 0.5) -> Tuple:
    total_samples = num_clients * samples_per_client

    X, y = make_classification(
        n_samples=total_samples,
        n_features=input_dim,
        n_informative=input_dim // 2,
        n_redundant=0,
        n_classes=num_classes,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=42
    )

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    client_datasets = []
    indices = torch.randperm(total_samples)
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_indices = indices[start_idx:end_idx]
        client_X = X[client_indices]
        client_y = y[client_indices]
        client_datasets.append(TensorDataset(client_X, client_y))

    test_size = min(len(X) // 5, 500)
    test_indices = np.random.choice(len(X), test_size, replace=False)
    test_X = X[test_indices]
    test_y = y[test_indices]
    global_test_dataset = TensorDataset(test_X, test_y)

    return client_datasets, global_test_dataset


def train_client(model: nn.Module, dataset: TensorDataset,
                epochs: int = 3, lr: float = 0.01) -> Tuple:
    if len(dataset) == 0:
        return model.state_dict(), float('inf'), 0

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    total_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(dataloader)

    avg_loss = total_loss / epochs
    return model.state_dict(), avg_loss, len(dataset)


def evaluate_model(model: nn.Module, test_dataset: TensorDataset) -> Tuple[float, float]:
    if len(test_dataset) == 0:
        return 0.0, float('inf')

    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else float('inf')
    return accuracy, avg_loss


# -----------------------------
# Comparison Logic
# -----------------------------
def compare_protocols(protocols_config: Dict, experiment_config: Dict) -> Dict:
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING PROTOCOL COMPARISON")
    print("=" * 70)

    client_datasets, test_dataset = generate_federated_data(
        num_clients=experiment_config['num_clients'],
        samples_per_client=experiment_config['samples_per_client'],
        input_dim=experiment_config['input_dim'],
        num_classes=experiment_config['num_classes'],
        heterogeneity=experiment_config['heterogeneity']
    )

    model_config = {
        'input_dim': experiment_config['input_dim'],
        'hidden_dim': experiment_config['hidden_dim'],
        'output_dim': experiment_config['num_classes']
    }

    results = {}
    for protocol_name, protocol_params in protocols_config.items():
        protocol = create_protocol(
            protocol_name.split("_")[0],  # e.g. improved_async
            num_clients=experiment_config['num_clients'],
            **protocol_params
        )
        protocol.set_global_model(SimpleNN(**model_config).state_dict())

        # Run simple loop
        start_time = time.time()
        while time.time() - start_time < experiment_config['duration']:
            for client_id in range(experiment_config['num_clients']):
                global_state = protocol.get_global_model()
                if global_state is None:
                    continue
                local_model = SimpleNN(**model_config)
                local_model.load_state_dict(global_state)

                updated_state, loss, data_size = train_client(
                    local_model, client_datasets[client_id], epochs=1, lr=0.01
                )

                update_dict = {}
                for param_name, param in updated_state.items():
                    if param_name in global_state and 'num_batches_tracked' not in param_name:
                        param_update = param.clone().float()
                        global_param = global_state[param_name].clone().float()
                        update_dict[param_name] = param_update - global_param

                update = ClientUpdate(
                    client_id=f"client_{client_id}",
                    update_data=update_dict,
                    model_version=protocol.model_version,
                    local_loss=loss,
                    data_size=data_size,
                    timestamp=time.time()
                )
                protocol.receive_update(update)

        # Evaluate
        final_model_state = protocol.get_global_model()
        eval_model = SimpleNN(**model_config)
        eval_model.load_state_dict(final_model_state)
        accuracy, loss = evaluate_model(eval_model, test_dataset)

        metrics = protocol.metrics.get_summary()
        metrics['final_accuracy'] = accuracy
        results[protocol_name] = metrics

        protocol.shutdown()
        print(f"\nResults for {protocol_name}:")
        print(f"  Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    return results


# -----------------------------
# Main
# -----------------------------
def main():
    """Main comparison experiment"""

    # Baseline protocols
    protocols_config = {
        'fedavg': {'participation_rate': 0.5, 'max_round_time': 10.0},
        'fedasync': {'max_staleness': 10, 'learning_rate': 0.8},
        'fedbuff': {'buffer_size': 5, 'max_staleness': 15},
        'scaffold': {'learning_rate': 0.8, 'max_round_time': 10.0}
    }

    # Add all Improved Async configs (scenario x compression)
    improved_async_configs = generate_all_configs()
    for name, cfg in improved_async_configs.items():
        protocols_config[f"improved_async_{name}"] = cfg

    experiment_config = {
        'num_clients': 50,
        'samples_per_client': 500,
        'input_dim': 12,
        'hidden_dim': 32,
        'num_classes': 3,
        'heterogeneity': 0.5,
        'duration': 360
    }

    results = compare_protocols(protocols_config, experiment_config)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for proto, metrics in results.items():
        print(f"{proto}: Final Acc={metrics['final_accuracy']:.4f}, "
              f"Comm={metrics['total_data_transmitted_mb']:.2f}MB, "
              f"Aggregations={metrics['aggregations_performed']}")


if __name__ == "__main__":
    main()
