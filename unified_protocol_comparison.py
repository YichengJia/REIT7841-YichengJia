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
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

from federated_protocol_framework import (
    create_protocol, ClientUpdate, FederatedProtocol
)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class SimpleNN(nn.Module):
    """Simple neural network for testing"""
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


def generate_federated_data(num_clients: int, samples_per_client: int,
                           input_dim: int, num_classes: int,
                           heterogeneity: float = 0.5) -> Tuple:
    """Generate federated dataset with heterogeneity"""
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

    if heterogeneity < 0.1:
        # IID distribution
        indices = torch.randperm(total_samples)
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            client_indices = indices[start_idx:end_idx]
            client_X = X[client_indices]
            client_y = y[client_indices]
            client_datasets.append(TensorDataset(client_X, client_y))
    else:
        # Non-IID distribution
        # Sort by labels
        sorted_indices = torch.argsort(y)
        sorted_X = X[sorted_indices]
        sorted_y = y[sorted_indices]

        # Distribute with skew
        for i in range(num_clients):
            # Each client gets more samples from certain classes
            primary_class = i % num_classes
            primary_ratio = 0.5 + heterogeneity * 0.4

            client_indices = []
            for c in range(num_classes):
                class_mask = sorted_y == c
                class_indices = torch.where(class_mask)[0]

                if c == primary_class:
                    num_samples = int(samples_per_client * primary_ratio)
                else:
                    num_samples = int(samples_per_client * (1 - primary_ratio) / (num_classes - 1))

                if len(class_indices) > 0:
                    selected = np.random.choice(
                        class_indices.numpy(),
                        size=min(num_samples, len(class_indices)),
                        replace=False
                    )
                    client_indices.extend(selected)

            client_X = sorted_X[client_indices[:samples_per_client]]
            client_y = sorted_y[client_indices[:samples_per_client]]
            client_datasets.append(TensorDataset(client_X, client_y))

    # Create global test set
    test_size = min(len(X) // 5, 500)
    test_indices = np.random.choice(len(X), test_size, replace=False)
    test_X = X[test_indices]
    test_y = y[test_indices]
    global_test_dataset = TensorDataset(test_X, test_y)

    return client_datasets, global_test_dataset


def train_client(model: nn.Module, dataset: TensorDataset,
                epochs: int = 3, lr: float = 0.01) -> Tuple:
    """Train client model and return update"""
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

            # Add gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        total_loss += epoch_loss / len(dataloader)

    avg_loss = total_loss / epochs
    return model.state_dict(), avg_loss, len(dataset)

def train_client_scaffold(model: nn.Module, dataset: TensorDataset,
                          c_global: Dict[str, torch.Tensor],
                          c_local: Dict[str, torch.Tensor],
                          epochs: int = 3, lr: float = 0.01) -> Tuple:
    """Train client with SCAFFOLD control variates"""
    if len(dataset) == 0:
        return model.state_dict(), float('inf'), 0, c_local

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
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

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in c_global:
                        grad = param.grad
                        corr = grad - c_local.get(name, torch.zeros_like(param)) + c_global[name]
                        param -= lr * corr
                        # update control variate
                        c_local[name] = c_global[name].clone()

            epoch_loss += loss.item()
        total_loss += epoch_loss / len(dataloader)

    avg_loss = total_loss / epochs
    return model.state_dict(), avg_loss, len(dataset), c_local


def evaluate_model(model: nn.Module, test_dataset: TensorDataset) -> Tuple[float, float]:
    """Evaluate model on test dataset"""
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
            try:
                outputs = model(X)
                loss = criterion(outputs, y)

                # Check for NaN
                if torch.isnan(loss):
                    continue

                total_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 and not torch.isnan(torch.tensor(total_loss)) else float('inf')

    # Additional check for NaN
    if torch.isnan(torch.tensor(avg_loss)):
        avg_loss = float('inf')

    return accuracy, avg_loss


class ProtocolTester:
    """Unified protocol testing framework"""

    def __init__(self, protocol: FederatedProtocol,
                 client_datasets: List[TensorDataset],
                 test_dataset: TensorDataset,
                 model_config: Dict):
        self.protocol = protocol
        self.client_datasets = client_datasets
        self.test_dataset = test_dataset
        self.model_config = model_config
        self.num_clients = len(client_datasets)

        # Client states
        self.client_versions = [0] * self.num_clients
        self.client_active = [True] * self.num_clients

    def simulate_client(self, client_id: int, duration: float,
                        client_type: str = 'normal'):
        """Simulate single client behavior"""
        start_time = time.time()
        update_count = 0

        # Client-specific parameters
        if client_type == 'fast':
            delay_range = (0.1, 0.3)
            dropout_prob = 0.05
            epochs = 3
        elif client_type == 'slow':
            delay_range = (1.0, 2.0)
            dropout_prob = 0.25
            epochs = 2
        else:  # normal
            delay_range = (0.3, 0.8)
            dropout_prob = 0.15
            epochs = 2

        # Scaffold needs local control variates
        c_local = {}

        while time.time() - start_time < duration and self.client_active[client_id]:
            if np.random.random() < dropout_prob:
                time.sleep(np.random.uniform(1.0, 3.0))
                continue

            global_model_state = self.protocol.get_global_model()
            if global_model_state is None:
                time.sleep(0.5)
                continue

            local_model = SimpleNN(**self.model_config)
            local_model.load_state_dict(global_model_state)

            # === 鍗忚鍒嗘敮 ===
            if self.protocol.__class__.__name__.lower() == "scaffold":
                c_global = getattr(self.protocol, "c_global", {})
                updated_state, local_loss, data_size, c_local = train_client_scaffold(
                    local_model, self.client_datasets[client_id],
                    c_global, c_local, epochs=epochs, lr=0.01
                )
            else:
                updated_state, local_loss, data_size = train_client(
                    local_model, self.client_datasets[client_id],
                    epochs=epochs, lr=0.01
                )

            # Calculate update (宸垎鍙傛暟)
            update_dict = {}
            for name, param in updated_state.items():
                if name in global_model_state:
                    if 'num_batches_tracked' in name:
                        continue
                    param_update = param.clone().float()
                    global_param = global_model_state[name].clone().float()
                    update_dict[name] = param_update - global_param

            update = ClientUpdate(
                client_id=f"client_{client_id}",
                update_data=update_dict,
                model_version=self.client_versions[client_id],
                local_loss=local_loss,
                data_size=data_size,
                timestamp=time.time()
            )

            accepted, new_version = self.protocol.receive_update(update)
            self.client_versions[client_id] = new_version
            update_count += 1

            time.sleep(np.random.uniform(*delay_range))

        return update_count

    def run_experiment(self, duration: float, eval_interval: float = 2.0) -> Dict:
        """Run complete experiment"""
        print(f"\nTesting {self.protocol.__class__.__name__}")
        print("-" * 50)

        # Initialize global model
        initial_model = SimpleNN(**self.model_config)
        self.protocol.set_global_model(initial_model.state_dict())

        # Assign client types (30% fast, 40% normal, 30% slow)
        num_fast = int(self.num_clients * 0.3)
        num_slow = int(self.num_clients * 0.3)
        num_normal = self.num_clients - num_fast - num_slow

        client_types = (['fast'] * num_fast +
                        ['normal'] * num_normal +
                        ['slow'] * num_slow)
        np.random.shuffle(client_types)

        # Start evaluation thread
        eval_results = {'accuracies': [], 'losses': [], 'timestamps': []}
        stop_eval = threading.Event()

        def evaluate_periodically():
            start_time = time.time()
            while not stop_eval.is_set():
                current_time = time.time() - start_time
                if current_time >= len(eval_results['timestamps']) * eval_interval:
                    global_model_state = self.protocol.get_global_model()
                    if global_model_state:
                        eval_model = SimpleNN(**self.model_config)
                        eval_model.load_state_dict(global_model_state)
                        accuracy, loss = evaluate_model(eval_model, self.test_dataset)

                        if accuracy > 0 and np.isfinite(loss):
                            eval_results['accuracies'].append(accuracy)
                            eval_results['losses'].append(loss)
                            eval_results['timestamps'].append(current_time)
                            self.protocol.metrics.update_performance(accuracy, loss, current_time)

                            if len(eval_results['accuracies']) % 5 == 0:
                                print(f"  [{current_time:.1f}s] Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
                time.sleep(0.5)

        eval_thread = threading.Thread(target=evaluate_periodically)
        eval_thread.start()

        # Start client threads
        client_threads = []
        for client_id in range(self.num_clients):
            thread = threading.Thread(
                target=self.simulate_client,
                args=(client_id, duration, client_types[client_id])
            )
            thread.start()
            client_threads.append(thread)

        # Wait for experiment to complete
        time.sleep(duration)

        # Stop evaluation
        stop_eval.set()
        eval_thread.join()

        # Stop clients
        self.client_active = [False] * self.num_clients
        for thread in client_threads:
            thread.join(timeout=2.0)

        # Shutdown protocol
        self.protocol.shutdown()

        # Get metrics
        final_metrics = self.protocol.metrics.get_summary()
        final_metrics.update(eval_results)

        global_model_state = self.protocol.get_global_model()
        if global_model_state:
            eval_model = SimpleNN(**self.model_config)
            eval_model.load_state_dict(global_model_state)
            accuracy, loss = evaluate_model(eval_model, self.test_dataset)

            if accuracy > 0 and np.isfinite(loss):
                final_metrics['final_accuracy'] = accuracy
                final_metrics['final_loss'] = loss
            else:
                print(" Final evaluation produced invalid result, keeping last valid metrics.")

        print(f"\nResults for {self.protocol.__class__.__name__}:")
        print(f"  Final Accuracy: {final_metrics['final_accuracy']:.4f}")
        print(f"  Max Accuracy: {final_metrics['max_accuracy']:.4f}")
        print(f"  Total Communication: {final_metrics['total_data_transmitted_mb']:.2f} MB")
        print(f"  Updates Accepted: {final_metrics['total_updates_accepted']}")
        print(f"  Aggregations: {final_metrics['aggregations_performed']}")
        print(f"  Throughput: {final_metrics['throughput_updates_per_second']:.2f} updates/s")

        return final_metrics


def compare_protocols(protocols_config: Dict, experiment_config: Dict) -> Dict:
    """Compare multiple protocols"""
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING PROTOCOL COMPARISON")
    print("=" * 70)

    # Generate data
    print("\nGenerating federated dataset...")
    client_datasets, test_dataset = generate_federated_data(
        num_clients=experiment_config['num_clients'],
        samples_per_client=experiment_config['samples_per_client'],
        input_dim=experiment_config['input_dim'],
        num_classes=experiment_config['num_classes'],
        heterogeneity=experiment_config['heterogeneity']
    )
    print(f"  Clients: {len(client_datasets)}")
    print(f"  Samples per client: {len(client_datasets[0])}")
    print(f"  Test set size: {len(test_dataset)}")
    print(f"  Heterogeneity: {experiment_config['heterogeneity']}")

    # Model configuration
    model_config = {
        'input_dim': experiment_config['input_dim'],
        'hidden_dim': experiment_config['hidden_dim'],
        'output_dim': experiment_config['num_classes']
    }

    # Run experiments
    results = {}
    for protocol_name, protocol_params in protocols_config.items():
        # Create protocol
        protocol = create_protocol(
            protocol_name,
            num_clients=experiment_config['num_clients'],
            **protocol_params
        )

        # Create tester
        tester = ProtocolTester(
            protocol=protocol,
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            model_config=model_config
        )

        # Run experiment
        protocol_results = tester.run_experiment(
            duration=experiment_config['duration'],
            eval_interval=experiment_config['eval_interval']
        )

        results[protocol_name] = protocol_results

    return results


def create_comparison_plots(results: Dict):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Federated Learning Protocol Comparison', fontsize=16, fontweight='bold')

    protocol_names = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(protocol_names)))

    # 1. Accuracy over time
    ax1 = axes[0, 0]
    for i, (name, data) in enumerate(results.items()):
        if 'timestamps' in data and 'accuracies' in data:
            ax1.plot(data['timestamps'], data['accuracies'],
                    label=name.upper(), color=colors[i], linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Loss over time
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        if 'timestamps' in data and 'losses' in data:
            ax2.plot(data['timestamps'], data['losses'],
                    label=name.upper(), color=colors[i], linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Communication efficiency
    ax3 = axes[0, 2]
    comm_data = [results[name]['total_data_transmitted_mb'] for name in protocol_names]
    acc_data = [results[name]['final_accuracy'] for name in protocol_names]

    # Efficiency = Accuracy / Communication
    efficiency = [acc/comm if comm > 0 else 0 for acc, comm in zip(acc_data, comm_data)]

    bars = ax3.bar(protocol_names, efficiency, color=colors, alpha=0.7)
    ax3.set_ylabel('Efficiency (Accuracy/MB)')
    ax3.set_title('Communication Efficiency')
    ax3.set_xticklabels([name.upper() for name in protocol_names], rotation=45)

    # Add value labels
    for bar, eff in zip(bars, efficiency):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{eff:.4f}', ha='center', va='bottom')

    # 4. Final accuracy comparison
    ax4 = axes[1, 0]
    final_accs = [results[name]['final_accuracy'] for name in protocol_names]
    max_accs = [results[name]['max_accuracy'] for name in protocol_names]

    x = np.arange(len(protocol_names))
    width = 0.35

    bars1 = ax4.bar(x - width/2, final_accs, width, label='Final', color='skyblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, max_accs, width, label='Max', color='lightcoral', alpha=0.8)

    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.upper() for name in protocol_names], rotation=45)
    ax4.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 5. Throughput comparison
    ax5 = axes[1, 1]
    throughput = [results[name]['throughput_updates_per_second'] for name in protocol_names]
    bars = ax5.bar(protocol_names, throughput, color=colors, alpha=0.7)
    ax5.set_ylabel('Updates per Second')
    ax5.set_title('Throughput Comparison')
    ax5.set_xticklabels([name.upper() for name in protocol_names], rotation=45)

    for bar, val in zip(bars, throughput):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom')

    # 6. Overall performance score
    ax6 = axes[1, 2]

    # Calculate normalized scores
    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val - min_val == 0:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    # Metrics for scoring (higher is better)
    norm_acc = normalize([results[n]['final_accuracy'] for n in protocol_names])
    norm_eff = normalize(efficiency)
    norm_thr = normalize(throughput)

    # Calculate overall score
    overall_scores = [0.5*acc + 0.3*eff + 0.2*thr
                     for acc, eff, thr in zip(norm_acc, norm_eff, norm_thr)]

    bars = ax6.bar(protocol_names, overall_scores, color=colors, alpha=0.7)
    ax6.set_ylabel('Overall Score')
    ax6.set_title('Overall Performance Score')
    ax6.set_xticklabels([name.upper() for name in protocol_names], rotation=45)
    ax6.set_ylim(0, 1.2)

    for bar, score in zip(bars, overall_scores):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('protocol_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nComparison plots saved as 'protocol_comparison_results.png'")


def main():
    """Main comparison experiment"""

    # Protocol configurations
    protocols_config = {
        'fedavg': {
            'participation_rate': 0.5,
            'max_round_time': 10.0
        },
        'fedasync': {
            'max_staleness': 10,
            'learning_rate': 0.8
        },
        'fedbuff': {
            'buffer_size': 5,
            'max_staleness': 15
        },
        'improved_async': {
            'max_staleness': 30,
            'min_buffer_size': 4,
            'max_buffer_size': 8,
            'adaptive_weighting': True,
            'momentum': 0.9,
            'compression_ratio': 0.8,
            'server_lr': 0.1
        },
        'scaffold': {
            'learning_rate': 0.8,
            'max_round_time': 10.0
        }
    }

    # Experiment configuration
    experiment_config = {
        'num_clients': 50,
        'samples_per_client': 200,
        'input_dim': 20,
        'hidden_dim': 64,
        'num_classes': 4,
        'heterogeneity': 0.6,
        'duration': 300,
        'eval_interval': 2.0
    }

    # Run comparison
    results = compare_protocols(protocols_config, experiment_config)

    # Create plots
    create_comparison_plots(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find best protocol for each metric
    metrics_to_compare = [
        ('final_accuracy', 'Final Accuracy', True),
        ('total_data_transmitted_mb', 'Communication Cost', False),
        ('throughput_updates_per_second', 'Throughput', True)
    ]

    for metric, name, higher_better in metrics_to_compare:
        values = {proto: results[proto][metric] for proto in results}
        if higher_better:
            best = max(values, key=values.get)
        else:
            best = min(values, key=values.get)

        print(f"\nBest {name}: {best.upper()}")
        for proto, val in values.items():
            print(f"  {proto.upper()}: {val:.4f}")

    print("\n" + "=" * 70)
    print("Experiment completed successfully!")

def run_debug_test():
    """Debug version with extensive logging"""
    import logging
    logging.basicConfig(level=logging.DEBUG)

    print("\n" + "="*70)
    print("RUNNING DEBUG TEST")
    print("="*70)

    # Test with minimal configuration
    experiment_config = {
        'num_clients': 5,  # Reduced
        'samples_per_client': 50,  # Reduced
        'input_dim': 10,
        'hidden_dim': 32,
        'num_classes': 3,
        'heterogeneity': 0.3,
        'duration': 30,  # Short test
        'eval_interval': 5.0
    }

    print(f"Configuration: {experiment_config}")

    # Test data generation
    print("\nTesting data generation...")
    try:
        client_datasets, test_dataset = generate_federated_data(
            num_clients=experiment_config['num_clients'],
            samples_per_client=experiment_config['samples_per_client'],
            input_dim=experiment_config['input_dim'],
            num_classes=experiment_config['num_classes'],
            heterogeneity=experiment_config['heterogeneity']
        )
        print(f"? Data generated successfully")
        print(f"  Client datasets: {len(client_datasets)}")
        print(f"  Test dataset size: {len(test_dataset)}")
    except Exception as e:
        print(f"? Data generation failed: {e}")
        return

    # Test single protocol
    print("\nTesting FedAvg protocol...")
    try:
        protocol = create_protocol(
            'fedavg',
            num_clients=experiment_config['num_clients'],
            participation_rate=0.5
        )
        print("? Protocol created")

        # Set initial model
        model_config = {
            'input_dim': experiment_config['input_dim'],
            'hidden_dim': experiment_config['hidden_dim'],
            'output_dim': experiment_config['num_classes']
        }
        initial_model = SimpleNN(**model_config)
        protocol.set_global_model(initial_model.state_dict())
        print("? Initial model set")

        # Test single client update
        print("\nTesting single client update...")
        global_state = protocol.get_global_model()
        local_model = SimpleNN(**model_config)
        local_model.load_state_dict(global_state)

        updated_state, loss, data_size = train_client(
            local_model, client_datasets[0], epochs=1, lr=0.01
        )
        print(f"? Client training completed")
        print(f"  Loss: {loss:.4f}")
        print(f"  Data size: {data_size}")

        # Calculate update
        update_dict = {}
        for name, param in updated_state.items():
            if name in global_state and 'num_batches_tracked' not in name:
                update_dict[name] = param - global_state[name]

        # Submit update
        update = ClientUpdate(
            client_id="test_client",
            update_data=update_dict,
            model_version=0,
            local_loss=loss,
            data_size=data_size,
            timestamp=time.time()
        )

        accepted, version = protocol.receive_update(update)
        print(f"? Update submitted: accepted={accepted}, version={version}")

        # Check metrics
        metrics = protocol.metrics.get_summary()
        print(f"\nMetrics after single update:")
        print(f"  Updates accepted: {metrics['total_updates_accepted']}")
        print(f"  Data transmitted: {metrics['total_data_transmitted_mb']:.4f} MB")

        protocol.shutdown()

    except Exception as e:
        print(f"? Protocol test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

