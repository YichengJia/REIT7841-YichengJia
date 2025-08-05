"""
Enhanced Machine Learning Performance Test
enhanced_ml_performance.py
Focus: High-quality model training and convergence analysis
"""

import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

from improved_async_fed_protocol import SuperiorAsyncFedProtocol
import logging
import copy

# Set logging level
logging.getLogger().setLevel(logging.ERROR)

# Set random seed
torch.manual_seed(123)
np.random.seed(123)


class RobustNeuralNet(nn.Module):
    """Robust neural network model - optimized for federated learning"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(RobustNeuralNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_dim // 4, output_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


def create_high_quality_federated_data(num_clients=12, samples_per_client=150,
                                     input_dim=25, num_classes=5,
                                     heterogeneity=0.6, random_state=42):
    """
    Create high-quality federated learning dataset
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Generate high-quality classification data
    total_samples = num_clients * samples_per_client * 2  # Generate more data

    X, y = make_classification(
        n_samples=total_samples,
        n_features=input_dim,
        n_informative=max(3, input_dim // 3),  # More informative features
        n_redundant=max(1, input_dim // 10),   # Fewer redundant features
        n_classes=num_classes,
        n_clusters_per_class=2,  # 2 clusters per class for complexity
        class_sep=1.5,  # Increase class separation
        flip_y=0.01,  # 1% label noise
        random_state=random_state
    )

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Take required number of samples
    X = X[:num_clients * samples_per_client]
    y = y[:num_clients * samples_per_client]

    # Convert to tensor
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    # Create non-IID distribution
    client_datasets = []

    if heterogeneity < 0.1:  # IID case
        # Random allocation
        indices = torch.randperm(len(X))
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            client_indices = indices[start_idx:end_idx]

            client_X = X[client_indices]
            client_y = y[client_indices]
            client_datasets.append(TensorDataset(client_X, client_y))

    else:  # Non-IID case
        # Create realistic data distribution using Dirichlet distribution
        alpha = max(0.1, (1 - heterogeneity) * 5)

        # Group by class
        class_indices = [torch.where(y == c)[0].numpy() for c in range(num_classes)]

        # Generate class distribution for each client
        client_class_props = np.random.dirichlet([alpha] * num_classes, num_clients)

        for client_id in range(num_clients):
            client_indices = []
            props = client_class_props[client_id]

            for class_id in range(num_classes):
                n_samples = int(samples_per_client * props[class_id])
                if n_samples > 0 and len(class_indices[class_id]) >= n_samples:
                    # Randomly select samples from this class
                    selected = np.random.choice(
                        class_indices[class_id],
                        size=n_samples,
                        replace=False
                    )
                    client_indices.extend(selected)
                    # Remove from available samples
                    class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected)

            # Ensure each client has enough data
            while len(client_indices) < samples_per_client // 2:  # At least half the data
                # Randomly select from remaining data
                available = np.concatenate([idx for idx in class_indices if len(idx) > 0])
                if len(available) > 0:
                    additional = np.random.choice(available, size=1)
                    client_indices.extend(additional)
                    # Update class_indices
                    for class_id in range(num_classes):
                        class_indices[class_id] = np.setdiff1d(class_indices[class_id], additional)
                else:
                    break

            if len(client_indices) > 0:
                client_X = X[client_indices]
                client_y = y[client_indices]
                client_datasets.append(TensorDataset(client_X, client_y))
            else:
                # Emergency handling: create a small random dataset
                emergency_indices = torch.randperm(len(X))[:10]
                client_datasets.append(TensorDataset(X[emergency_indices], y[emergency_indices]))

    # Create global test set
    test_size = min(500, len(X) // 4)
    test_indices = torch.randperm(len(X))[:test_size]
    global_test_dataset = TensorDataset(X[test_indices], y[test_indices])

    return client_datasets, global_test_dataset


def advanced_client_training(model, dataset, epochs=3, lr=0.02, weight_decay=1e-4):
    """
    Advanced client training function
    """
    if len(dataset) == 0:
        return model.state_dict(), float('inf'), 0

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing

    dataloader = DataLoader(
        dataset,
        batch_size=min(64, max(8, len(dataset) // 4)),  # Dynamic batch size
        shuffle=True,
        drop_last=False
    )

    total_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if num_batches > 0:
            total_loss += epoch_loss / num_batches
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    return model.state_dict(), avg_loss, len(dataset)


def accurate_model_evaluation(model, test_dataset):
    """
    Accurate model evaluation
    """
    if len(test_dataset) == 0:
        return 0.0, float('inf')

    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            loss = criterion(outputs, y)

            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == y).sum().item()

            total_correct += correct
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

    return accuracy, avg_loss


def test_enhanced_ml_convergence():
    """Test enhanced model convergence performance"""
    print("\n" + "="*70)
    print("=== Enhanced Machine Learning Convergence Performance Test ===")
    print("="*70 + "\n")

    # Optimized parameter settings
    num_clients = 12
    samples_per_client = 120
    input_dim = 25
    hidden_dim = 128
    num_classes = 5
    training_duration = 90  # Increased to 90 seconds

    print(f"Test parameters:")
    print(f"- Number of clients: {num_clients}")
    print(f"- Samples per client: {samples_per_client}")
    print(f"- Network architecture: {input_dim}→{hidden_dim}→{hidden_dim//2}→{hidden_dim//4}→{num_classes}")
    print(f"- Training duration: {training_duration} seconds\n")

    # Generate high-quality data
    print("Generating high-quality federated data...")
    client_datasets, global_test_dataset = create_high_quality_federated_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        input_dim=input_dim,
        num_classes=num_classes,
        heterogeneity=0.7
    )

    print(f"Data statistics:")
    print(f"- Client data sizes: {[len(ds) for ds in client_datasets]}")
    print(f"- Global test set: {len(global_test_dataset)} samples")

    # Check data quality
    sample_X, sample_y = global_test_dataset.tensors
    print(f"- Feature range: [{sample_X.min():.3f}, {sample_X.max():.3f}]")
    print(f"- Class distribution: {torch.bincount(sample_y).tolist()}\n")

    # Initialize protocol
    base_model = RobustNeuralNet(input_dim, hidden_dim, num_classes)

    # Pre-train base model for better initialization
    print("Pre-training base model...")
    pretrain_data = TensorDataset(
        torch.cat([ds.tensors[0] for ds in client_datasets[:3]]),  # Use first 3 clients' data
        torch.cat([ds.tensors[1] for ds in client_datasets[:3]])
    )
    base_model.load_state_dict(advanced_client_training(base_model, pretrain_data, epochs=5, lr=0.001)[0])

    # Evaluate pre-training effect
    init_acc, init_loss = accurate_model_evaluation(base_model, global_test_dataset)
    print(f"Post pre-training accuracy: {init_acc:.4f}, loss: {init_loss:.4f}\n")

    async_protocol = SuperiorAsyncFedProtocol(
        max_staleness=25.0,
        min_buffer_size=2,
        max_buffer_size=5,
        adaptive_weighting=True,
        momentum=0.95,  # Increase momentum
        staleness_penalty='adaptive',
        learning_rate_decay=0.98
    )

    # Set initial model
    async_protocol.global_model = copy.deepcopy(base_model.state_dict())

    # Performance recording
    accuracies = []
    losses = []
    timestamps = []
    convergence_metrics = []

    # Client states
    client_versions = [0] * num_clients
    client_active_time = [0.0] * num_clients

    start_time = time.time()
    eval_lock = threading.Lock()

    print("Starting federated learning training...")

    # Evaluation thread
    stop_evaluation = threading.Event()

    def continuous_evaluation():
        eval_count = 0
        last_acc = 0.0

        while not stop_evaluation.is_set():
            current_time = time.time() - start_time

            if current_time > eval_count * 3.0:  # Evaluate every 3 seconds
                global_state = async_protocol.get_global_model()
                if global_state is not None:
                    with eval_lock:
                        eval_model = RobustNeuralNet(input_dim, hidden_dim, num_classes)
                        eval_model.load_state_dict(global_state)

                        acc, loss = accurate_model_evaluation(eval_model, global_test_dataset)

                        accuracies.append(acc)
                        losses.append(loss)
                        timestamps.append(current_time)

                        # Calculate convergence metrics
                        improvement = acc - last_acc if last_acc > 0 else 0
                        convergence_metrics.append(improvement)
                        last_acc = acc

                        eval_count += 1

                        if eval_count % 5 == 0:
                            stats = async_protocol.get_stats()
                            print(f"  [{current_time:.1f}s] Accuracy: {acc:.4f}, Loss: {loss:.4f}, "
                                  f"Improvement: {improvement:+.4f}, Aggregations: {stats['aggregations_performed']}")

            time.sleep(1.0)

    # Start evaluation thread
    eval_thread = threading.Thread(target=continuous_evaluation)
    eval_thread.start()

    # Client training function
    def intelligent_client_training(client_id):
        update_count = 0
        local_model = RobustNeuralNet(input_dim, hidden_dim, num_classes)

        # Client-specific learning rate
        client_lr = 0.02 * (0.8 + 0.4 * np.random.random())  # 0.016-0.024

        # Simulate client computational capability
        compute_speed = np.random.choice(['fast', 'medium', 'slow'], p=[0.4, 0.4, 0.2])
        epochs_per_round = {'fast': 4, 'medium': 3, 'slow': 2}[compute_speed]

        while time.time() - start_time < training_duration:
            # Simulate network fluctuations
            if np.random.random() < 0.15:  # 15% probability of network problems
                time.sleep(np.random.uniform(2, 5))
                continue

            # Check data availability
            if len(client_datasets[client_id]) == 0:
                time.sleep(1)
                continue

            # Get global model
            global_state = async_protocol.get_global_model()
            if global_state is None:
                time.sleep(0.5)
                continue

            # Load global model
            local_model.load_state_dict(global_state)

            # Local training
            training_start = time.time()
            updated_state, local_loss, data_size = advanced_client_training(
                local_model,
                client_datasets[client_id],
                epochs=epochs_per_round,
                lr=client_lr,
                weight_decay=1e-4
            )
            training_time = time.time() - training_start
            client_active_time[client_id] += training_time

            # Calculate model updates
            update_dict = {}
            for name, param in updated_state.items():
                update_dict[name] = param - global_state[name]

            # Submit update
            accepted, new_version = async_protocol.submit_update(
                f"client_{client_id}",
                update_dict,
                client_versions[client_id],
                local_loss,
                data_size
            )

            if accepted:
                client_versions[client_id] = new_version
                update_count += 1

            # Adaptive rest time
            rest_time = {
                'fast': np.random.uniform(0.5, 1.5),
                'medium': np.random.uniform(1.0, 2.5),
                'slow': np.random.uniform(2.0, 4.0)
            }[compute_speed]

            time.sleep(rest_time)

        print(f"  Client{client_id} completed: {update_count} updates, active time: {client_active_time[client_id]:.1f}s")

    # Start all client threads
    client_threads = []
    for client_id in range(num_clients):
        thread = threading.Thread(target=intelligent_client_training, args=(client_id,))
        thread.start()
        client_threads.append(thread)

    # Wait for training completion
    time.sleep(training_duration)
    stop_evaluation.set()

    # Wait for all threads to complete
    eval_thread.join()
    for thread in client_threads:
        thread.join(timeout=3.0)

    # Get final statistics
    final_stats = async_protocol.get_stats()
    async_protocol.shutdown()

    # Calculate advanced metrics
    final_accuracy = accuracies[-1] if accuracies else 0.0
    max_accuracy = max(accuracies) if accuracies else 0.0
    accuracy_improvement = final_accuracy - init_acc

    # Convergence analysis
    convergence_rate = np.mean([abs(c) for c in convergence_metrics[-10:]]) if len(convergence_metrics) >= 10 else 0
    stability = 1.0 - np.std(accuracies[-10:]) if len(accuracies) >= 10 else 0

    print("\n" + "="*70)
    print("=== Training Results Analysis ===")
    print("="*70)

    print(f"\nPerformance metrics:")
    print(f"  - Initial accuracy: {init_acc:.4f}")
    print(f"  - Final accuracy: {final_accuracy:.4f}")
    print(f"  - Max accuracy: {max_accuracy:.4f}")
    print(f"  - Accuracy improvement: {accuracy_improvement:+.4f}")

    print(f"\nConvergence analysis:")
    print(f"  - Convergence speed: {convergence_rate:.6f}")
    print(f"  - Model stability: {stability:.4f}")
    print(f"  - Total aggregations: {final_stats['aggregations_performed']}")

    print(f"\nProtocol statistics:")
    print(f"  - Accepted updates: {final_stats['accepted_updates']}")
    print(f"  - Rejected updates: {final_stats['rejected_updates']}")
    print(f"  - High quality update ratio: {final_stats.get('high_quality_ratio', 0):.3f}")
    print(f"  - Network health: {final_stats.get('network_health', 0):.3f}")

    # Create convergence plots
    create_convergence_plots(accuracies, losses, timestamps, convergence_metrics)

    return {
        'accuracies': accuracies,
        'losses': losses,
        'timestamps': timestamps,
        'final_accuracy': final_accuracy,
        'max_accuracy': max_accuracy,
        'convergence_metrics': convergence_metrics,
        'stats': final_stats
    }


def test_heterogeneity_robustness():
    """Test robustness in heterogeneous environments"""
    print("\n" + "="*70)
    print("=== Heterogeneous Environment Robustness Test ===")
    print("="*70 + "\n")

    heterogeneity_levels = [0.2, 0.5, 0.8]  # Low, medium, high heterogeneity
    results = {}

    base_params = {
        'num_clients': 10,
        'samples_per_client': 100,
        'input_dim': 20,
        'num_classes': 4,
        'training_duration': 45
    }

    for hetero_level in heterogeneity_levels:
        print(f"\nTesting heterogeneity level: {hetero_level:.1f} ({'Low' if hetero_level < 0.4 else 'Medium' if hetero_level < 0.7 else 'High'})")

        # Generate data
        client_datasets, global_test_dataset = create_high_quality_federated_data(
            num_clients=base_params['num_clients'],
            samples_per_client=base_params['samples_per_client'],
            input_dim=base_params['input_dim'],
            num_classes=base_params['num_classes'],
            heterogeneity=hetero_level
        )

        # Initialize model and protocol
        base_model = RobustNeuralNet(
            base_params['input_dim'],
            96,
            base_params['num_classes']
        )

        async_protocol = SuperiorAsyncFedProtocol(
            max_staleness=20.0,
            min_buffer_size=2,
            max_buffer_size=4,
            adaptive_weighting=True,
            momentum=0.9
        )

        async_protocol.global_model = base_model.state_dict()

        # Short-term training test
        start_time = time.time()
        client_versions = [0] * base_params['num_clients']

        # Simulate training process
        for round_num in range(15):  # 15 rounds quick test
            participating_clients = np.random.choice(
                base_params['num_clients'],
                size=max(2, base_params['num_clients'] // 2),  # 50% participation rate
                replace=False
            )

            for client_id in participating_clients:
                if len(client_datasets[client_id]) > 0:
                    # Get global model
                    global_state = async_protocol.get_global_model()
                    if global_state is None:
                        continue

                    # Create local model
                    local_model = RobustNeuralNet(
                        base_params['input_dim'], 96, base_params['num_classes']
                    )
                    local_model.load_state_dict(global_state)

                    # Training
                    updated_state, local_loss, data_size = advanced_client_training(
                        local_model, client_datasets[client_id], epochs=2, lr=0.02
                    )

                    # Calculate updates
                    update_dict = {}
                    for name, param in updated_state.items():
                        update_dict[name] = param - global_state[name]

                    # Submit update
                    accepted, new_version = async_protocol.submit_update(
                        f"client_{client_id}",
                        update_dict,
                        client_versions[client_id],
                        local_loss,
                        data_size
                    )

                    if accepted:
                        client_versions[client_id] = new_version

            time.sleep(0.2)  # Brief rest

        # Wait for aggregation completion
        time.sleep(2.0)

        # Final evaluation
        final_global_state = async_protocol.get_global_model()
        if final_global_state:
            eval_model = RobustNeuralNet(
                base_params['input_dim'], 96, base_params['num_classes']
            )
            eval_model.load_state_dict(final_global_state)
            final_acc, final_loss = accurate_model_evaluation(eval_model, global_test_dataset)
        else:
            final_acc, final_loss = 0.0, float('inf')

        # Collect statistics
        stats = async_protocol.get_stats()
        async_protocol.shutdown()

        results[hetero_level] = {
            'final_accuracy': final_acc,
            'final_loss': final_loss,
            'aggregations': stats['aggregations_performed'],
            'accepted_updates': stats['accepted_updates'],
            'high_quality_ratio': stats.get('high_quality_ratio', 0)
        }

        print(f"  Final accuracy: {final_acc:.4f}")
        print(f"  Aggregation count: {stats['aggregations_performed']}")
        print(f"  High quality update ratio: {stats.get('high_quality_ratio', 0):.3f}")

    # Create heterogeneity analysis plots
    create_heterogeneity_plots(results)

    return results


def create_convergence_plots(accuracies, losses, timestamps, convergence_metrics):
    """Create detailed convergence analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning Convergence Performance Analysis', fontsize=14, fontweight='bold')

    # 1. Accuracy over time
    ax1 = axes[0, 0]
    ax1.plot(timestamps, accuracies, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Convergence Curve')
    ax1.grid(True, alpha=0.3)

    # Add trend line
    if len(timestamps) > 5:
        z = np.polyfit(timestamps, accuracies, 2)
        p = np.poly1d(z)
        ax1.plot(timestamps, p(timestamps), "r--", alpha=0.8, label='Trend Line')
        ax1.legend()

    # 2. Loss over time
    ax2 = axes[0, 1]
    ax2.plot(timestamps, losses, 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss Convergence Curve')
    ax2.grid(True, alpha=0.3)

    # 3. Convergence speed analysis
    ax3 = axes[1, 0]
    if len(convergence_metrics) > 1:
        # Calculate moving average
        window_size = min(5, len(convergence_metrics) // 2)
        if window_size > 0:
            moving_avg = np.convolve(convergence_metrics, np.ones(window_size)/window_size, mode='valid')
            ax3.plot(range(len(moving_avg)), moving_avg, 'g-', linewidth=2)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Evaluation Round')
            ax3.set_ylabel('Accuracy Improvement')
            ax3.set_title('Convergence Speed Analysis')
            ax3.grid(True, alpha=0.3)

    # 4. Performance distribution
    ax4 = axes[1, 1]
    if len(accuracies) > 10:
        # Accuracy distribution histogram
        ax4.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(np.mean(accuracies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(accuracies):.4f}')
        ax4.axvline(np.median(accuracies), color='green', linestyle='--',
                   label=f'Median: {np.median(accuracies):.4f}')
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Accuracy Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_ml_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nConvergence analysis chart saved as 'enhanced_ml_convergence.png'")


def create_heterogeneity_plots(results):
    """Create heterogeneity analysis charts"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Protocol Robustness Analysis in Heterogeneous Environments', fontsize=14, fontweight='bold')

    hetero_levels = list(results.keys())
    hetero_labels = [f'{h:.1f}' for h in hetero_levels]

    # 1. Final accuracy
    accuracies = [results[h]['final_accuracy'] for h in hetero_levels]
    axes[0].plot(hetero_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Heterogeneity Level')
    axes[0].set_ylabel('Final Accuracy')
    axes[0].set_title('Accuracy vs Data Heterogeneity')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(hetero_levels)
    axes[0].set_xticklabels(hetero_labels)

    # 2. Aggregation efficiency
    aggregations = [results[h]['aggregations'] for h in hetero_levels]
    axes[1].bar(range(len(hetero_levels)), aggregations, color='green', alpha=0.7)
    axes[1].set_xlabel('Heterogeneity Level')
    axes[1].set_ylabel('Aggregation Count')
    axes[1].set_title('Aggregation Efficiency vs Data Heterogeneity')
    axes[1].set_xticks(range(len(hetero_levels)))
    axes[1].set_xticklabels(hetero_labels)

    # 3. Update quality
    quality_ratios = [results[h]['high_quality_ratio'] for h in hetero_levels]
    axes[2].plot(hetero_levels, quality_ratios, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Heterogeneity Level')
    axes[2].set_ylabel('High Quality Update Ratio')
    axes[2].set_title('Update Quality vs Data Heterogeneity')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(hetero_levels)
    axes[2].set_xticklabels(hetero_labels)

    plt.tight_layout()
    plt.savefig('heterogeneity_robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nHeterogeneity analysis chart saved as 'heterogeneity_robustness_analysis.png'")


if __name__ == "__main__":
    print("Starting enhanced machine learning performance test...")

    # Main convergence test
    convergence_results = test_enhanced_ml_convergence()

    # Heterogeneity robustness test
    print("\n" + "="*70)
    heterogeneity_results = test_heterogeneity_robustness()

    print("\n" + "="*70)
    print("Enhanced ML test completed!")
    print("Generated files:")
    print("  - enhanced_ml_convergence.png: Convergence performance analysis")
    print("  - heterogeneity_robustness_analysis.png: Heterogeneous environment robustness")
    print("="*70)