"""
Optimized Parameter Analysis Test
optimized_parameter_analysis.py
Focus: Finding optimal protocol parameters, proving superiority of async protocol
"""

import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

from improved_async_fed_protocol import SuperiorAsyncFedProtocol
from enhanced_ml_performance import (
    RobustNeuralNet, create_high_quality_federated_data,
    advanced_client_training, accurate_model_evaluation
)
import logging

# Set logging level
logging.getLogger().setLevel(logging.ERROR)

# Set random seed
torch.manual_seed(456)
np.random.seed(456)


def comprehensive_staleness_analysis():
    """Comprehensive staleness parameter analysis"""
    print("\n" + "="*70)
    print("=== Comprehensive Staleness (Max Staleness) Parameter Analysis ===")
    print("="*70 + "\n")

    # More reasonable staleness range
    staleness_values = [5, 10, 15, 20, 30, 50]  # From small to large
    results = {}

    # Test parameters
    num_clients = 10
    samples_per_client = 80
    input_dim = 20
    hidden_dim = 64
    num_classes = 4
    training_duration = 40  # Sufficient training time

    print(f"Test parameters:")
    print(f"- Staleness range: {staleness_values}")
    print(f"- Number of clients: {num_clients}")
    print(f"- Training duration: {training_duration} seconds")
    print(f"- Network environment: Simulating 30% high latency, 20% intermittent interruptions\n")

    # Generate stable dataset
    client_datasets, global_test_dataset = create_high_quality_federated_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        input_dim=input_dim,
        num_classes=num_classes,
        heterogeneity=0.6,
        random_state=42
    )

    for max_staleness in staleness_values:
        print(f"\nTesting max_staleness = {max_staleness}")

        # Initialize protocol
        async_protocol = SuperiorAsyncFedProtocol(
            max_staleness=float(max_staleness),
            min_buffer_size=2,
            max_buffer_size=4,
            adaptive_weighting=True,
            momentum=0.9,
            staleness_penalty='adaptive'
        )

        # Initialize model
        base_model = RobustNeuralNet(input_dim, hidden_dim, num_classes)
        async_protocol.global_model = base_model.state_dict()

        # Performance recording
        accuracies = []
        timestamps = []
        convergence_points = []

        # Client states
        client_versions = [0] * num_clients
        client_delays = {
            'fast': lambda: np.random.uniform(0.1, 0.3),
            'medium': lambda: np.random.uniform(0.5, 1.0),
            'slow': lambda: np.random.uniform(1.5, 3.0)
        }

        # Assign type to each client
        client_types = (['fast'] * 4 + ['medium'] * 3 + ['slow'] * 3)[:num_clients]
        np.random.shuffle(client_types)

        start_time = time.time()
        eval_interval = 2.0
        last_eval = 0

        # Evaluation function
        def evaluate_periodically():
            nonlocal last_eval
            while time.time() - start_time < training_duration:
                current_time = time.time() - start_time
                if current_time - last_eval >= eval_interval:
                    global_state = async_protocol.get_global_model()
                    if global_state:
                        eval_model = RobustNeuralNet(input_dim, hidden_dim, num_classes)
                        eval_model.load_state_dict(global_state)
                        acc, _ = accurate_model_evaluation(eval_model, global_test_dataset)

                        accuracies.append(acc)
                        timestamps.append(current_time)

                        # Detect convergence points
                        if len(accuracies) >= 5:
                            recent_trend = np.polyfit(range(-5, 0), accuracies[-5:], 1)[0]
                            if abs(recent_trend) < 0.001:  # Convergence threshold
                                convergence_points.append(current_time)

                        last_eval = current_time

                time.sleep(1.0)

        # Start evaluation thread
        eval_thread = threading.Thread(target=evaluate_periodically)
        eval_thread.start()

        # Client training simulation
        def simulate_realistic_client(client_id):
            client_type = client_types[client_id]
            update_count = 0
            staleness_encounters = 0

            while time.time() - start_time < training_duration:
                # Simulate network problems
                if np.random.random() < 0.2:  # 20% probability of network interruption
                    time.sleep(np.random.uniform(1, 3))
                    continue

                # Get global model
                global_state = async_protocol.get_global_model()
                if global_state is None:
                    time.sleep(0.5)
                    continue

                # Local training
                local_model = RobustNeuralNet(input_dim, hidden_dim, num_classes)
                local_model.load_state_dict(global_state)

                # Client-specific training parameters
                epochs = {'fast': 3, 'medium': 2, 'slow': 2}[client_type]
                lr = {'fast': 0.02, 'medium': 0.015, 'slow': 0.01}[client_type]

                updated_state, local_loss, data_size = advanced_client_training(
                    local_model, client_datasets[client_id], epochs=epochs, lr=lr
                )

                # Calculate updates
                update_dict = {}
                for name, param in updated_state.items():
                    update_dict[name] = param - global_state[name]

                # Simulate transmission delay
                transmission_delay = client_delays[client_type]()
                time.sleep(transmission_delay * 0.1)  # Scale time

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
                else:
                    staleness_encounters += 1

        # Start all clients
        client_threads = []
        for client_id in range(num_clients):
            thread = threading.Thread(target=simulate_realistic_client, args=(client_id,))
            thread.start()
            client_threads.append(thread)

        # Wait for training completion
        time.sleep(training_duration)

        # Wait for threads to finish
        eval_thread.join()
        for thread in client_threads:
            thread.join(timeout=2.0)

        # Collect statistics
        stats = async_protocol.get_stats()
        async_protocol.shutdown()

        # Calculate key metrics
        final_accuracy = accuracies[-1] if accuracies else 0.0
        max_accuracy = max(accuracies) if accuracies else 0.0
        convergence_time = convergence_points[0] if convergence_points else training_duration

        # Calculate stability
        if len(accuracies) >= 10:
            stability = 1.0 - np.std(accuracies[-10:]) / max(0.001, np.mean(accuracies[-10:]))
        else:
            stability = 0.5

        # Calculate efficiency metric
        efficiency = (final_accuracy * stats['aggregations_performed']) / training_duration

        results[max_staleness] = {
            'final_accuracy': final_accuracy,
            'max_accuracy': max_accuracy,
            'convergence_time': convergence_time,
            'stability': max(0, stability),
            'efficiency': efficiency,
            'rejection_rate': stats['rejected_updates'] / max(1, stats['total_updates']),
            'aggregations': stats['aggregations_performed'],
            'avg_staleness': stats['average_staleness'],
            'high_quality_ratio': stats.get('high_quality_ratio', 0),
            'accuracies': accuracies,
            'timestamps': timestamps
        }

        print(f"  Final accuracy: {final_accuracy:.4f}")
        print(f"  Rejection rate: {results[max_staleness]['rejection_rate']:.3f}")
        print(f"  Average actual staleness: {stats['average_staleness']:.2f}")
        print(f"  Aggregation count: {stats['aggregations_performed']}")
        print(f"  Efficiency metric: {efficiency:.4f}")

    # Create staleness analysis plots
    create_advanced_staleness_plots(results)

    # Find optimal parameters
    find_optimal_staleness(results)

    return results


def advanced_adaptive_weighting_comparison():
    """Advanced adaptive weighting comparison analysis"""
    print("\n" + "=" * 70)
    print("=== Advanced Adaptive Weighting Comparison Analysis ===")
    print("=" * 70 + "\n")

    # Test different weighting strategies
    weighting_strategies = {
        'no_weighting': {
            'adaptive_weighting': False,
            'staleness_penalty': 'linear',
            'description': 'No Adaptive Weighting'
        },
        'basic_weighting': {
            'adaptive_weighting': True,
            'staleness_penalty': 'linear',
            'description': 'Basic Adaptive Weighting'
        },
        'advanced_weighting': {
            'adaptive_weighting': True,
            'staleness_penalty': 'adaptive',
            'description': 'Advanced Adaptive Weighting'
        }
    }

    # Test parameters
    num_clients = 12
    samples_per_client = 100
    input_dim = 24
    num_classes = 5
    training_duration = 50

    print(f"Test environment:")
    print(f"- Number of clients: {num_clients}")
    print(f"- Highly heterogeneous environment (heterogeneity=0.8)")
    print(f"- Including low-quality clients (30%)")
    print(f"- Training duration: {training_duration} seconds\n")

    # Generate heterogeneous data (high heterogeneity)
    client_datasets, global_test_dataset = create_high_quality_federated_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        input_dim=input_dim,
        num_classes=num_classes,
        heterogeneity=0.8,  # High heterogeneity
        random_state=123
    )

    # Define client quality - fixed key part
    client_quality_map = {
        'high': 0.9,  # High quality clients
        'medium': 0.6,  # Medium quality clients
        'low': 0.3  # Low quality clients
    }

    # Fix: Ensure correct client quality list length
    high_count = int(num_clients * 0.3)
    medium_count = int(num_clients * 0.4)
    low_count = num_clients - high_count - medium_count  # Ensure correct total

    client_qualities = (
            ['high'] * high_count +
            ['medium'] * medium_count +
            ['low'] * low_count
    )

    # Verify length
    assert len(client_qualities) == num_clients, f"Client quality list length error: {len(client_qualities)} != {num_clients}"

    np.random.shuffle(client_qualities)

    # Print allocation
    print(f"Client quality allocation:")
    print(f"- High quality: {high_count} clients")
    print(f"- Medium quality: {medium_count} clients")
    print(f"- Low quality: {low_count} clients")
    print(f"- Total: {len(client_qualities)} clients\n")

    results = {}

    for strategy_name, config in weighting_strategies.items():
        print(f"\nTesting strategy: {config['description']}")

        # Initialize protocol
        async_protocol = SuperiorAsyncFedProtocol(
            max_staleness=20.0,  # Use better staleness value
            min_buffer_size=2,
            max_buffer_size=5,
            adaptive_weighting=config['adaptive_weighting'],
            staleness_penalty=config['staleness_penalty'],
            momentum=0.9
        )

        # Initialize model
        base_model = RobustNeuralNet(input_dim, 96, num_classes)
        async_protocol.global_model = base_model.state_dict()

        # Performance recording
        accuracies = []
        timestamps = []
        quality_metrics = []

        start_time = time.time()
        eval_lock = threading.Lock()
        client_versions = [0] * num_clients  # Add client version tracking

        # Evaluation thread
        def continuous_evaluation():
            eval_count = 0
            while time.time() - start_time < training_duration:
                current_time = time.time() - start_time
                if current_time > eval_count * 2.5:  # Evaluate every 2.5 seconds
                    global_state = async_protocol.get_global_model()
                    if global_state:
                        with eval_lock:
                            eval_model = RobustNeuralNet(input_dim, 96, num_classes)
                            eval_model.load_state_dict(global_state)
                            acc, loss = accurate_model_evaluation(eval_model, global_test_dataset)

                            accuracies.append(acc)
                            timestamps.append(current_time)

                            # Record quality metrics
                            stats = async_protocol.get_stats()
                            quality_metrics.append({
                                'high_quality_ratio': stats.get('high_quality_ratio', 0),
                                'network_health': stats.get('network_health', 0),
                                'aggregations': stats['aggregations_performed']
                            })

                            eval_count += 1

                time.sleep(1.0)

        eval_thread = threading.Thread(target=continuous_evaluation)
        eval_thread.start()

        # Client training function
        def quality_aware_client_training(client_id):
            # Add boundary check
            if client_id >= len(client_qualities):
                print(f"Warning: Client ID {client_id} out of range")
                return

            client_quality = client_quality_map[client_qualities[client_id]]
            noise_level = 1.0 - client_quality  # Lower quality means higher noise

            while time.time() - start_time < training_duration:
                # Low quality clients more likely to disconnect
                if np.random.random() < noise_level * 0.3:
                    time.sleep(np.random.uniform(2, 5))
                    continue

                # Get global model
                global_state = async_protocol.get_global_model()
                if global_state is None:
                    time.sleep(0.5)
                    continue

                # Local training
                local_model = RobustNeuralNet(input_dim, 96, num_classes)
                local_model.load_state_dict(global_state)

                # Quality-related training parameters
                epochs = max(1, int(3 * client_quality))
                lr = 0.02 * client_quality

                updated_state, local_loss, data_size = advanced_client_training(
                    local_model, client_datasets[client_id],
                    epochs=epochs, lr=lr
                )

                # Add noise for low quality clients
                if client_quality < 0.5:
                    for name, param in updated_state.items():
                        noise = torch.randn_like(param) * noise_level * 0.05
                        updated_state[name] = param + noise

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

                # Client-specific delay
                delay = np.random.uniform(0.5, 2.0) * (2.0 - client_quality)
                time.sleep(delay * 0.1)

        # Start all clients
        client_threads = []
        for client_id in range(num_clients):
            thread = threading.Thread(target=quality_aware_client_training, args=(client_id,))
            thread.start()
            client_threads.append(thread)

        # Wait for training completion
        time.sleep(training_duration)

        # Stop threads
        eval_thread.join()
        for thread in client_threads:
            thread.join(timeout=2.0)

        # Collect final results
        final_stats = async_protocol.get_stats()
        async_protocol.shutdown()

        results[strategy_name] = {
            'accuracies': accuracies,
            'timestamps': timestamps,
            'final_accuracy': accuracies[-1] if accuracies else 0.0,
            'max_accuracy': max(accuracies) if accuracies else 0.0,
            'quality_metrics': quality_metrics,
            'stats': final_stats,
            'config': config
        }

        print(f"  Final accuracy: {results[strategy_name]['final_accuracy']:.4f}")
        print(f"  Max accuracy: {results[strategy_name]['max_accuracy']:.4f}")
        print(f"  High quality update ratio: {final_stats.get('high_quality_ratio', 0):.3f}")
        print(f"  Aggregation count: {final_stats['aggregations_performed']}")

    # Create adaptive weighting analysis plots
    create_adaptive_weighting_analysis_plots(results)

    return results


def intelligent_buffer_size_optimization():
    """Intelligent buffer size optimization analysis"""
    print("\n" + "=" * 70)
    print("=== Intelligent Buffer Size Optimization Analysis ===")
    print("=" * 70 + "\n")

    # Test different buffer configurations
    buffer_configs = [
        {'min': 1, 'max': 3, 'name': 'Small Buffer(1-3)'},
        {'min': 2, 'max': 5, 'name': 'Medium Buffer(2-5)'},
        {'min': 3, 'max': 8, 'name': 'Large Buffer(3-8)'},
        {'min': 5, 'max': 12, 'name': 'Extra Large Buffer(5-12)'}
    ]

    # Test parameters
    num_clients = 10
    samples_per_client = 90
    input_dim = 20
    num_classes = 4
    training_duration = 35

    print(f"Test configuration:")
    print(f"- Buffer configurations: {[c['name'] for c in buffer_configs]}")
    print(f"- Simulate different network conditions")
    print(f"- Focus on: Latency vs Throughput tradeoff\n")

    # Generate data
    client_datasets, global_test_dataset = create_high_quality_federated_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        input_dim=input_dim,
        num_classes=num_classes,
        heterogeneity=0.5
    )

    results = {}

    for config in buffer_configs:
        print(f"\nTesting {config['name']}")

        # Initialize protocol
        async_protocol = SuperiorAsyncFedProtocol(
            max_staleness=15.0,
            min_buffer_size=config['min'],
            max_buffer_size=config['max'],
            adaptive_weighting=True,
            momentum=0.9
        )

        base_model = RobustNeuralNet(input_dim, 64, num_classes)
        async_protocol.global_model = base_model.state_dict()

        # Record detailed metrics
        performance_log = {
            'accuracies': [],
            'timestamps': [],
            'buffer_sizes': [],
            'aggregation_delays': [],
            'throughput_points': []
        }

        start_time = time.time()
        last_aggregation_count = 0

        # Enhanced monitoring thread
        def enhanced_monitoring():
            nonlocal last_aggregation_count
            eval_count = 0

            while time.time() - start_time < training_duration:
                current_time = time.time() - start_time
                if current_time > eval_count * 2.0:

                    # Performance evaluation
                    global_state = async_protocol.get_global_model()
                    if global_state:
                        eval_model = RobustNeuralNet(input_dim, 64, num_classes)
                        eval_model.load_state_dict(global_state)
                        acc, _ = accurate_model_evaluation(eval_model, global_test_dataset)

                        performance_log['accuracies'].append(acc)
                        performance_log['timestamps'].append(current_time)

                    # Buffer and throughput monitoring
                    stats = async_protocol.get_stats()
                    current_buffer_size = stats.get('current_buffer_size', 0)
                    current_aggregations = stats['aggregations_performed']

                    performance_log['buffer_sizes'].append(current_buffer_size)

                    # Calculate throughput
                    if eval_count > 0:
                        throughput = (current_aggregations - last_aggregation_count) / 2.0
                        performance_log['throughput_points'].append(throughput)

                    last_aggregation_count = current_aggregations
                    eval_count += 1

                time.sleep(1.0)

        monitor_thread = threading.Thread(target=enhanced_monitoring)
        monitor_thread.start()

        # Simplified client simulation
        def buffer_test_client(client_id):
            update_interval = np.random.uniform(1.0, 3.0)  # Different frequencies for different clients

            while time.time() - start_time < training_duration:
                # Get global model and train
                global_state = async_protocol.get_global_model()
                if global_state and len(client_datasets[client_id]) > 0:
                    local_model = RobustNeuralNet(input_dim, 64, num_classes)
                    local_model.load_state_dict(global_state)

                    updated_state, local_loss, data_size = advanced_client_training(
                        local_model, client_datasets[client_id], epochs=2, lr=0.02
                    )

                    # Calculate updates
                    update_dict = {}
                    for name, param in updated_state.items():
                        update_dict[name] = param - global_state[name]

                    # Record submission time (for latency analysis)
                    submit_time = time.time()
                    accepted, _ = async_protocol.submit_update(
                        f"client_{client_id}", update_dict, 0, local_loss, data_size
                    )

                    if accepted:
                        delay = time.time() - submit_time
                        performance_log['aggregation_delays'].append(delay)

                time.sleep(update_interval)

        # Start clients
        client_threads = []
        for client_id in range(num_clients):
            thread = threading.Thread(target=buffer_test_client, args=(client_id,))
            thread.start()
            client_threads.append(thread)

        # Wait for completion
        time.sleep(training_duration)
        monitor_thread.join()
        for thread in client_threads:
            thread.join(timeout=2.0)

        # Analyze results
        final_stats = async_protocol.get_stats()
        async_protocol.shutdown()

        # Calculate key metrics
        final_accuracy = performance_log['accuracies'][-1] if performance_log['accuracies'] else 0.0
        avg_delay = np.mean(performance_log['aggregation_delays']) if performance_log['aggregation_delays'] else 0.0
        avg_throughput = np.mean(performance_log['throughput_points']) if performance_log['throughput_points'] else 0.0
        avg_buffer_usage = np.mean(performance_log['buffer_sizes']) if performance_log['buffer_sizes'] else 0.0

        # Efficiency score = accuracy * throughput / latency
        efficiency_score = (final_accuracy * avg_throughput) / max(0.001, avg_delay)

        results[config['name']] = {
            'config': config,
            'final_accuracy': final_accuracy,
            'avg_delay': avg_delay,
            'avg_throughput': avg_throughput,
            'avg_buffer_usage': avg_buffer_usage,
            'efficiency_score': efficiency_score,
            'performance_log': performance_log,
            'stats': final_stats
        }

        print(f"  Final accuracy: {final_accuracy:.4f}")
        print(f"  Average delay: {avg_delay:.3f}s")
        print(f"  Average throughput: {avg_throughput:.2f} aggregations/sec")
        print(f"  Efficiency score: {efficiency_score:.4f}")

    # Create buffer analysis plots
    create_buffer_optimization_plots(results)

    return results


def create_advanced_staleness_plots(results):
    """Create advanced staleness analysis charts"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Staleness Parameter Analysis', fontsize=16, fontweight='bold')

    staleness_vals = list(results.keys())

    # 1. Accuracy vs staleness
    ax1 = axes[0, 0]
    final_accs = [results[s]['final_accuracy'] for s in staleness_vals]
    max_accs = [results[s]['max_accuracy'] for s in staleness_vals]

    ax1.plot(staleness_vals, final_accs, 'b-o', linewidth=2, markersize=8, label='Final Accuracy')
    ax1.plot(staleness_vals, max_accs, 'g--s', linewidth=2, markersize=6, label='Max Accuracy')
    ax1.set_xlabel('Max Staleness')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Staleness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Rejection rate vs staleness
    ax2 = axes[0, 1]
    rejection_rates = [results[s]['rejection_rate'] for s in staleness_vals]
    ax2.plot(staleness_vals, rejection_rates, 'r-^', linewidth=2, markersize=8)
    ax2.set_xlabel('Max Staleness')
    ax2.set_ylabel('Update Rejection Rate')
    ax2.set_title('Rejection Rate vs Staleness')
    ax2.grid(True, alpha=0.3)

    # 3. Efficiency metric
    ax3 = axes[0, 2]
    efficiencies = [results[s]['efficiency'] for s in staleness_vals]
    ax3.plot(staleness_vals, efficiencies, 'purple', linewidth=3, marker='D', markersize=10)
    ax3.set_xlabel('Max Staleness')
    ax3.set_ylabel('Efficiency Metric')
    ax3.set_title('Overall Efficiency vs Staleness')
    ax3.grid(True, alpha=0.3)

    # Mark optimal point
    best_idx = np.argmax(efficiencies)
    best_staleness = staleness_vals[best_idx]
    ax3.axvline(x=best_staleness, color='red', linestyle='--', alpha=0.7)
    ax3.text(best_staleness, max(efficiencies) * 0.9, f'Optimal: {best_staleness}',
             ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 4. Convergence time
    ax4 = axes[1, 0]
    conv_times = [results[s]['convergence_time'] for s in staleness_vals]
    ax4.bar(range(len(staleness_vals)), conv_times, color='orange', alpha=0.7)
    ax4.set_xlabel('Max Staleness')
    ax4.set_ylabel('Convergence Time (seconds)')
    ax4.set_title('Convergence Speed vs Staleness')
    ax4.set_xticks(range(len(staleness_vals)))
    ax4.set_xticklabels(staleness_vals)

    # 5. Stability analysis
    ax5 = axes[1, 1]
    stabilities = [results[s]['stability'] for s in staleness_vals]
    ax5.plot(staleness_vals, stabilities, 'brown', linewidth=2, marker='*', markersize=10)
    ax5.set_xlabel('Max Staleness')
    ax5.set_ylabel('Model Stability')
    ax5.set_title('Stability vs Staleness')
    ax5.grid(True, alpha=0.3)

    # 6. Actual vs theoretical staleness
    ax6 = axes[1, 2]
    actual_staleness = [results[s]['avg_staleness'] for s in staleness_vals]
    ax6.plot(staleness_vals, actual_staleness, 'teal', linewidth=2, marker='h', markersize=8, label='Actual Staleness')
    ax6.plot(staleness_vals, staleness_vals, 'k--', alpha=0.5, label='Theoretical Maximum')
    ax6.set_xlabel('Set Max Staleness')
    ax6.set_ylabel('Actual Average Staleness')
    ax6.set_title('Actual vs Theoretical Staleness')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('advanced_staleness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAdvanced staleness analysis chart saved as 'advanced_staleness_analysis.png'")


def create_adaptive_weighting_analysis_plots(results):
    """Create adaptive weighting analysis charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Adaptive Weighting Strategy Deep Comparison', fontsize=14, fontweight='bold')

    # 1. Convergence curve comparison
    ax1 = axes[0, 0]
    colors = ['red', 'orange', 'blue']
    for i, (strategy, data) in enumerate(results.items()):
        if data['accuracies'] and data['timestamps']:
            ax1.plot(data['timestamps'], data['accuracies'],
                     color=colors[i], linewidth=2, label=data['config']['description'])

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Convergence Curve Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final performance comparison
    ax2 = axes[0, 1]
    strategies = list(results.keys())
    final_accs = [results[s]['final_accuracy'] for s in strategies]
    max_accs = [results[s]['max_accuracy'] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, final_accs, width, label='Final Accuracy', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x + width / 2, max_accs, width, label='Max Accuracy', color='lightcoral', alpha=0.8)

    ax2.set_xlabel('Weighting Strategy')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Final Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([results[s]['config']['description'] for s in strategies], rotation=15)
    ax2.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Quality metric changes
    ax3 = axes[1, 0]
    for strategy, data in results.items():
        if data['quality_metrics']:
            quality_ratios = [qm['high_quality_ratio'] for qm in data['quality_metrics']]
            time_points = range(len(quality_ratios))
            ax3.plot(time_points, quality_ratios, linewidth=2,
                     label=data['config']['description'], marker='o', markersize=4)

    ax3.set_xlabel('Evaluation Round')
    ax3.set_ylabel('High Quality Update Ratio')
    ax3.set_title('Update Quality Trend Changes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Aggregation efficiency comparison
    ax4 = axes[1, 1]
    aggregation_counts = [results[s]['stats']['aggregations_performed'] for s in strategies]
    high_quality_ratios = [results[s]['stats'].get('high_quality_ratio', 0) for s in strategies]

    # Create compound bar chart
    x = np.arange(len(strategies))

    # Aggregation count (left axis)
    color1 = 'tab:blue'
    ax4.set_xlabel('Weighting Strategy')
    ax4.set_ylabel('Aggregation Count', color=color1)
    bars1 = ax4.bar(x, aggregation_counts, color=color1, alpha=0.7, label='Aggregation Count')
    ax4.tick_params(axis='y', labelcolor=color1)

    # High quality ratio (right axis)
    ax4_twin = ax4.twinx()
    color2 = 'tab:orange'
    ax4_twin.set_ylabel('High Quality Update Ratio', color=color2)
    line1 = ax4_twin.plot(x, high_quality_ratios, color=color2, linewidth=3,
                          marker='D', markersize=8, label='Quality Ratio')
    ax4_twin.tick_params(axis='y', labelcolor=color2)

    ax4.set_title('Aggregation Efficiency vs Update Quality')
    ax4.set_xticks(x)
    ax4.set_xticklabels([results[s]['config']['description'] for s in strategies], rotation=15)

    # Merge legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('adaptive_weighting_deep_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAdaptive weighting deep analysis chart saved as 'adaptive_weighting_deep_analysis.png'")


def create_buffer_optimization_plots(results):
    """Create buffer optimization analysis charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Buffer Size Optimization Analysis', fontsize=14, fontweight='bold')

    configs = list(results.keys())

    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    accuracies = [results[c]['final_accuracy'] for c in configs]
    bars = ax1.bar(range(len(configs)), accuracies, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Buffer Configuration')
    ax1.set_ylabel('Final Accuracy')
    ax1.set_title('Accuracy vs Buffer Size')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')

    # 2. Latency vs throughput
    ax2 = axes[0, 1]
    delays = [results[c]['avg_delay'] for c in configs]
    throughputs = [results[c]['avg_throughput'] for c in configs]

    scatter = ax2.scatter(delays, throughputs, s=100, alpha=0.7, c=range(len(configs)), cmap='viridis')
    ax2.set_xlabel('Average Delay (seconds)')
    ax2.set_ylabel('Average Throughput (aggregations/sec)')
    ax2.set_title('Latency vs Throughput Tradeoff')

    # Add configuration labels
    for i, config in enumerate(configs):
        ax2.annotate(config, (delays[i], throughputs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # 3. Efficiency score comparison
    ax3 = axes[1, 0]
    efficiency_scores = [results[c]['efficiency_score'] for c in configs]
    bars = ax3.bar(range(len(configs)), efficiency_scores, color='orange', alpha=0.7)
    ax3.set_xlabel('Buffer Configuration')
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('Overall Efficiency Score')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=45)

    # Add value labels
    for bar, score in zip(bars, efficiency_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

    # 4. Buffer usage rate
    ax4 = axes[1, 1]
    buffer_usages = [results[c]['avg_buffer_usage'] for c in configs]
    bars = ax4.bar(range(len(configs)), buffer_usages, color='green', alpha=0.7)
    ax4.set_xlabel('Buffer Configuration')
    ax4.set_ylabel('Average Buffer Usage Rate')
    ax4.set_title('Buffer Usage Efficiency')
    ax4.set_xticks(range(len(configs)))
    ax4.set_xticklabels(configs, rotation=45)

    # Add value labels
    for bar, usage in zip(bars, buffer_usages):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{usage:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('buffer_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nBuffer optimization analysis chart saved as 'buffer_optimization_analysis.png'")


def find_optimal_staleness(results):
    """Find optimal staleness parameter"""
    print(f"\nStaleness Parameter Optimization Recommendations:")

    # Find optimal values based on different metrics
    best_accuracy = max(results.keys(), key=lambda x: results[x]['final_accuracy'])
    best_efficiency = max(results.keys(), key=lambda x: results[x]['efficiency'])
    best_stability = max(results.keys(), key=lambda x: results[x]['stability'])

    print(f"  • Highest accuracy: max_staleness = {best_accuracy} (accuracy: {results[best_accuracy]['final_accuracy']:.4f})")
    print(f"  • Highest efficiency: max_staleness = {best_efficiency} (efficiency: {results[best_efficiency]['efficiency']:.4f})")
    print(f"  • Highest stability: max_staleness = {best_stability} (stability: {results[best_stability]['stability']:.4f})")

    # Composite score
    def composite_score(staleness):
        data = results[staleness]
        return (0.4 * data['final_accuracy'] + 0.3 * data['efficiency'] + 0.3 * data['stability'])

    best_overall = max(results.keys(), key=composite_score)
    print(f"  • Overall optimal: max_staleness = {best_overall} (composite score: {composite_score(best_overall):.4f})")

    return best_overall


if __name__ == "__main__":
    print("Starting parameter optimization analysis...")

    # 1. Staleness analysis
    staleness_results = comprehensive_staleness_analysis()

    # 2. Adaptive weighting analysis
    weighting_results = advanced_adaptive_weighting_comparison()

    # 3. Buffer analysis
    buffer_results = intelligent_buffer_size_optimization()

    print("\n" + "="*70)
    print("Parameter optimization analysis completed!")
    print("Generated files:")
    print("  - advanced_staleness_analysis.png: Staleness parameter analysis")
    print("  - adaptive_weighting_deep_analysis.png: Adaptive weighting analysis")
    print("  - buffer_optimization_analysis.png: Buffer optimization analysis")
    print("="*70)