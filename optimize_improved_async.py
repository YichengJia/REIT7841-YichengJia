"""
optimize_improved_async.py
Find optimal parameters for improved async protocol
"""

import torch
import numpy as np
from federated_protocol_framework import create_protocol, ClientUpdate
from unified_protocol_comparison import SimpleNN, generate_federated_data, train_client, evaluate_model
import json

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


def test_configuration(config, client_datasets, test_dataset, model_config, num_rounds=10):
    """Test a specific configuration"""
    protocol = create_protocol('improved_async', num_clients=len(client_datasets), **config)

    # Set initial model
    initial_model = SimpleNN(**model_config)
    protocol.set_global_model(initial_model.state_dict())

    # Simulate training
    for round_num in range(num_rounds):
        for client_id in range(len(client_datasets)):
            global_state = protocol.get_global_model()
            if global_state is None:
                continue

            # Local training
            local_model = SimpleNN(**model_config)
            local_model.load_state_dict(global_state)

            updated_state, loss, data_size = train_client(
                local_model, client_datasets[client_id],
                epochs=2, lr=0.01
            )

            # Calculate update
            update_dict = {}
            for name, param in updated_state.items():
                if name in global_state and 'num_batches_tracked' not in name:
                    param_update = param.clone().float()
                    global_param = global_state[name].clone().float()
                    update_dict[name] = param_update - global_param

            # Submit update
            update = ClientUpdate(
                client_id=f"client_{client_id}",
                update_data=update_dict,
                model_version=protocol.model_version,
                local_loss=loss,
                data_size=data_size,
                timestamp=0.0
            )

            protocol.receive_update(update)

    # Final evaluation
    final_model_state = protocol.get_global_model()
    if final_model_state:
        eval_model = SimpleNN(**model_config)
        eval_model.load_state_dict(final_model_state)
        accuracy, loss = evaluate_model(eval_model, test_dataset)

        metrics = protocol.metrics.get_summary()
        protocol.shutdown()

        return {
            'accuracy': accuracy,
            'loss': loss,
            'aggregations': metrics['aggregations_performed'],
            'communication_mb': metrics['total_data_transmitted_mb']
        }

    protocol.shutdown()
    return None


def find_optimal_parameters():
    """Find optimal parameters for improved async protocol"""
    print("=" * 70)
    print("OPTIMIZING IMPROVED ASYNC PROTOCOL PARAMETERS")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    client_datasets, test_dataset = generate_federated_data(
        num_clients=5,
        samples_per_client=50,
        input_dim=10,
        num_classes=3,
        heterogeneity=0.5
    )

    model_config = {
        'input_dim': 10,
        'hidden_dim': 32,
        'output_dim': 3
    }

    # Test different configurations
    configurations = [
        # Baseline (current)
        {
            'name': 'Baseline',
            'config': {
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'compression_ratio': 0.5,
                'momentum': 0.9,
                'adaptive_weighting': True
            }
        },
        # Less compression
        {
            'name': 'Less Compression',
            'config': {
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'compression_ratio': 0.8,  # Less aggressive
                'momentum': 0.9,
                'adaptive_weighting': True
            }
        },
        # No compression
        {
            'name': 'No Compression',
            'config': {
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'compression_ratio': 1.0,  # No compression
                'momentum': 0.9,
                'adaptive_weighting': True
            }
        },
        # Larger buffer
        {
            'name': 'Larger Buffer',
            'config': {
                'max_staleness': 15,
                'min_buffer_size': 3,
                'max_buffer_size': 6,
                'compression_ratio': 0.7,
                'momentum': 0.9,
                'adaptive_weighting': True
            }
        },
        # More frequent aggregation
        {
            'name': 'Frequent Aggregation',
            'config': {
                'max_staleness': 10,
                'min_buffer_size': 1,  # Aggregate more often
                'max_buffer_size': 2,
                'compression_ratio': 0.8,
                'momentum': 0.9,
                'adaptive_weighting': True
            }
        },
        # Adjusted momentum
        {
            'name': 'Lower Momentum',
            'config': {
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 4,
                'compression_ratio': 0.7,
                'momentum': 0.7,  # Lower momentum
                'adaptive_weighting': True
            }
        },
        # Balanced
        {
            'name': 'Balanced',
            'config': {
                'max_staleness': 15,
                'min_buffer_size': 2,
                'max_buffer_size': 5,
                'compression_ratio': 0.75,
                'momentum': 0.85,
                'adaptive_weighting': True
            }
        },
        # Optimized for accuracy
        {
            'name': 'Accuracy Focus',
            'config': {
                'max_staleness': 8,
                'min_buffer_size': 1,
                'max_buffer_size': 3,
                'compression_ratio': 1.0,  # No compression for accuracy
                'momentum': 0.95,
                'adaptive_weighting': True
            }
        }
    ]

    results = []
    best_config = None
    best_score = 0

    print("\nTesting configurations...")
    print("-" * 70)

    for cfg in configurations:
        print(f"\nTesting: {cfg['name']}")
        print(f"  Config: {cfg['config']}")

        result = test_configuration(
            cfg['config'],
            client_datasets,
            test_dataset,
            model_config,
            num_rounds=15
        )

        if result:
            # Calculate composite score
            # Balance accuracy and communication efficiency
            score = result['accuracy'] * 0.7 + (1.0 / (1.0 + result['communication_mb'])) * 0.3

            results.append({
                'name': cfg['name'],
                'config': cfg['config'],
                'accuracy': result['accuracy'],
                'loss': result['loss'],
                'communication_mb': result['communication_mb'],
                'aggregations': result['aggregations'],
                'score': score
            })

            print(f"  Results:")
            print(f"    Accuracy: {result['accuracy']:.4f}")
            print(f"    Loss: {result['loss']:.4f}")
            print(f"    Communication: {result['communication_mb']:.3f} MB")
            print(f"    Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_config = cfg['config']
                print("    *** NEW BEST ***")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\nTop 3 Configurations:")
    for i, r in enumerate(results[:3], 1):
        print(f"\n{i}. {r['name']}")
        print(f"   Score: {r['score']:.4f}")
        print(f"   Accuracy: {r['accuracy']:.4f}")
        print(f"   Communication: {r['communication_mb']:.3f} MB")

    if best_config:
        print("\n" + "-" * 70)
        print("BEST CONFIGURATION:")
        print("-" * 70)
        for key, value in best_config.items():
            print(f"  {key}: {value}")

        # Save best config
        with open('optimal_improved_async_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)

        print("\n✓ Best configuration saved to 'optimal_improved_async_config.json'")

    return best_config, results


def compare_with_baseline():
    """Compare optimized config with baseline protocols"""
    print("\n" + "=" * 70)
    print("COMPARING OPTIMIZED CONFIG WITH BASELINE PROTOCOLS")
    print("=" * 70)

    # Load optimal config
    try:
        with open('optimal_improved_async_config.json', 'r') as f:
            optimal_config = json.load(f)
    except:
        print("No optimal config found. Running optimization first...")
        optimal_config, _ = find_optimal_parameters()

    # Generate test data
    client_datasets, test_dataset = generate_federated_data(
        num_clients=5,
        samples_per_client=50,
        input_dim=10,
        num_classes=3,
        heterogeneity=0.5
    )

    model_config = {
        'input_dim': 10,
        'hidden_dim': 32,
        'output_dim': 3
    }

    protocols = {
        'FedAsync': {
            'protocol': 'fedasync',
            'config': {'max_staleness': 10}
        },
        'FedBuff': {
            'protocol': 'fedbuff',
            'config': {'buffer_size': 3, 'max_staleness': 10}
        },
        'Improved (Original)': {
            'protocol': 'improved_async',
            'config': {
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'compression_ratio': 0.5,
                'momentum': 0.9
            }
        },
        'Improved (Optimized)': {
            'protocol': 'improved_async',
            'config': optimal_config
        }
    }

    print("\nRunning comparison...")
    print("-" * 70)

    results = {}

    for name, proto_info in protocols.items():
        print(f"\nTesting {name}...")

        protocol = create_protocol(
            proto_info['protocol'],
            num_clients=len(client_datasets),
            **proto_info['config']
        )

        # Set initial model
        initial_model = SimpleNN(**model_config)
        protocol.set_global_model(initial_model.state_dict())

        # Train
        for round_num in range(15):
            for client_id in range(len(client_datasets)):
                global_state = protocol.get_global_model()
                if global_state is None:
                    continue

                local_model = SimpleNN(**model_config)
                local_model.load_state_dict(global_state)

                updated_state, loss, data_size = train_client(
                    local_model, client_datasets[client_id],
                    epochs=2, lr=0.01
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
                    timestamp=0.0
                )

                protocol.receive_update(update)

        # Evaluate
        final_model_state = protocol.get_global_model()
        if final_model_state:
            eval_model = SimpleNN(**model_config)
            eval_model.load_state_dict(final_model_state)
            accuracy, loss = evaluate_model(eval_model, test_dataset)

            metrics = protocol.metrics.get_summary()

            results[name] = {
                'accuracy': accuracy,
                'loss': loss,
                'communication_mb': metrics['total_data_transmitted_mb'],
                'aggregations': metrics['aggregations_performed']
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Communication: {metrics['total_data_transmitted_mb']:.3f} MB")

        protocol.shutdown()

    # Print comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Protocol':<20} {'Accuracy':<12} {'Comm (MB)':<12} {'Efficiency':<12}")
    print("-" * 56)

    for name, metrics in results.items():
        efficiency = metrics['accuracy'] / (1.0 + metrics['communication_mb'])
        print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['communication_mb']:<12.3f} {efficiency:<12.4f}")

    # Check improvement
    if 'Improved (Optimized)' in results and 'Improved (Original)' in results:
        orig = results['Improved (Original)']['accuracy']
        opt = results['Improved (Optimized)']['accuracy']
        improvement = ((opt - orig) / orig) * 100

        print(f"\n✓ Accuracy improvement: {improvement:+.1f}%")

        if 'FedAsync' in results:
            fedasync_acc = results['FedAsync']['accuracy']
            if opt >= fedasync_acc * 0.95:
                print(f"✓ Optimized version is competitive with FedAsync!")
            else:
                gap = fedasync_acc - opt
                print(f"  Still {gap:.4f} behind FedAsync")


def main():
    """Main optimization pipeline"""
    print("\nIMPROVED ASYNC PROTOCOL OPTIMIZATION\n")

    # Step 1: Find optimal parameters
    best_config, all_results = find_optimal_parameters()

    # Step 2: Compare with baselines
    compare_with_baseline()

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nUse the configuration in 'optimal_improved_async_config.json'")
    print("for best performance in your experiments.")


if __name__ == "__main__":
    main()