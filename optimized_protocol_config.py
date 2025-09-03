"""
optimized_protocol_config.py
Better configuration for improved async protocol based on analysis
"""

OPTIMIZED_IMPROVED_ASYNC_CONFIG = {

    'max_staleness': 25,
    'min_buffer_size': 2,
    'max_buffer_size': 5,


    'compression_ratio': 0.2,

    'momentum': 0.9,
    'adaptive_weighting': True,
    'quality_threshold': 0.3,
    'learning_rate_decay': 0.98,
}


SCENARIO_CONFIGS = {

    'high_accuracy': {
        'max_staleness': 10,
        'min_buffer_size': 1,
        'max_buffer_size': 3,
        'compression_ratio': 1.0,
        'momentum': 0.95,
        'adaptive_weighting': True,
    },


    'balanced': {
        'max_staleness': 15,
        'min_buffer_size': 2,
        'max_buffer_size': 5,
        'compression_ratio': 0.75,
        'momentum': 0.85,
        'adaptive_weighting': True,
    },


    'low_communication': {
        'max_staleness': 20,
        'min_buffer_size': 3,
        'max_buffer_size': 8,
        'compression_ratio': 0.4,
        'momentum': 0.8,
        'adaptive_weighting': True,
    }
}


def get_improved_config(scenario='balanced'):
    """Get improved configuration for specific scenario"""
    if scenario in SCENARIO_CONFIGS:
        return SCENARIO_CONFIGS[scenario]
    return OPTIMIZED_IMPROVED_ASYNC_CONFIG


def print_config_comparison():
    """Print comparison of original vs optimized configs"""
    print("=" * 70)
    print("CONFIGURATION COMPARISON")
    print("=" * 70)

    original = {
        'max_staleness': 20,
        'min_buffer_size': 3,
        'max_buffer_size': 8,
        'compression_ratio': 0.5,
        'momentum': 0.9,
    }

    optimized = OPTIMIZED_IMPROVED_ASYNC_CONFIG

    print(f"\n{'Parameter':<20} {'Original':<15} {'Optimized':<15} {'Change'}")
    print("-" * 65)

    for key in original:
        orig_val = original[key]
        opt_val = optimized.get(key, orig_val)

        if isinstance(orig_val, float):
            change = f"{(opt_val - orig_val):+.2f}"
        else:
            change = f"{(opt_val - orig_val):+d}"

        print(f"{key:<20} {str(orig_val):<15} {str(opt_val):<15} {change}")


def quick_test_improved_config():
    """Quick test of the improved configuration"""
    from federated_protocol_framework import create_protocol
    from unified_protocol_comparison import (
        SimpleNN, generate_federated_data,
        train_client, evaluate_model, ClientUpdate
    )
    import torch
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("TESTING IMPROVED CONFIGURATION")
    print("=" * 70)

    # Generate test data
    client_datasets, test_dataset = generate_federated_data(
        num_clients=3,
        samples_per_client=30,
        input_dim=8,
        num_classes=2,
        heterogeneity=0.3
    )

    model_config = {
        'input_dim': 8,
        'hidden_dim': 16,
        'output_dim': 2
    }

    configs_to_test = {
        'Original': {
            'max_staleness': 20,
            'min_buffer_size': 3,
            'max_buffer_size': 8,
            'compression_ratio': 0.5,
            'momentum': 0.9,
            'adaptive_weighting': True
        },
        'Optimized': OPTIMIZED_IMPROVED_ASYNC_CONFIG
    }

    for name, config in configs_to_test.items():
        print(f"\nTesting {name} configuration...")

        protocol = create_protocol('improved_async', num_clients=3, **config)

        # Set initial model
        initial_model = SimpleNN(**model_config)
        protocol.set_global_model(initial_model.state_dict())

        # Train for a few rounds
        for round_num in range(10):
            for client_id in range(3):
                global_state = protocol.get_global_model()
                if global_state is None:
                    continue

                local_model = SimpleNN(**model_config)
                local_model.load_state_dict(global_state)

                updated_state, loss, data_size = train_client(
                    local_model, client_datasets[client_id],
                    epochs=1, lr=0.01
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

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Communication: {metrics['total_data_transmitted_mb']:.3f} MB")
            print(f"  Aggregations: {metrics['aggregations_performed']}")

        protocol.shutdown()


if __name__ == "__main__":
    print_config_comparison()
    quick_test_improved_config()