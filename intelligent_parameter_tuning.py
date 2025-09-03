"""
intelligent_parameter_tuning.py
Intelligent parameter tuning for federated learning protocols
"""

import numpy as np
import torch
import time
from typing import Dict
import matplotlib


matplotlib.use('Agg')

from federated_protocol_framework import create_protocol, ClientUpdate
from unified_protocol_comparison import (
    SimpleNN, generate_federated_data,
    train_client, evaluate_model
)


# Set seeds
torch.manual_seed(42)
np.random.seed(42)


class ParameterTuner:
    """Automated parameter tuning for protocols"""

    def __init__(self, protocol_name: str, base_config: Dict,
                 param_ranges: Dict, experiment_config: Dict):
        self.protocol_name = protocol_name
        self.base_config = base_config
        self.param_ranges = param_ranges
        self.experiment_config = experiment_config
        self.results = []

        # Generate data once
        self.client_datasets, self.test_dataset = generate_federated_data(
            num_clients=experiment_config['num_clients'],
            samples_per_client=experiment_config['samples_per_client'],
            input_dim=experiment_config['input_dim'],
            num_classes=experiment_config['num_classes'],
            heterogeneity=experiment_config['heterogeneity']
        )

        self.model_config = {
            'input_dim': experiment_config['input_dim'],
            'hidden_dim': experiment_config['hidden_dim'],
            'output_dim': experiment_config['num_classes']
        }

def run_experiment_with_config(config, num_clients=10, samples_per_client=100, duration=180):
    """Run experiment for ImprovedAsyncProtocol with given config"""
    client_datasets, test_dataset = generate_federated_data(
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        input_dim=12,
        num_classes=3,
        heterogeneity=0.5
    )

    model_config = {'input_dim': 12, 'hidden_dim': 32, 'output_dim': 3}
    protocol = create_protocol('improved_async', num_clients=num_clients, **config)
    initial_model = SimpleNN(**model_config)
    protocol.set_global_model(initial_model.state_dict())

    start_time = time.time()
    while time.time() - start_time < duration:
        for client_id in range(num_clients):
            global_state = protocol.get_global_model()
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

    final_model_state = protocol.get_global_model()
    eval_model = SimpleNN(**model_config)
    eval_model.load_state_dict(final_model_state)
    accuracy, loss = evaluate_model(eval_model, test_dataset)
    metrics = protocol.metrics.get_summary()
    protocol.shutdown()
    return accuracy, loss, metrics

def tune_parameters():
    """Grid search for best parameters"""
    param_ranges = {
        'max_staleness': [10, 15, 20, 25],
        'min_buffer_size': [2, 3, 4],
        'max_buffer_size': [5, 6, 8],
        'momentum': [0.8, 0.9],
        'compression': ['topk', 'signsgd', 'qsgd'],
        'k': [50, 100],
        'num_bits': [4, 8]
    }

    best_config = None
    best_score = -float('inf')

    for staleness in param_ranges['max_staleness']:
        for min_buf in param_ranges['min_buffer_size']:
            for max_buf in param_ranges['max_buffer_size']:
                for momentum in param_ranges['momentum']:
                    for comp in param_ranges['compression']:
                        for k in param_ranges['k']:
                            for bits in param_ranges['num_bits']:
                                config = {
                                    'max_staleness': staleness,
                                    'min_buffer_size': min_buf,
                                    'max_buffer_size': max_buf,
                                    'momentum': momentum,
                                    'adaptive_weighting': True,
                                    'compression': comp,
                                    'k': k,
                                    'num_bits': bits
                                }
                                acc, loss, metrics = run_experiment_with_config(config)
                                score = acc - 0.01 * metrics['total_data_transmitted_mb']
                                if score > best_score:
                                    best_score = score
                                    best_config = config
                                print(f"Score: {score:.4f}, Acc: {acc:.4f}, Config: {config}")

    print("\nBest configuration found:")
    print(best_config)
    return best_config



def main():
    """Main tuning and analysis"""
    tune_parameters()


if __name__ == "__main__":
    main()