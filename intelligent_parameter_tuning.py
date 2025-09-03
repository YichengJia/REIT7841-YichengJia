"""
intelligent_parameter_tuning.py
Intelligent parameter tuning for federated learning protocols
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
import json

matplotlib.use('Agg')

from federated_protocol_framework import create_protocol
from unified_protocol_comparison import (
    generate_federated_data, ProtocolTester, SimpleNN
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

    def evaluate_configuration(self, config: Dict) -> Dict:
        """Evaluate a single configuration"""
        # Create protocol
        protocol = create_protocol(
            self.protocol_name,
            num_clients=self.experiment_config['num_clients'],
            **config
        )

        # Create tester
        tester = ProtocolTester(
            protocol=protocol,
            client_datasets=self.client_datasets,
            test_dataset=self.test_dataset,
            model_config=self.model_config
        )

        # Run shorter experiment for tuning
        results = tester.run_experiment(
            duration=self.experiment_config.get('duration', 30),
            eval_interval=2.0
        )

        return results

    def grid_search(self, num_samples: int = None) -> Dict:
        """Perform grid search over parameter ranges"""
        print(f"\nStarting Grid Search for {self.protocol_name}")
        print("=" * 60)

        # Create parameter grid
        param_names = list(self.param_ranges.keys())
        param_values = [self.param_ranges[name] for name in param_names]

        # Generate all combinations or sample
        all_combinations = list(product(*param_values))

        if num_samples and num_samples < len(all_combinations):
            # Random sampling for large search spaces
            np.random.shuffle(all_combinations)
            combinations = all_combinations[:num_samples]
            print(f"Sampling {num_samples} configurations from {len(all_combinations)} total")
        else:
            combinations = all_combinations
            print(f"Testing {len(combinations)} configurations")

        best_score = -float('inf')
        best_config = None

        for i, combo in enumerate(combinations):
            # Create configuration
            config = self.base_config.copy()
            for name, value in zip(param_names, combo):
                config[name] = value

            print(f"\n[{i + 1}/{len(combinations)}] Testing configuration:")
            for name, value in zip(param_names, combo):
                print(f"  {name}: {value}")

            # Evaluate
            try:
                results = self.evaluate_configuration(config)

                # Calculate composite score
                score = self.calculate_score(results)

                self.results.append({
                    'config': config,
                    'score': score,
                    'metrics': results
                })

                print(f"  Score: {score:.4f}")
                print(f"  Accuracy: {results['final_accuracy']:.4f}")
                print(f"  Communication: {results['total_data_transmitted_mb']:.2f} MB")

                if score > best_score:
                    best_score = score
                    best_config = config.copy()
                    print("  *** New best configuration! ***")

            except Exception as e:
                print(f"  Failed: {e}")
                continue

        return {'best_config': best_config, 'best_score': best_score, 'all_results': self.results}

    def bayesian_optimization(self, n_iterations: int = 20) -> Dict:
        """Bayesian optimization for continuous parameters"""
        print(f"\nStarting Bayesian Optimization for {self.protocol_name}")
        print("=" * 60)

        # Simple Bayesian optimization using Gaussian Process surrogate
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        # Initialize with random samples
        n_init = min(5, n_iterations // 2)

        # Convert parameter ranges to continuous space
        param_names = list(self.param_ranges.keys())
        bounds = []
        for name in param_names:
            values = self.param_ranges[name]
            if isinstance(values[0], (int, float)):
                bounds.append((min(values), max(values)))
            else:
                # For non-numeric, use indices
                bounds.append((0, len(values) - 1))

        # Initial random sampling
        X_samples = []
        y_samples = []

        for i in range(n_init):
            # Random configuration
            x = []
            config = self.base_config.copy()

            for j, name in enumerate(param_names):
                if isinstance(self.param_ranges[name][0], (int, float)):
                    value = np.random.uniform(bounds[j][0], bounds[j][1])
                    if isinstance(self.param_ranges[name][0], int):
                        value = int(value)
                else:
                    idx = np.random.randint(0, len(self.param_ranges[name]))
                    value = self.param_ranges[name][idx]
                    x.append(idx)

                config[name] = value
                if isinstance(value, (int, float)):
                    x.append(value)

            X_samples.append(x)

            print(f"\nInitial sample {i + 1}/{n_init}")
            results = self.evaluate_configuration(config)
            score = self.calculate_score(results)
            y_samples.append(score)

            print(f"  Score: {score:.4f}")

        # Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        # Optimization loop
        best_score = max(y_samples)
        best_idx = np.argmax(y_samples)
        best_x = X_samples[best_idx]

        for i in range(n_iterations - n_init):
            # Fit GP
            gp.fit(X_samples, y_samples)

            # Acquisition function (Upper Confidence Bound)
            def acquisition(x):
                mu, sigma = gp.predict([x], return_std=True)
                return mu[0] + 2.0 * sigma[0]  # UCB with kappa=2

            # Find next point to evaluate
            # Simple random search for acquisition maximum
            best_acq = -float('inf')
            best_next_x = None

            for _ in range(100):
                next_x = []
                for j in range(len(bounds)):
                    if isinstance(bounds[j][0], (int, float)):
                        val = np.random.uniform(bounds[j][0], bounds[j][1])
                    else:
                        val = np.random.randint(bounds[j][0], bounds[j][1] + 1)
                    next_x.append(val)

                acq_value = acquisition(next_x)
                if acq_value > best_acq:
                    best_acq = acq_value
                    best_next_x = next_x

            # Evaluate
            config = self.base_config.copy()
            for j, name in enumerate(param_names):
                if isinstance(self.param_ranges[name][0], (int, float)):
                    value = best_next_x[j]
                    if isinstance(self.param_ranges[name][0], int):
                        value = int(value)
                else:
                    value = self.param_ranges[name][int(best_next_x[j])]
                config[name] = value

            print(f"\nBayesian iteration {i + 1}/{n_iterations - n_init}")
            results = self.evaluate_configuration(config)
            score = self.calculate_score(results)

            X_samples.append(best_next_x)
            y_samples.append(score)

            print(f"  Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_x = best_next_x
                print("  *** New best configuration! ***")

        # Convert best_x back to configuration
        best_config = self.base_config.copy()
        for j, name in enumerate(param_names):
            if isinstance(self.param_ranges[name][0], (int, float)):
                value = best_x[j]
                if isinstance(self.param_ranges[name][0], int):
                    value = int(value)
            else:
                value = self.param_ranges[name][int(best_x[j])]
            best_config[name] = value

        return {'best_config': best_config, 'best_score': best_score}

    def calculate_score(self, metrics: Dict) -> float:
        """Calculate composite score for configuration"""
        # Normalize metrics
        accuracy = metrics['final_accuracy']

        # Communication efficiency (inverse of data transmitted)
        comm_efficiency = 1.0 / (1.0 + metrics['total_data_transmitted_mb'])

        # Convergence speed
        if metrics['convergence_time'] < float('inf'):
            conv_speed = 1.0 / (1.0 + metrics['convergence_time'])
        else:
            conv_speed = 0.1

        # Throughput
        throughput_norm = min(1.0, metrics['throughput_updates_per_second'] / 10.0)

        # Weighted combination (adjustable based on priorities)
        score = (0.7 * accuracy +
                 0.15 * comm_efficiency +
                 0.05 * conv_speed +
                 0.1 * throughput_norm)

        return score


def tune_improved_async_protocol():
    """Tune parameters for improved async protocol"""
    print("\n" + "=" * 70)
    print("PARAMETER TUNING FOR IMPROVED ASYNC PROTOCOL")
    print("=" * 70)

    base_config = {
        'adaptive_weighting': True,
        'momentum': 0.9,
    }

    # Define parameter ranges to search
    param_ranges = {
        'max_staleness': [10, 15, 20, 25, 30],
        'min_buffer_size': [2, 3, 4, 5],
        'max_buffer_size': [5, 6, 8, 10],
        'compression_ratio': [0.3, 0.4, 0.5, 0.6, 0.7],
    }

    experiment_config = {
        'num_clients': 8,
        'samples_per_client': 80,
        'input_dim': 16,
        'hidden_dim': 48,
        'num_classes': 4,
        'heterogeneity': 0.6,
        'duration': 90,
    }

    tuner = ParameterTuner(
        protocol_name='improved_async',
        base_config=base_config,
        param_ranges=param_ranges,
        experiment_config=experiment_config
    )

    # Perform grid search
    results = tuner.grid_search(num_samples=20)  # Sample 20 configurations

    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)
    print(f"\nBest Configuration Found:")
    for key, value in results['best_config'].items():
        print(f"  {key}: {value}")
    print(f"\nBest Score: {results['best_score']:.4f}")

    # Save results
    with open('tuning_results.json', 'w') as f:
        json.dump({
            'best_config': results['best_config'],
            'best_score': float(results['best_score']),
        }, f, indent=2)

    print("\nResults saved to 'tuning_results.json'")

    return results


def analyze_parameter_sensitivity():
    """Analyze sensitivity of parameters"""
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    base_config = {
        'max_staleness': 20,
        'min_buffer_size': 3,
        'max_buffer_size': 8,
        'adaptive_weighting': True,
        'momentum': 0.9,
        'compression_ratio': 0.5,
    }

    parameters_to_test = {
        'max_staleness': [10, 15, 20, 25, 30, 40],
        'compression_ratio': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'min_buffer_size': [1, 2, 3, 4, 5, 6],
    }

    experiment_config = {
        'num_clients': 6,
        'samples_per_client': 60,
        'input_dim': 12,
        'hidden_dim': 32,
        'num_classes': 3,
        'heterogeneity': 0.5,
        'duration': 20,
    }

    results = {}

    for param_name, param_values in parameters_to_test.items():
        print(f"\nTesting sensitivity of {param_name}")
        print("-" * 40)

        param_results = []

        for value in param_values:
            config = base_config.copy()
            config[param_name] = value

            print(f"  {param_name} = {value}")

            # Create and test protocol
            tuner = ParameterTuner(
                protocol_name='improved_async',
                base_config={},
                param_ranges={},
                experiment_config=experiment_config
            )

            try:
                metrics = tuner.evaluate_configuration(config)
                score = tuner.calculate_score(metrics)

                param_results.append({
                    'value': value,
                    'score': score,
                    'accuracy': metrics['final_accuracy'],
                    'communication': metrics['total_data_transmitted_mb']
                })

                print(f"    Score: {score:.4f}, Acc: {metrics['final_accuracy']:.4f}")
            except:
                print(f"    Failed")
                continue

        results[param_name] = param_results

    # Create sensitivity plots
    create_sensitivity_plots(results)

    return results


def create_sensitivity_plots(results: Dict):
    """Create parameter sensitivity visualization"""
    n_params = len(results)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    for idx, (param_name, param_results) in enumerate(results.items()):
        if not param_results:
            continue

        values = [r['value'] for r in param_results]
        scores = [r['score'] for r in param_results]
        accuracies = [r['accuracy'] for r in param_results]

        ax = axes[idx]

        # Plot score and accuracy
        ax2 = ax.twinx()

        line1 = ax.plot(values, scores, 'b-o', label='Score', linewidth=2)
        line2 = ax2.plot(values, accuracies, 'r--s', label='Accuracy', linewidth=2)

        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Score', color='b')
        ax2.set_ylabel('Accuracy', color='r')

        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')

        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sensitivity: {param_name}')

    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSensitivity plots saved as 'parameter_sensitivity.png'")


def main():
    """Main tuning and analysis"""

    # 1. Tune improved async protocol
    tuning_results = tune_improved_async_protocol()

    # 2. Analyze parameter sensitivity
    sensitivity_results = analyze_parameter_sensitivity()

    print("\n" + "=" * 70)
    print("PARAMETER TUNING COMPLETE")
    print("=" * 70)
    print("\nOptimized configuration saved to 'tuning_results.json'")
    print("Sensitivity analysis saved to 'parameter_sensitivity.png'")


if __name__ == "__main__":
    main()