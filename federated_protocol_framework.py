"""
federated_protocol_framework.py
Unified framework for federated learning protocol comparison
"""

import copy
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod  # Abstract Base Classes for interface definition
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """Unified client update structure"""
    client_id: str
    update_data: Dict[str, torch.Tensor]
    model_version: int
    local_loss: float
    data_size: int
    timestamp: float
    staleness: Optional[float] = None


class ProtocolMetrics:
    """Unified metrics collector for all protocols"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            # Communication metrics
            'total_data_transmitted_mb': 0.0,
            'total_updates_sent': 0,
            'total_updates_accepted': 0,
            'total_updates_rejected': 0,

            # Performance metrics
            'aggregations_performed': 0,
            'final_accuracy': 0.0,
            'max_accuracy': 0.0,
            'convergence_time': float('inf'),

            # Efficiency metrics
            'average_round_time': 0.0,
            'idle_time_percentage': 0.0,
            'throughput_updates_per_second': 0.0,

            # Quality metrics
            'average_staleness': 0.0,
            'high_quality_updates': 0,

            # Time series data
            'accuracy_history': [],
            'loss_history': [],
            'timestamps': []
        }

    def update_communication(self, data_size_mb: float, accepted: bool = True):
        """Update communication metrics"""
        self.metrics['total_data_transmitted_mb'] += data_size_mb
        self.metrics['total_updates_sent'] += 1
        if accepted:
            self.metrics['total_updates_accepted'] += 1
        else:
            self.metrics['total_updates_rejected'] += 1

    def update_performance(self, accuracy: float, loss: float, timestamp: float):
        """Update performance metrics"""
        self.metrics['accuracy_history'].append(accuracy)
        self.metrics['loss_history'].append(loss)
        self.metrics['timestamps'].append(timestamp)

        if accuracy > self.metrics['max_accuracy']:
            self.metrics['max_accuracy'] = accuracy

        # Check convergence (accuracy stable for last 5 measurements)
        if len(self.metrics['accuracy_history']) >= 5:
            recent_std = np.std(self.metrics['accuracy_history'][-5:])
            if recent_std < 0.01 and self.metrics['convergence_time'] == float('inf'):
                self.metrics['convergence_time'] = timestamp

    def finalize(self):
        """Calculate final metrics"""
        if self.metrics['accuracy_history']:
            self.metrics['final_accuracy'] = self.metrics['accuracy_history'][-1]

        if self.metrics['timestamps'] and self.metrics['total_updates_accepted'] > 0:
            total_time = self.metrics['timestamps'][-1]
            self.metrics['throughput_updates_per_second'] = (
                self.metrics['total_updates_accepted'] / total_time
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        self.finalize()
        return self.metrics.copy()


class FederatedProtocol(ABC):
    """Base class for all federated learning protocols"""

    def __init__(self, num_clients: int, **kwargs):
        self.num_clients = num_clients
        self.global_model = None
        self.model_version = 0
        self.metrics = ProtocolMetrics()
        self.running = True
        self._lock = threading.RLock()

        # Protocol-specific parameters
        self.configure(**kwargs)

    @abstractmethod
    def configure(self, **kwargs):
        """Configure protocol-specific parameters"""
        pass

    @abstractmethod
    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        """
        Receive update from client
        Returns: (accepted, new_model_version)
        """
        pass

    @abstractmethod
    def aggregate_updates(self):
        """Perform aggregation of updates"""
        pass

    def set_global_model(self, model_state: Dict[str, torch.Tensor]):
        """Set initial global model"""
        with self._lock:
            self.global_model = copy.deepcopy(model_state)
            self.model_version = 0

    def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current global model"""
        with self._lock:
            if self.global_model is not None:
                return copy.deepcopy(self.global_model)
            return None

    def calculate_update_size(self, update_data: Dict[str, torch.Tensor]) -> float:
        """Calculate update size in MB"""
        total_bytes = 0
        for tensor in update_data.values():
            if tensor is not None:
                total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024 * 1024)

    def shutdown(self):
        """Shutdown protocol"""
        self.running = False
        logger.info(f"{self.__class__.__name__} shutdown")


class SyncFedAvg(FederatedProtocol):
    """Traditional synchronous FedAvg implementation"""

    def configure(self, **kwargs):
        self.round_participation_rate = kwargs.get('participation_rate', 0.5)
        self.max_round_time = kwargs.get('max_round_time', 30.0)
        self.current_round = 0
        self.round_buffer = []
        self.round_start_time = time.time()

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            # Check if update is for current round
            if update.model_version != self.current_round:
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data),
                    accepted=False
                )
                return False, self.current_round

            # Add to round buffer
            self.round_buffer.append(update)

            # Record metrics
            update_size = self.calculate_update_size(update.update_data)
            self.metrics.update_communication(update_size, accepted=True)

            # Check if we should aggregate
            min_clients = max(2, int(self.num_clients * self.round_participation_rate))
            if len(self.round_buffer) >= min_clients:
                self.aggregate_updates()
                return True, self.current_round + 1

            return True, self.current_round

    def aggregate_updates(self):
        """Perform FedAvg aggregation"""
        if not self.round_buffer:
            return

        # Calculate weighted average
        total_data_size = sum(u.data_size for u in self.round_buffer)
        aggregated = {}

        for update in self.round_buffer:
            weight = update.data_size / total_data_size
            for name, param in update.update_data.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(param, dtype=torch.float32)
                # Ensure both tensors are float type for aggregation
                if param.dtype != torch.float32:
                    param = param.float()
                aggregated[name] = aggregated[name] + (param * weight)

        # Apply to global model
        if self.global_model is None:
            self.global_model = {}
            for name, param in aggregated.items():
                self.global_model[name] = param.clone()
        else:
            for name, param in aggregated.items():
                if name in self.global_model:
                    # Ensure compatible types
                    if self.global_model[name].dtype != torch.float32:
                        self.global_model[name] = self.global_model[name].float()
                    self.global_model[name] = self.global_model[name] + param
                else:
                    self.global_model[name] = param.clone()

        # Update metrics and prepare for next round
        self.metrics.metrics['aggregations_performed'] += 1
        round_time = time.time() - self.round_start_time
        self.metrics.metrics['average_round_time'] = (
            0.9 * self.metrics.metrics['average_round_time'] + 0.1 * round_time
        )

        # Log before clearing buffer
        num_clients = len(self.round_buffer)

        # Prepare for next round
        self.current_round += 1
        self.model_version = self.current_round
        self.round_buffer.clear()
        self.round_start_time = time.time()

        logger.info(f"FedAvg Round {self.current_round} completed with {num_clients} clients")


class AsyncFedAvg(FederatedProtocol):
    """Basic asynchronous FedAvg (FedAsync)"""

    def configure(self, **kwargs):
        self.max_staleness = kwargs.get('max_staleness', 10)
        self.learning_rate = kwargs.get('learning_rate', 1.0)
        self.aggregation_thread = threading.Thread(target=self._continuous_aggregation, daemon=True)
        self.aggregation_thread.start()

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            current_version = self.model_version
            staleness = current_version - update.model_version

            # Reject if too stale
            if staleness > self.max_staleness:
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data),
                    accepted=False
                )
                return False, current_version

            # Accept and apply immediately
            update.staleness = staleness
            self._apply_update(update)

            update_size = self.calculate_update_size(update.update_data)
            self.metrics.update_communication(update_size, accepted=True)

            return True, current_version

    def _apply_update(self, update: ClientUpdate):
        """Apply single update to global model"""
        # Simple staleness penalty
        staleness_factor = 1.0 / (1.0 + update.staleness)
        effective_lr = self.learning_rate * staleness_factor

        # Apply update
        if self.global_model is None:
            self.global_model = {}
            for name, param in update.update_data.items():
                # Convert to float32 for computation
                if param.dtype != torch.float32:
                    param = param.float()
                self.global_model[name] = param * effective_lr
        else:
            for name, param in update.update_data.items():
                if name in self.global_model:
                    # Ensure compatible types
                    if param.dtype != torch.float32:
                        param = param.float()
                    if self.global_model[name].dtype != torch.float32:
                        self.global_model[name] = self.global_model[name].float()
                    self.global_model[name] = self.global_model[name] + (param * effective_lr)

        self.model_version += 1
        self.metrics.metrics['aggregations_performed'] += 1

        # Update staleness metrics
        self.metrics.metrics['average_staleness'] = (
            0.9 * self.metrics.metrics['average_staleness'] +
            0.1 * update.staleness
        )

    def _continuous_aggregation(self):
        """Placeholder for continuous aggregation (immediate in basic FedAsync)"""
        while self.running:
            time.sleep(1)

    def aggregate_updates(self):
        """No batch aggregation in basic FedAsync"""
        pass


class FedBuff(FederatedProtocol):
    """FedBuff - Buffered asynchronous aggregation"""

    def configure(self, **kwargs):
        self.buffer_size = kwargs.get('buffer_size', 5)
        self.max_staleness = kwargs.get('max_staleness', 15)
        self.update_buffer = deque()
        self.aggregation_thread = threading.Thread(target=self._buffer_aggregation, daemon=True)
        self.aggregation_thread.start()

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            current_version = self.model_version
            staleness = current_version - update.model_version

            if staleness > self.max_staleness:
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data),
                    accepted=False
                )
                return False, current_version

            update.staleness = staleness
            self.update_buffer.append(update)

            update_size = self.calculate_update_size(update.update_data)
            self.metrics.update_communication(update_size, accepted=True)

            return True, current_version

    def _buffer_aggregation(self):
        """Aggregate when buffer is full"""
        while self.running:
            time.sleep(0.1)

            with self._lock:
                if len(self.update_buffer) >= self.buffer_size:
                    self.aggregate_updates()

    def aggregate_updates(self):
        """Aggregate buffered updates with server learning rate and clipping"""
        if not self.update_buffer:
            return

        updates_to_aggregate = list(self.update_buffer)
        self.update_buffer.clear()

        # Weighted aggregation of DELTAS
        aggregated_delta = {}
        total_weight = 0

        for update in updates_to_aggregate:
            staleness_weight = 1.0 / (1.0 + update.staleness)
            weight = staleness_weight * update.data_size
            total_weight += weight

            for name, delta in update.update_data.items():
                if name not in aggregated_delta:
                    aggregated_delta[name] = torch.zeros_like(delta, dtype=torch.float32)
                if delta.dtype != torch.float32:
                    delta = delta.float()
                aggregated_delta[name] = aggregated_delta[name] + (delta * weight)

        # Apply aggregated delta to global model
        if total_weight > 0:
            server_lr = getattr(self, "server_lr", 0.1)

            # Normalize and clip
            for name in aggregated_delta:
                aggregated_delta[name] = aggregated_delta[name] / total_weight
                aggregated_delta[name] = torch.clamp(aggregated_delta[name], -5.0, 5.0)

            if self.global_model is None:
                self.global_model = {}
                for name in aggregated_delta:
                    self.global_model[name] = server_lr * aggregated_delta[name]
            else:
                for name in aggregated_delta:
                    if name in self.global_model:
                        self.global_model[name] = self.global_model[name] + server_lr * aggregated_delta[name]
                    else:
                        self.global_model[name] = server_lr * aggregated_delta[name]

        self.model_version += 1
        self.metrics.metrics['aggregations_performed'] += 1
        logger.info(f"FedBuff aggregated {len(updates_to_aggregate)} updates")

class ImprovedAsyncProtocol(FederatedProtocol):
    """Your improved asynchronous protocol with optimizations"""

    def configure(self, **kwargs):
        # Core parameters
        self.max_staleness = kwargs.get('max_staleness', 20)
        self.min_buffer_size = kwargs.get('min_buffer_size', 3)
        self.max_buffer_size = kwargs.get('max_buffer_size', 8)

        # Adaptive features
        self.adaptive_weighting = kwargs.get('adaptive_weighting', True)
        self.momentum = kwargs.get('momentum', 0.9)
        self.compression_ratio = kwargs.get('compression_ratio', 0.5)  # Important!

        # Internal state
        self.update_buffer = deque()
        self.momentum_buffer = {}
        self.client_contribution_scores = defaultdict(lambda: 1.0)
        self.network_health = 0.5

        # Start aggregation thread
        self.aggregation_thread = threading.Thread(target=self._smart_aggregation, daemon=True)
        self.aggregation_thread.start()

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            current_version = self.model_version
            staleness = current_version - update.model_version

            # Adaptive staleness threshold based on network health
            effective_max_staleness = self.max_staleness * (1.5 if self.network_health < 0.3 else 1.0)

            if staleness > effective_max_staleness:
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data) * self.compression_ratio,
                    accepted=False
                )
                return False, current_version

            # Apply compression (simulate by reducing communication cost)
            compressed_size = self.calculate_update_size(update.update_data) * self.compression_ratio

            update.staleness = staleness
            self.update_buffer.append(update)

            self.metrics.update_communication(compressed_size, accepted=True)

            # Update network health
            self._update_network_health(staleness)

            return True, current_version

    def _update_network_health(self, staleness: float):
        """Update network health estimation"""
        staleness_score = max(0, 1.0 - staleness / self.max_staleness)
        self.network_health = 0.9 * self.network_health + 0.1 * staleness_score

    def _smart_aggregation(self):
        """Smart aggregation with adaptive buffer size"""
        while self.running:
            time.sleep(0.05)

            with self._lock:
                buffer_size = len(self.update_buffer)

                # Adaptive buffer threshold
                if self.network_health > 0.7:
                    threshold = self.min_buffer_size
                elif self.network_health > 0.4:
                    threshold = (self.min_buffer_size + self.max_buffer_size) // 2
                else:
                    threshold = self.max_buffer_size

                # Aggregate if buffer reaches threshold or timeout
                should_aggregate = False

                if buffer_size >= threshold:
                    should_aggregate = True
                elif buffer_size >= self.min_buffer_size and self.update_buffer:
                    # Check oldest update age
                    oldest_age = time.time() - self.update_buffer[0].timestamp
                    if oldest_age > 2.0:  # 2 second timeout
                        should_aggregate = True

                if should_aggregate:
                    self.aggregate_updates()

    def aggregate_updates(self):
        """Intelligent aggregation with quality-based weighting + server learning rate + clipping"""
        if not self.update_buffer:
            return

        # Select best updates (quality-based selection)
        num_to_aggregate = min(len(self.update_buffer), self.max_buffer_size)

        # Sort by quality score with indices
        scored_updates = []
        for i, update in enumerate(self.update_buffer):
            quality_score = self._calculate_quality_score(update)
            scored_updates.append((quality_score, i, update))

        scored_updates.sort(key=lambda x: x[0], reverse=True)

        # Get selected updates and their indices
        selected_indices = set()
        selected_updates = []
        for score, idx, update in scored_updates[:num_to_aggregate]:
            selected_indices.add(idx)
            selected_updates.append(update)

        # Keep remaining updates (not selected)
        remaining_updates = [update for i, update in enumerate(self.update_buffer)
                             if i not in selected_indices]
        self.update_buffer.clear()
        self.update_buffer.extend(remaining_updates)

        # Weighted aggregation of DELTAS
        aggregated_delta = {}
        total_weight = 0

        for update in selected_updates:
            weight = self._calculate_weight(update)
            total_weight += weight

            for name, delta in update.update_data.items():
                if name not in aggregated_delta:
                    aggregated_delta[name] = torch.zeros_like(delta, dtype=torch.float32)
                if delta.dtype != torch.float32:
                    delta = delta.float()
                aggregated_delta[name] = aggregated_delta[name] + (delta * weight)

        if total_weight > 0:
            server_lr = getattr(self, "server_lr", 0.1)

            # Normalize
            for name in aggregated_delta:
                aggregated_delta[name] = aggregated_delta[name] / total_weight

            # Momentum smoothing
            if self.momentum > 0:
                for name in aggregated_delta:
                    if name in self.momentum_buffer:
                        self.momentum_buffer[name] = (
                                self.momentum * self.momentum_buffer[name] +
                                (1 - self.momentum) * aggregated_delta[name]
                        )
                        aggregated_delta[name] = self.momentum_buffer[name]
                    else:
                        self.momentum_buffer[name] = aggregated_delta[name]

            # Clip to prevent gradient explosion
            for name in aggregated_delta:
                aggregated_delta[name] = torch.clamp(aggregated_delta[name], -5.0, 5.0)

            # Apply delta to global model
            if self.global_model is None:
                self.global_model = {}
                for name in aggregated_delta:
                    self.global_model[name] = server_lr * aggregated_delta[name]
            else:
                for name in aggregated_delta:
                    if name in self.global_model:
                        self.global_model[name] = self.global_model[name] + server_lr * aggregated_delta[name]
                    else:
                        self.global_model[name] = server_lr * aggregated_delta[name]

        self.model_version += 1
        self.metrics.metrics['aggregations_performed'] += 1

        # Update metrics
        avg_staleness = np.mean([u.staleness for u in selected_updates])
        self.metrics.metrics['average_staleness'] = (
                0.9 * self.metrics.metrics['average_staleness'] + 0.1 * avg_staleness
        )
        high_quality_count = sum(1 for u in selected_updates
                                 if self._calculate_quality_score(u) > 0.7)
        self.metrics.metrics['high_quality_updates'] += high_quality_count

    def _calculate_quality_score(self, update: ClientUpdate) -> float:
        """Calculate update quality score"""
        # Staleness penalty
        staleness_score = max(0.1, 1.0 - update.staleness / self.max_staleness)

        # Loss improvement score
        loss_score = 1.0 / (1.0 + update.local_loss) if update.local_loss > 0 else 0.5

        # Client reliability score
        client_score = self.client_contribution_scores.get(update.client_id, 0.5)

        # Combined score
        return 0.4 * staleness_score + 0.3 * loss_score + 0.3 * client_score

    def _calculate_weight(self, update: ClientUpdate) -> float:
        """Calculate aggregation weight for update"""
        if not self.adaptive_weighting:
            return update.data_size

        quality_score = self._calculate_quality_score(update)
        staleness_penalty = max(0.2, 1.0 - update.staleness / self.max_staleness) ** 2

        return update.data_size * quality_score * staleness_penalty

class Scaffold(FederatedProtocol):
    """SCAFFOLD protocol: variance reduction with control variates"""

    def configure(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1.0)
        self.max_round_time = kwargs.get('max_round_time', 30.0)

        # Global control variate (same shape as model)
        self.c_global = {}
        self.client_controls = defaultdict(dict)

        self.current_round = 0
        self.round_buffer = []
        self.round_start_time = time.time()

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            # Add to round buffer
            self.round_buffer.append(update)

            # Record communication
            update_size = self.calculate_update_size(update.update_data)
            self.metrics.update_communication(update_size, accepted=True)

            min_clients = max(2, int(self.num_clients * 0.5))
            if len(self.round_buffer) >= min_clients:
                self.aggregate_updates()
                return True, self.current_round + 1

            return True, self.current_round

    def aggregate_updates(self):
        """FedAvg + control variate correction"""
        if not self.round_buffer:
            return

        total_data_size = sum(u.data_size for u in self.round_buffer)
        aggregated = {}

        # FedAvg aggregation (deltas)
        for update in self.round_buffer:
            weight = update.data_size / total_data_size
            for name, delta in update.update_data.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(delta, dtype=torch.float32)
                aggregated[name] += weight * delta.float()

        # Apply aggregated updates to global model
        if self.global_model is None:
            self.global_model = {}
            for name, param in aggregated.items():
                self.global_model[name] = param.clone()
        else:
            for name, param in aggregated.items():
                if name in self.global_model:
                    self.global_model[name] += self.learning_rate * param
                else:
                    self.global_model[name] = self.learning_rate * param

        for update in self.round_buffer:
            client_c = self.client_controls[update.client_id]
            for name, delta in update.update_data.items():
                if name not in self.c_global:
                    self.c_global[name] = torch.zeros_like(delta, dtype=torch.float32)
                if name in client_c:
                    self.c_global[name] += (delta - client_c[name]) / total_data_size
                else:
                    self.c_global[name] += delta / total_data_size

            self.client_controls[update.client_id] = copy.deepcopy(self.c_global)

        self.metrics.metrics['aggregations_performed'] += 1
        self.current_round += 1
        self.model_version = self.current_round
        self.round_buffer.clear()
        self.round_start_time = time.time()

        logger.info(f"SCAFFOLD Round {self.current_round} completed with {len(self.client_controls)} clients")

# Factory function to create protocols
def create_protocol(protocol_name: str, num_clients: int, **kwargs) -> FederatedProtocol:
    """Factory function to create protocol instances"""
    protocols = {
        'fedavg': SyncFedAvg,
        'fedasync': AsyncFedAvg,
        'fedbuff': FedBuff,
        'improved_async': ImprovedAsyncProtocol,
        'scaffold': Scaffold
    }

    if protocol_name.lower() not in protocols:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    return protocols[protocol_name.lower()](num_clients, **kwargs)