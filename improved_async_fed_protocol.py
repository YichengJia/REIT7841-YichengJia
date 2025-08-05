"""
improved_async_fed_protocol.py
Key improvements:
Better aggression strategy, better staleness penalty, better adaptive weighting
"""

import copy
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math

import numpy as np
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UpdateInfo:
    """Update information structure"""
    client_id: str
    timestamp: float
    update_data: Dict[str, torch.Tensor]
    staleness: float = 0.0
    local_loss: float = float('inf')
    data_size: int = 0


class TraditionalFedAvg:
    """Traditional FedAvg"""
    def __init__(self, num_clients: int, rounds: int = 100, wait_timeout: float = 30.0):
        self.num_clients = num_clients
        self.rounds = rounds
        self.wait_timeout = wait_timeout
        self.global_model = None
        self.communication_stats = {
            'total_data_transmitted': 0.0,
            'rounds_completed': 0,
            'failed_rounds': 0,
            'average_waiting_time': 0.0,
            'total_updates': 0,
            'total_time': 0.0
        }

    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]],
                  client_data_sizes: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return {}
        if client_data_sizes is None:
            client_data_sizes = [1] * len(client_updates)
        total_data_size = sum(client_data_sizes)
        weights = [size / total_data_size for size in client_data_sizes]
        aggregated = {}
        for i, update in enumerate(client_updates):
            weight = weights[i]
            for name, param in update.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(param, dtype=torch.float)
                aggregated[name] += (param * weight).to(torch.float)
        for update in client_updates:
            self.communication_stats['total_data_transmitted'] += self._calculate_size(update)
        self.communication_stats['total_updates'] += len(client_updates)
        self.communication_stats['rounds_completed'] += 1
        return aggregated

    def _calculate_size(self, update: Dict[str, torch.Tensor]) -> float:
        total_size = 0
        for tensor in update.values():
            if tensor is not None:
                total_size += tensor.numel() * tensor.element_size()
        return total_size / (1024 * 1024)


class SuperiorAsyncFedProtocol:
    def __init__(self,
                 max_staleness: float = 30.0,
                 min_buffer_size: int = 2,
                 max_buffer_size: int = 8,
                 adaptive_weighting: bool = True,
                 momentum: float = 0.9,
                 staleness_penalty: str = 'adaptive',
                 learning_rate_decay: float = 0.95,
                 quality_threshold: float = 0.1):
        self.max_staleness = max_staleness
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.adaptive_weighting = adaptive_weighting
        self.momentum = momentum
        self.staleness_penalty = staleness_penalty
        self.learning_rate_decay = learning_rate_decay
        self.quality_threshold = quality_threshold
        self.update_buffer = deque()
        self.global_model = None
        self.model_version = 0
        self.momentum_buffer = {}
        self.global_learning_rate = 1.0
        self.client_versions = defaultdict(int)
        self.client_weights = defaultdict(lambda: 1.0)
        self.client_history = defaultdict(lambda: deque(maxlen=20))
        self.client_loss_history = defaultdict(lambda: deque(maxlen=10))
        self.client_contribution_scores = defaultdict(float)
        self.current_buffer_target = min_buffer_size
        self.recent_convergence_rate = deque(maxlen=10)
        self.buffer_lock = threading.Lock()
        self.model_lock = threading.RLock()
        self.stats = {
            'total_updates': 0,
            'accepted_updates': 0,
            'rejected_updates': 0,
            'high_quality_updates': 0,
            'low_quality_updates': 0,
            'average_staleness': 0.0,
            'total_data_transmitted': 0.0,
            'aggregations_performed': 0,
            'average_buffer_wait': 0.0,
            'convergence_improvements': 0,
            'adaptive_lr_adjustments': 0
        }
        self.running = True
        self.aggregator = threading.Thread(target=self._intelligent_aggregation_loop, daemon=True)
        self.aggregator.start()

    def submit_update(self, client_id: str, update_data: Dict[str, torch.Tensor],
                      client_version: int, local_loss: float = float('inf'),
                      data_size: int = 1) -> Tuple[bool, int]:
        self.stats['total_updates'] += 1
        with self.model_lock:
            current_version = self.model_version
            staleness = current_version - client_version
            effective_max_staleness = self._compute_effective_max_staleness()
            if staleness > effective_max_staleness:
                self.stats['rejected_updates'] += 1
                logger.debug(f"Rejected update from {client_id}: staleness {staleness} > {effective_max_staleness}")
                return False, current_version
            quality_score = self._evaluate_update_quality(client_id, local_loss, staleness)
            update_info = UpdateInfo(
                client_id=client_id,
                timestamp=time.time(),
                update_data=update_data,
                staleness=max(0, staleness),
                local_loss=local_loss,
                data_size=data_size
            )
            with self.buffer_lock:
                self.update_buffer.append(update_info)
                self.stats['accepted_updates'] += 1
                self.stats['total_data_transmitted'] += self._calculate_size(update_data)
                if quality_score > self.quality_threshold:
                    self.stats['high_quality_updates'] += 1
                else:
                    self.stats['low_quality_updates'] += 1
            self.client_versions[client_id] = current_version
            self.client_loss_history[client_id].append(local_loss)
            if self.adaptive_weighting:
                self._update_client_metrics(client_id, update_info, quality_score)
            return True, current_version

    def _intelligent_aggregation_loop(self):
        while self.running:
            time.sleep(0.05)
            should_aggregate, updates_to_use = self._should_perform_aggregation()
            if should_aggregate and updates_to_use:
                with self.buffer_lock:
                    selected_updates = []
                    remaining_updates = []
                    for update in self.update_buffer:
                        if len(selected_updates) < updates_to_use:
                            selected_updates.append(update)
                        else:
                            remaining_updates.append(update)
                    self.update_buffer.clear()
                    self.update_buffer.extend(remaining_updates)
                if selected_updates:
                    self._perform_superior_aggregation(selected_updates)

    def _should_perform_aggregation(self) -> Tuple[bool, int]:
        with self.buffer_lock:
            buffer_size = len(self.update_buffer)
            if buffer_size == 0:
                return False, 0
            self._adjust_buffer_target()
            if buffer_size >= self.current_buffer_target:
                return True, min(buffer_size, self.max_buffer_size)
            if buffer_size >= self.min_buffer_size:
                oldest_update = min(self.update_buffer, key=lambda x: x.timestamp)
                wait_time = time.time() - oldest_update.timestamp
                if wait_time > 2.0:
                    return True, buffer_size
            high_quality_count = sum(1 for u in self.update_buffer
                                   if self._evaluate_update_quality(u.client_id, u.local_loss, u.staleness) > self.quality_threshold)
            if high_quality_count >= self.min_buffer_size and buffer_size >= self.min_buffer_size:
                return True, buffer_size
            return False, 0

    def _perform_superior_aggregation(self, updates: List[UpdateInfo]):
        with self.model_lock:
            if not updates:
                return
            weights = self._compute_intelligent_weights(updates)
            total_weight = sum(weights)
            if total_weight <= 0:
                return
            weights = [w / total_weight for w in weights]
            aggregated = {}
            for update, weight in zip(updates, weights):
                for name, param in update.update_data.items():
                    if name not in aggregated:
                        aggregated[name] = torch.zeros_like(param)
                    aggregated[name] += param * weight
            self._adjust_global_learning_rate(updates)
            for name, param in aggregated.items():
                param = param * self.global_learning_rate
                if name in self.momentum_buffer:
                    self.momentum_buffer[name] = (
                        self.momentum * self.momentum_buffer[name] +
                        (1 - self.momentum) * param
                    )
                    aggregated[name] = self.momentum_buffer[name]
                else:
                    self.momentum_buffer[name] = param
            if self.global_model is None:
                self.global_model = aggregated
            else:
                for name, param in aggregated.items():
                    if name in self.global_model:
                        self.global_model[name] += param
                    else:
                        self.global_model[name] = param
            self._update_aggregation_stats(updates)
            self.model_version += 1
            self.stats['aggregations_performed'] += 1
            logger.info(f"Superior aggregation #{self.stats['aggregations_performed']}: "
                       f"version {self.model_version}, {len(updates)} updates, "
                       f"avg_staleness: {np.mean([u.staleness for u in updates]):.2f}, "
                       f"global_lr: {self.global_learning_rate:.4f}")

    def _compute_intelligent_weights(self, updates: List[UpdateInfo]) -> List[float]:
        weights = []
        for update in updates:
            staleness_weight = self._compute_advanced_staleness_penalty(update.staleness)
            quality_weight = self._compute_quality_weight(update.client_id)
            if update.local_loss != float('inf') and update.local_loss > 0:
                loss_weight = 1.0 / (1.0 + update.local_loss)
            else:
                loss_weight = 0.5
            data_weight = math.sqrt(update.data_size)
            contribution_weight = self.client_contribution_scores.get(update.client_id, 1.0)
            final_weight = (
                0.3 * staleness_weight +
                0.25 * quality_weight +
                0.2 * loss_weight +
                0.15 * data_weight +
                0.1 * contribution_weight
            )
            weights.append(max(0.01, final_weight))
        return weights

    def _compute_advanced_staleness_penalty(self, staleness: float) -> float:
        if staleness <= 0:
            return 1.0
        if self.staleness_penalty == 'adaptive':
            network_health = self._estimate_network_health()
            base_penalty = 1.0 - (staleness / self.max_staleness)
            if network_health > 0.8:
                return max(0.1, base_penalty ** 1.5)
            elif network_health > 0.5:
                return max(0.2, base_penalty)
            else:
                return max(0.3, base_penalty ** 0.7)
        elif self.staleness_penalty == 'polynomial':
            return max(0.1, (1.0 - staleness / self.max_staleness) ** 2)
        else:
            return max(0.1, 1.0 - staleness / self.max_staleness)

    def _estimate_network_health(self) -> float:
        if not self.client_history:
            return 0.5
        recent_updates = []
        for client_updates in self.client_history.values():
            recent_updates.extend(list(client_updates)[-5:])
        if not recent_updates:
            return 0.5
        avg_staleness = np.mean([u.staleness for u in recent_updates])
        update_frequency = len(recent_updates) / max(1, len(self.client_history))
        staleness_score = max(0, 1.0 - avg_staleness / self.max_staleness)
        frequency_score = min(1.0, update_frequency / 5.0)
        return 0.6 * staleness_score + 0.4 * frequency_score

    def _evaluate_update_quality(self, client_id: str, local_loss: float, staleness: float) -> float:
        if local_loss != float('inf') and local_loss > 0:
            loss_score = 1.0 / (1.0 + local_loss)
        else:
            loss_score = 0.5
        freshness_score = max(0.1, 1.0 - staleness / self.max_staleness)
        history_score = self.client_contribution_scores.get(client_id, 0.5)
        return 0.4 * loss_score + 0.4 * freshness_score + 0.2 * history_score

    def _compute_quality_weight(self, client_id: str) -> float:
        history = self.client_history[client_id]
        if len(history) < 2:
            return 1.0
        loss_history = self.client_loss_history[client_id]
        if len(loss_history) >= 2:
            recent_losses = list(loss_history)[-5:]
            if len(recent_losses) >= 2:
                loss_trend = recent_losses[0] - recent_losses[-1]
                trend_weight = 0.5 + 0.5 * math.tanh(loss_trend)
            else:
                trend_weight = 0.5
        else:
            trend_weight = 0.5
        timestamps = [h.timestamp for h in history]
        if len(timestamps) >= 3:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            stability_weight = 1.0 / (1.0 + std_interval / (avg_interval + 1e-6))
        else:
            stability_weight = 0.5
        return 0.6 * trend_weight + 0.4 * stability_weight

    def _adjust_global_learning_rate(self, updates: List[UpdateInfo]):
        avg_loss = np.mean([u.local_loss for u in updates if u.local_loss != float('inf')])
        if not hasattr(self, 'prev_avg_loss'):
            self.prev_avg_loss = avg_loss
            return
        if avg_loss < self.prev_avg_loss:
            self.global_learning_rate = min(2.0, self.global_learning_rate * 1.02)
            self.stats['convergence_improvements'] += 1
        else:
            self.global_learning_rate = max(0.1, self.global_learning_rate * 0.98)
        self.prev_avg_loss = avg_loss
        self.stats['adaptive_lr_adjustments'] += 1

    def _adjust_buffer_target(self):
        network_health = self._estimate_network_health()
        if network_health > 0.8:
            self.current_buffer_target = min(self.max_buffer_size, self.min_buffer_size + 2)
        elif network_health < 0.4:
            self.current_buffer_target = self.min_buffer_size
        else:
            self.current_buffer_target = self.min_buffer_size + 1

    def _update_client_metrics(self, client_id: str, update_info: UpdateInfo, quality_score: float):
        self.client_history[client_id].append(update_info)
        alpha = 0.1
        old_score = self.client_contribution_scores.get(client_id, 0.5)
        self.client_contribution_scores[client_id] = (
            alpha * quality_score + (1 - alpha) * old_score
        )

    def _update_aggregation_stats(self, updates: List[UpdateInfo]):
        avg_staleness = np.mean([u.staleness for u in updates])
        self.stats['average_staleness'] = (
            0.9 * self.stats['average_staleness'] + 0.1 * avg_staleness
        )
        current_time = time.time()
        avg_wait = np.mean([current_time - u.timestamp for u in updates])
        self.stats['average_buffer_wait'] = (
            0.9 * self.stats['average_buffer_wait'] + 0.1 * avg_wait
        )

    def _compute_effective_max_staleness(self) -> float:
        network_health = self._estimate_network_health()
        if network_health > 0.8:
            return self.max_staleness * 0.8
        elif network_health < 0.4:
            return self.max_staleness * 1.5
        else:
            return self.max_staleness

    def _calculate_size(self, update: Dict[str, torch.Tensor]) -> float:
        total_size = 0
        for tensor in update.values():
            if tensor is not None:
                total_size += tensor.numel() * tensor.element_size()
        return total_size / (1024 * 1024)

    def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        with self.model_lock:
            if self.global_model is not None:
                return copy.deepcopy(self.global_model)
            return None

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats.update({
            'current_buffer_size': len(self.update_buffer),
            'active_clients': len(self.client_versions),
            'global_learning_rate': self.global_learning_rate,
            'current_buffer_target': self.current_buffer_target,
            'network_health': self._estimate_network_health(),
            'high_quality_ratio': (
                self.stats['high_quality_updates'] /
                max(1, self.stats['accepted_updates'])
            ),
            'contribution_scores': dict(self.client_contribution_scores)
        })
        return stats

    def shutdown(self):
        self.running = False
        if self.aggregator.is_alive():
            self.aggregator.join(timeout=5.0)
        logger.info("Superior async protocol shutdown complete")