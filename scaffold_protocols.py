"""
scaffold_protocols.py
Implementation of SCAFFOLD and Asynchronous SCAFFOLD for federated learning
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
import copy
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import logging

from federated_protocol_framework import (
    FederatedProtocol, ClientUpdate, ProtocolMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SCAFFOLDUpdate(ClientUpdate):
    """Extended update for SCAFFOLD that includes control variates"""
    def __init__(self, client_id: str, update_data: Dict[str, torch.Tensor],
                 control_delta: Dict[str, torch.Tensor], model_version: int,
                 local_loss: float, data_size: int, timestamp: float,
                 staleness: Optional[float] = None):
        super().__init__(client_id, update_data, model_version,
                        local_loss, data_size, timestamp, staleness)
        self.control_delta = control_delta


class SyncSCAFFOLD(FederatedProtocol):
    """Synchronous SCAFFOLD implementation"""

    def configure(self, **kwargs):
        self.round_participation_rate = kwargs.get('participation_rate', 0.5)
        self.server_lr = kwargs.get('server_lr', 1.0)
        self.local_steps = kwargs.get('local_steps', 5)
        self.gradient_clip = kwargs.get('gradient_clip', 10.0)

        # SCAFFOLD specific
        self.server_control = {}  # c
        self.client_controls = defaultdict(dict)  # c_i for each client

        self.current_round = 0
        self.round_buffer = []
        self.round_start_time = time.time()

    def initialize_controls(self, model_state: Dict[str, torch.Tensor]):
        """Initialize control variates to zero"""
        for name, param in model_state.items():
            if 'num_batches_tracked' not in name:
                self.server_control[name] = torch.zeros_like(param, dtype=torch.float32)

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            # Ensure it's a SCAFFOLD update
            if not isinstance(update, SCAFFOLDUpdate):
                logger.warning("Received non-SCAFFOLD update in SCAFFOLD protocol")
                return False, self.current_round

            # Check if update is for current round
            if update.model_version != self.current_round:
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data) * 2,  # x2 for control variates
                    accepted=False
                )
                return False, self.current_round

            # Add to round buffer
            self.round_buffer.append(update)

            # Record metrics (double communication for control variates)
            update_size = self.calculate_update_size(update.update_data)
            control_size = self.calculate_update_size(update.control_delta)
            self.metrics.update_communication(update_size + control_size, accepted=True)

            # Check if we should aggregate
            min_clients = max(2, int(self.num_clients * self.round_participation_rate))
            if len(self.round_buffer) >= min_clients:
                self.aggregate_updates()
                return True, self.current_round + 1

            return True, self.current_round

    def aggregate_updates(self):
        """SCAFFOLD aggregation with control variate updates"""
        if not self.round_buffer:
            return

        # Initialize server control if needed
        if not self.server_control and self.round_buffer:
            sample_update = self.round_buffer[0]
            for name in sample_update.update_data:
                if 'num_batches_tracked' not in name:
                    self.server_control[name] = torch.zeros_like(
                        sample_update.update_data[name], dtype=torch.float32
                    )

        # Aggregate model updates (deltas)
        total_data_size = sum(u.data_size for u in self.round_buffer)
        aggregated_delta = {}
        aggregated_control_delta = {}

        for update in self.round_buffer:
            weight = update.data_size / total_data_size

            # Aggregate model deltas
            for name, delta in update.update_data.items():
                if 'num_batches_tracked' in name:
                    continue

                if name not in aggregated_delta:
                    aggregated_delta[name] = torch.zeros_like(delta, dtype=torch.float32)

                if delta.dtype != torch.float32:
                    delta = delta.float()

                # Clip gradients to prevent explosion
                delta = torch.clamp(delta, -self.gradient_clip, self.gradient_clip)
                aggregated_delta[name] = aggregated_delta[name] + (delta * weight)

            # Aggregate control deltas
            for name, c_delta in update.control_delta.items():
                if 'num_batches_tracked' in name:
                    continue

                if name not in aggregated_control_delta:
                    aggregated_control_delta[name] = torch.zeros_like(c_delta, dtype=torch.float32)

                if c_delta.dtype != torch.float32:
                    c_delta = c_delta.float()

                aggregated_control_delta[name] = aggregated_control_delta[name] + (c_delta * weight)

        # Update global model
        if self.global_model is None:
            self.global_model = {}

        for name in aggregated_delta:
            if name not in self.global_model:
                self.global_model[name] = torch.zeros_like(aggregated_delta[name])

            # Apply server learning rate
            self.global_model[name] = self.global_model[name] + self.server_lr * aggregated_delta[name]

        # Update server control variate
        option_ii = len(self.round_buffer) / self.num_clients  # Participation ratio
        for name in aggregated_control_delta:
            if name in self.server_control:
                self.server_control[name] = (self.server_control[name] +
                                            option_ii * aggregated_control_delta[name])

        # Update metrics
        self.metrics.metrics['aggregations_performed'] += 1
        round_time = time.time() - self.round_start_time
        self.metrics.metrics['average_round_time'] = (
            0.9 * self.metrics.metrics['average_round_time'] + 0.1 * round_time
        )

        # Prepare for next round
        num_clients = len(self.round_buffer)
        self.current_round += 1
        self.model_version = self.current_round
        self.round_buffer.clear()
        self.round_start_time = time.time()

        logger.info(f"SCAFFOLD Round {self.current_round} completed with {num_clients} clients")

    def get_control_variate(self, client_id: str) -> Dict[str, torch.Tensor]:
        """Get client control variate (or initialize if new)"""
        if client_id not in self.client_controls or not self.client_controls[client_id]:
            # Initialize client control to server control
            self.client_controls[client_id] = copy.deepcopy(self.server_control)
        return self.client_controls[client_id]

    def update_client_control(self, client_id: str, new_control: Dict[str, torch.Tensor]):
        """Update client's control variate"""
        self.client_controls[client_id] = copy.deepcopy(new_control)


class AsyncSCAFFOLD(FederatedProtocol):
    """Asynchronous SCAFFOLD implementation"""

    def configure(self, **kwargs):
        self.max_staleness = kwargs.get('max_staleness', 15)
        self.server_lr = kwargs.get('server_lr', 0.9)
        self.momentum = kwargs.get('momentum', 0.9)
        self.gradient_clip = kwargs.get('gradient_clip', 10.0)
        self.buffer_size = kwargs.get('buffer_size', 5)

        # SCAFFOLD specific
        self.server_control = {}
        self.client_controls = defaultdict(dict)
        self.momentum_buffer = {}

        # Async specific
        self.update_buffer = deque()
        self.aggregation_thread = threading.Thread(
            target=self._async_aggregation, daemon=True
        )
        self.aggregation_thread.start()

    def initialize_controls(self, model_state: Dict[str, torch.Tensor]):
        """Initialize control variates to zero"""
        for name, param in model_state.items():
            if 'num_batches_tracked' not in name:
                self.server_control[name] = torch.zeros_like(param, dtype=torch.float32)

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            # Ensure it's a SCAFFOLD update
            if not isinstance(update, SCAFFOLDUpdate):
                logger.warning("Received non-SCAFFOLD update")
                return False, self.model_version

            current_version = self.model_version
            staleness = current_version - update.model_version

            # Reject if too stale
            if staleness > self.max_staleness:
                update_size = self.calculate_update_size(update.update_data)
                control_size = self.calculate_update_size(update.control_delta)
                self.metrics.update_communication(
                    update_size + control_size, accepted=False
                )
                return False, current_version

            # Accept update
            update.staleness = staleness
            self.update_buffer.append(update)

            # Record metrics
            update_size = self.calculate_update_size(update.update_data)
            control_size = self.calculate_update_size(update.control_delta)
            self.metrics.update_communication(update_size + control_size, accepted=True)

            # Update staleness metrics
            self.metrics.metrics['average_staleness'] = (
                0.9 * self.metrics.metrics['average_staleness'] + 0.1 * staleness
            )

            return True, current_version

    def _async_aggregation(self):
        """Asynchronous aggregation thread"""
        while self.running:
            time.sleep(0.1)

            with self._lock:
                if len(self.update_buffer) >= self.buffer_size:
                    self.aggregate_updates()

    def aggregate_updates(self):
        """Async SCAFFOLD aggregation with staleness weighting"""
        if not self.update_buffer:
            return

        # Initialize controls if needed
        if not self.server_control and self.update_buffer:
            sample_update = self.update_buffer[0]
            for name in sample_update.update_data:
                if 'num_batches_tracked' not in name:
                    self.server_control[name] = torch.zeros_like(
                        sample_update.update_data[name], dtype=torch.float32
                    )

        # Select updates to aggregate
        num_to_aggregate = min(len(self.update_buffer), self.buffer_size * 2)
        updates_to_aggregate = []
        for _ in range(num_to_aggregate):
            if self.update_buffer:
                updates_to_aggregate.append(self.update_buffer.popleft())

        # Weighted aggregation with staleness penalty
        aggregated_delta = {}
        aggregated_control_delta = {}
        total_weight = 0

        for update in updates_to_aggregate:
            # Staleness-aware weighting
            staleness_penalty = 1.0 / (1.0 + update.staleness)
            weight = staleness_penalty * update.data_size
            total_weight += weight

            # Aggregate model deltas
            for name, delta in update.update_data.items():
                if 'num_batches_tracked' in name:
                    continue

                if name not in aggregated_delta:
                    aggregated_delta[name] = torch.zeros_like(delta, dtype=torch.float32)

                if delta.dtype != torch.float32:
                    delta = delta.float()

                # Clip to prevent explosion
                delta = torch.clamp(delta, -self.gradient_clip, self.gradient_clip)
                aggregated_delta[name] = aggregated_delta[name] + (delta * weight)

            # Aggregate control deltas
            for name, c_delta in update.control_delta.items():
                if 'num_batches_tracked' in name:
                    continue

                if name not in aggregated_control_delta:
                    aggregated_control_delta[name] = torch.zeros_like(c_delta, dtype=torch.float32)

                if c_delta.dtype != torch.float32:
                    c_delta = c_delta.float()

                aggregated_control_delta[name] = aggregated_control_delta[name] + (c_delta * weight)

        if total_weight > 0:
            # Normalize aggregated deltas
            for name in aggregated_delta:
                aggregated_delta[name] = aggregated_delta[name] / total_weight

            for name in aggregated_control_delta:
                aggregated_control_delta[name] = aggregated_control_delta[name] / total_weight

            # Apply momentum
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

            # Update global model
            if self.global_model is None:
                self.global_model = {}

            for name in aggregated_delta:
                if name not in self.global_model:
                    self.global_model[name] = torch.zeros_like(aggregated_delta[name])

                self.global_model[name] = (self.global_model[name] +
                                         self.server_lr * aggregated_delta[name])

            # Update server control
            participation_ratio = len(updates_to_aggregate) / max(1, self.num_clients)
            for name in aggregated_control_delta:
                if name in self.server_control:
                    self.server_control[name] = (self.server_control[name] +
                                                participation_ratio * aggregated_control_delta[name])

        self.model_version += 1
        self.metrics.metrics['aggregations_performed'] += 1

        logger.info(f"Async SCAFFOLD aggregated {len(updates_to_aggregate)} updates")

    def get_control_variate(self, client_id: str) -> Dict[str, torch.Tensor]:
        """Get client control variate"""
        if client_id not in self.client_controls or not self.client_controls[client_id]:
            self.client_controls[client_id] = copy.deepcopy(self.server_control)
        return self.client_controls[client_id]

    def update_client_control(self, client_id: str, new_control: Dict[str, torch.Tensor]):
        """Update client's control variate"""
        self.client_controls[client_id] = copy.deepcopy(new_control)


def create_scaffold_protocol(protocol_type: str, num_clients: int, **kwargs) -> FederatedProtocol:
    """Factory function for SCAFFOLD protocols"""
    if protocol_type.lower() == 'scaffold':
        return SyncSCAFFOLD(num_clients, **kwargs)
    elif protocol_type.lower() == 'async_scaffold':
        return AsyncSCAFFOLD(num_clients, **kwargs)
    else:
        raise ValueError(f"Unknown SCAFFOLD protocol type: {protocol_type}")


# Helper function for SCAFFOLD client training
def train_client_scaffold(model: nn.Module, dataset, global_model: Dict,
                         server_control: Dict, client_control: Dict,
                         epochs: int = 3, lr: float = 0.01) -> Tuple[Dict, Dict, float, int]:
    """
    Train client with SCAFFOLD corrections
    Returns: (model_delta, control_delta, loss, data_size)
    """
    if len(dataset) == 0:
        return {}, {}, float('inf'), 0

    from torch.utils.data import DataLoader
    import torch.optim as optim

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    # Save initial model state
    initial_state = copy.deepcopy(model.state_dict())

    total_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Apply SCAFFOLD correction (c - c_i)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in server_control:
                        correction = server_control[name] - client_control.get(name,
                                                                              torch.zeros_like(param))
                        param.grad = param.grad - correction

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        total_loss += epoch_loss / len(dataloader)

    avg_loss = total_loss / epochs

    # Calculate model delta
    model_delta = {}
    final_state = model.state_dict()
    for name in final_state:
        if 'num_batches_tracked' not in name:
            model_delta[name] = final_state[name] - initial_state[name]

    # Update client control variate
    # c_i^+ = c_i - c + (1/(K*eta)) * (x - y)
    # where K is local steps, eta is learning rate
    new_client_control = {}
    for name in model_delta:
        if name in client_control and name in server_control:
            # Option II from SCAFFOLD paper
            new_client_control[name] = (client_control[name] - server_control[name] +
                                       model_delta[name] / (epochs * lr))
        else:
            new_client_control[name] = model_delta[name] / (epochs * lr)

    # Control delta (c_i^+ - c_i)
    control_delta = {}
    for name in new_client_control:
        old_control = client_control.get(name, torch.zeros_like(new_client_control[name]))
        control_delta[name] = new_client_control[name] - old_control

    return model_delta, control_delta, avg_loss, len(dataset)