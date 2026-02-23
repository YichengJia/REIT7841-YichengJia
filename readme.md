# Staleness-Aware Asynchronous Federated Learning for Heterogeneous Robotic Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-yellow)](#)

A unified research codebase for **asynchronous federated learning (FL)** under:
- client heterogeneity (non-IID, dynamic participation),
- unstable networks (stale updates),
- communication constraints (compressed updates).

The core method is an **Improved Async Protocol** with:
1. staleness-aware aggregation,
2. adaptive buffering,
3. pluggable compression.

---

## Table of Contents
- [1. Highlights](#1-highlights)
- [2. Problem Setting](#2-problem-setting)
- [3. Core Method](#3-core-method)
- [4. Repository Structure](#4-repository-structure)
- [5. Installation](#5-installation)
- [6. Quick Start](#6-quick-start)
- [7. Journal-Oriented Evaluation](#7-journal-oriented-evaluation)
- [8. Reproducibility Notes](#8-reproducibility-notes)
- [9. Citation](#9-citation)

---

## 1. Highlights

- **Asynchronous FL protocols in one framework**: FedAvg, FedAsync, FedBuff, SCAFFOLD, Improved Async.
- **Configurable staleness decay** for improved async: `linear`, `quadratic`, `exp`.
- **Compression options**: Top-K, SignSGD, QSGD.
- **Budget-aware evaluation**:
  - communication budget (`comm_budget_mb`),
  - latency budget (`latency_budget_sec`),
  - tri-objective score (Accuracy–Communication–Latency),
  - feasibility flag (`within_budget`).

---

## 2. Problem Setting

We target cross-device FL with:
- heterogeneous clients,
- asynchronous update arrivals,
- stale updates (server receives updates based on older model versions),
- communication-constrained uplink.

The code supports reporting results under explicit constrained objectives:
- maximize accuracy under communication/latency budgets,
- or rank by a tri-objective composite score.

---

## 3. Core Method

### Improved Async (main algorithm)
- staleness-aware update weighting,
- adaptive buffer trigger based on network health,
- optional scale-aware parameter defaults (`auto_scale_params`),
- configurable staleness decay (`staleness_mode`, `staleness_floor`),
- momentum smoothing + gradient clipping on server update.

### Compression
- Top-K sparsification,
- SignSGD 1-bit sign packing with magnitude scaling,
- QSGD quantization.

---

## 4. Repository Structure

```text
.
├── federated_protocol_framework.py      # Protocol implementations + factory
├── compression_strategies.py            # TopK / SignSGD / QSGD
├── metrics.py                           # F1, BLEU, tri-objective, budget checks
├── unified_protocol_comparison.py       # Main protocol comparison runner
├── intelligent_parameter_tuning.py      # Grid-style tuner (tri-objective ranking)
├── joint_protocol_topk_study.py         # Protocol × Top-K study
├── optimize_improved_async.py           # Improved-async optimization helper
├── optimized_protocol_config.py         # Scenario configs + scale sweep helpers
└── readme.md
