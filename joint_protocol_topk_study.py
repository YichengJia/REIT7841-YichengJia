# joint_protocol_topk_study.py
# Compare multiple protocols across Top-K values, using Intent-F1 and Explanation-BLEU.

import time
import json
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, TensorDataset

from federated_protocol_framework import create_protocol, ClientUpdate
from compression_strategies import TopKCompressor
from unified_protocol_comparison import SimpleNN, generate_federated_data, train_client
from metrics import macro_f1, corpus_bleu
from unified_protocol_comparison import evaluate_model  # keep if you need loss logging
from unified_protocol_comparison import set_seed
from unified_protocol_comparison import evaluate_with_intent_and_explanation  # from section III

# ------------------ experiment config ------------------
PROTOCOLS = ["fedavg", "fedasync", "fedbuff", "scaffold", "improved_async"]
TOPK_LIST = [None, 1, 10, 50, 100, 200, 500, 1000]  # None = no compression
NUM_CLIENTS = 20
ROUNDS = 100
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
INPUT_DIM = 16
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Efficiency score weights (all in 0~1)
W_INTENT = 0.6
W_BLEU   = 0.2
W_COMM   = 0.2  # uses normalized communication: higher is better (1 - comm/comm_ref)

def normalized_score(intent_f1, bleu, comm_mb, comm_ref):
    comm_norm = max(0.0, 1.0 - comm_mb / comm_ref) if comm_ref > 0 else 0.0
    return W_INTENT * intent_f1 + W_BLEU * bleu + W_COMM * comm_norm

def main():
    set_seed(42)
    print("\n" + "="*70)
    print("JOINT STUDY: Protocols x Top-K (Intent-F1 & Explanation-BLEU)")
    print("="*70)

    # -------- data --------
    client_datasets, test_dataset = generate_federated_data(
        num_clients=NUM_CLIENTS,
        samples_per_client=500,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        heterogeneity=0.5
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # reference comm to normalize (approx upper bound: largest k expected payload * rounds * clients rate)
    COMM_REF_MB = 40.0  # adjust after first run if you prefer tighter normalization

    results = defaultdict(dict)

    for proto in PROTOCOLS:
        for k in TOPK_LIST:
            cfg = {
                "participation_rate": 0.5,
                "buffer_size": 5,
                "max_staleness": 10,
                "compression": ("topk" if k is not None else None),
                "k": (k if k is not None else 0),
                "num_bits": 8,
                "server_lr": 0.2,
                "gradient_clip": 1.0,
                "adaptive_weighting": True,
                "min_buffer_size": 2,
                "max_buffer_size": 6,
                "momentum": 0.8,
            }
            protocol = create_protocol(proto, NUM_CLIENTS, **cfg)

            # init global model
            global_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES).to(DEVICE)
            protocol.set_global_model(global_model.state_dict())

            print(f"\nRunning {proto} with k={k}")
            start = time.time()

            # train rounds
            for r in range(ROUNDS):
                for cid, ds in enumerate(client_datasets):
                    # local train -> compute delta on CPU to match train_client
                    local_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES)
                    # pull
                    state = protocol.get_global_model()
                    if state:
                        local_model.load_state_dict(state, strict=False)

                    updated_state, local_loss, n_samples = train_client(
                        local_model, ds, epochs=LOCAL_EPOCHS, lr=0.01
                    )

                    # compute parameter delta = updated - pulled_global
                    update_dict = {}
                    if state:
                        for name, p_new in updated_state.items():
                            # ignore BatchNorm counters
                            if name in state and 'num_batches_tracked' not in name:
                                update_dict[name] = (p_new.float() - state[name].float())

                    else:
                        # first round: treat weights as delta from zeros
                        for name, p_new in updated_state.items():
                            if 'num_batches_tracked' not in name:
                                update_dict[name] = p_new.float()

                    update = ClientUpdate(
                        client_id=str(cid),
                        update_data=update_dict,  # pass deltas (server may compress if a compressor is configured)
                        model_version=protocol.model_version,
                        local_loss=float(local_loss),
                        data_size=int(n_samples),
                        timestamp=time.time(),
                    )
                    protocol.receive_update(update)

                # synchronous protocols aggregate inside receive_update; for async we can sleep a bit
                time.sleep(0.01)

            # evaluate
            # reconstruct global model on a fresh model instance
            eval_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES).to(DEVICE)
            state = protocol.get_global_model()
            if state:
                eval_model.load_state_dict(state, strict=False)

            intent_f1, expl_bleu, _, _ = evaluate_with_intent_and_explanation(
                eval_model, test_loader, DEVICE, id2label={i: f"intent_{i}" for i in range(NUM_CLASSES)}
            )

            comm_mb = protocol.metrics.metrics["total_data_transmitted_mb"]
            aggs   = protocol.metrics.metrics["aggregations_performed"]
            score  = normalized_score(intent_f1, expl_bleu, comm_mb, COMM_REF_MB)
            elapsed = time.time() - start

            results[proto][str(k)] = {
                "intent_f1": round(intent_f1, 4),
                "explanation_bleu": round(expl_bleu, 4),
                "communication_mb": round(comm_mb, 2),
                "aggregations": int(aggs),
                "elapsed_sec": round(elapsed, 1),
                "score": round(score, 4),
            }

            print(f"  Intent-F1={intent_f1:.4f}  BLEU={expl_bleu:.4f}  Comm={comm_mb:.2f}MB  Agg={aggs}  Score={score:.4f}")

    # pretty print + save
    print("\n" + "="*70)
    print("SUMMARY (best per protocol)")
    print("="*70)
    for proto in PROTOCOLS:
        best_k, best = None, None
        for k, v in results[proto].items():
            if best is None or v["score"] > best["score"]:
                best_k, best = k, v
        print(f"{proto:12s}  best_k={best_k:>4}  score={best['score']:.4f}  "
              f"Intent-F1={best['intent_f1']:.4f}  BLEU={best['explanation_bleu']:.4f}  "
              f"Comm={best['communication_mb']:.4f}MB  Agg={best['aggregations']}")

    with open("joint_protocol_topk_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: joint_protocol_topk_results.json")

if __name__ == "__main__":
    main()
