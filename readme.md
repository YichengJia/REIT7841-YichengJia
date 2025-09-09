# Federated Asynchronous Communication Protocol for Heterogeneous Robotic Systems

Project Structure
federated-robotic-protocol/
│
├── Core Framework
│   ├── federated_protocol_framework.py    # Protocol implementations
│   ├── compression_strategies.py          # Compression methods
│   └── metrics.py                         # Evaluation metrics
│
├── Experiments
│   ├── unified_protocol_comparison.py     # Main comparison
│   ├── intelligent_parameter_tuning.py    # Auto-tuning
│   ├── joint_protocol_topk_study.py      # Compression study
│   └── optimized_protocol_config.py      # Config management
│
├── Documentation
│   ├── README.md                         # This file
│   └── Thesis Draft.docx                 # Full thesis
│
└── Results
    ├── optimal_improved_async_config.json
    └── joint_protocol_topk_results.json

# Setup
## Clone the repository
git clone https://github.com/yourusername/federated-robotic-protocol.git
cd federated-robotic-protocol

## Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies
pip install torch torchvision numpy scikit-learn matplotlib

# Basic usage
from federated_protocol_framework import create_protocol
from unified_protocol_comparison import SimpleNN, generate_federated_data

## Create protocol with optimal configuration
protocol = create_protocol(
    'improved_async',
    num_clients=50,
    max_staleness=15,
    min_buffer_size=2,
    max_buffer_size=5,
    momentum=0.9,
    compression='signsgd'
)

## Initialize model
model = SimpleNN(input_dim=12, hidden_dim=32, output_dim=3)
protocol.set_global_model(model.state_dict())

## Run Comprehensive Comparison
### Compare all protocols
python unified_protocol_comparison.py

### Run parameter optimization
python intelligent_parameter_tuning.py

### Evaluate protocol-compression combinations
python joint_protocol_topk_study.py
