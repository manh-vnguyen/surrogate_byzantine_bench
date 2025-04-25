# FL-SNNs: Benchmarking the Byzantine-Robustness of Uniquely-Shaped Surrogate Gradients

This repository contains the implementation and benchmarking code for our research on Byzantine-robust federated learning using spiking neural networks (SNNs) with uniquely-shaped surrogate gradients.

## Overview

This project investigates the robustness of federated learning systems using spiking neural networks against Byzantine attacks, with a particular focus on how different surrogate gradient shapes affect the system's resilience. The implementation includes:

- Federated learning framework with support for multiple clients
- Spiking neural network models with various surrogate gradient implementations
- Byzantine attack scenarios and defense mechanisms
- Comprehensive benchmarking tools and experiment management

## Repository Structure

- `__actors.py`: Implementation of federated learning actors (clients and server)
- `__atks.py`: Byzantine attack implementations
- `__datasets.py`: Dataset loading and preprocessing utilities
- `__defs.py`: Core definitions and configurations
- `__lib.py`: Utility functions and helper classes
- `__models.py`: Neural network model implementations
- `__surr_grad.py`: Surrogate gradient implementations for SNNs
- `gen_exps_ann.py`: Experiment generation for artificial neural networks
- `gen_exps_snn.py`: Experiment generation for spiking neural networks
- `run_exps.py`: Main experiment runner with parallel processing support
- `plotting/`: Directory containing visualization and analysis scripts

## Requirements

- Python 3.x
- PyTorch
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:manh-vnguyen/surrogate_byzantine_bench.git
cd surrogate_byzantine_bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

To run experiments, use the `run_exps.py` script:

```bash
python run_exps.py --run_id [experiment_id]
```

### Generating New Experiments

To generate new experiment configurations:

```bash
# For SNN experiments
python gen_exps_snn.py

# For ANN experiments
python gen_exps_ann.py
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your-paper-citation,
  title={Federated Learning with Spiking Neural Networks: Benchmarking the Byzantine-Robustness of Uniquely-Shaped Surrogate Gradients},
  author={[Your Names]},
  journal={[Journal/Conference Name]},
  year={[Year]},
  publisher={[Publisher]}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]
