# Chemeleon-DNG: Chemeleon for De Novo Generation

While [Chemeleon](https://github.com/hspark1212/chemeleon) GitHub repository focuses on text-guided crystal structure generation, this repository provides a framework for **De Novo Generation (DNG)** and **Crystal Structure Prediction (CSP)** tasks.

- **CSP (Crystal Structure Prediction)**: Predicts stable crystal structures from given atom types
- **DNG (De Novo Generation)**: Generates new crystal structures from scratch

## Installation

### Prerequisites

- Python 3.9+
- PyTorch >= 2.1.0
- CUDA (optional, for GPU acceleration)

### Install the Package

```bash
conda create -n chemeleon-dng python=3.11
conda activate chemeleon-dng
git clone https://github.com/hspark1212/chemeleon-dng.git
cd chemeleon-dng
pip install -e .
```

## Quick Start

### Crystal Structure Prediction (CSP)

Generate crystal structures for given chemical formulas:

```python
from chemeleon_dng.sample import sample

sample(
    task="csp",
    formulas=["NaCl", "LiMnO2"],
    num_samples=10,
    output_dir="results",
    device="cpu"
)
```

> [!TIP]
> Invoke help(sample) to explore all available parameters and usage examples.

For the command line interface, you can use the following command:

```bash
python chemeleon_dng/sample.py --task=csp --formulas="NaCl,LiMnO2" --num_samples=10 --output_dir="results" --device=cpu
```

This command generates 10 crystal structures for the given formulas using the CSP task and saves the CIF files of the generated structures in the `results/` directory using CPU.

### De Novo Generation (DNG)

Generate novel crystal structures without predefined compositions:

```python
from chemeleon_dng.sample import sample

sample(
    task="dng",
    num_samples=200,
    batch_size=100,
    output_dir="results",
    device="cuda"
)
```

For the command line interface, you can use the following command:

```bash
python scripts/sample.py --task=dng --num_samples=200 --batch_size=100 --output_dir="results" --device=cuda
```

This command generates 200 random crystal structures using the DNG task with two batches of 100 each, and saves the generated structures in the `results/` directory using GPU.

## Pretrained Models

When you execute `scripts/sample.py`, it will automatically download the pretrained models from the [figshare](https://figshare.com/articles/dataset/Chemeleon-dng/29196176?file=54966305) repository and save them in the `ckpts/` directory (if not already present). The pretrained models were trained on `mp-20` and `alex_mp_20` datasets.

The framework includes pretrained checkpoints located in the `ckpts/` directory:

- `chemeleon_csp_alex_mp_20_v0.0.2.ckpt`
- `chemeleon_dng_alex_mp_20_v0.0.2.ckpt`
- `chemeleon_csp_mp_20_v0.0.2.ckpt`
- `chemeleon_dng_mp_20_v0.0.2.ckpt`

## Benchmarks

For benchmarking purposes, we provide 10,000 sampled structures for the `DNG` task trained on [`mp-20`](benchmarks/chemeleon_dng_mp_20_v0.0.2.json.gz) and [`alex_mp_20`](benchmarks/chemeleon_dng_alex_mp_20_v0.0.2.json.gz) datasets in the `benchmarks/` directory. The sampled structures are saved in CIF format and compressed JSON format.

## Citation

If you find our work helpful, please cite the following publication:

**"Exploration of crystal chemical space using text-guided generative artificial intelligence"** *Nature Communications* (2025)  
DOI: [10.1038/s41467-025-59636-y](https://doi.org/10.1038/s41467-025-59636-y)

```bibtex
@article{park2025exploration,
  title={Exploration of crystal chemical space using text-guided generative artificial intelligence},
  author={Park, Hyunsoo and Onwuli, Anthony and Walsh, Aron},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={1--14},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License, developed by [Hyunsoo Park](https://hspark1212.github.io) as part of the [Materials Design Group](https://github.com/wmd-group) at Imperial College London.  
See the [LICENSE file](https://github.com/hspark1212/chemeleon/blob/main/LICENSE) for more details.
