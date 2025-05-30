from setuptools import setup

setup(
    name="chemeleon-rl",
    py_modules=["chemeleon_rl"],
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2.1.0",
        "sacred",
        "ase",
        "torch-geometric",
        "torchmetrics",
        "ase",
        "tqdm",
        "wandb",
        "pydantic",
        "jupyterlab",
        "pymatgen",
        "fire",
    ],
)
