from setuptools import setup

setup(
    name="chemeleon-dng",
    py_modules=["chemeleon_dng"],
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
