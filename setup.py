from setuptools import find_packages, setup

setup(
    name="rl-pcb-v2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26",
        "scipy>=1.12",
        "gymnasium>=0.29",
        "torch>=2.2",
        "networkx>=3.2",
        "pyyaml>=6.0",
    ],
)
