from setuptools import find_packages, setup

setup(
    name="clickrank",
    version="0.1.0",
    description="CTR prediction benchmark for ad ranking models",
    author="Sanjith Shanmugavel",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ],
)
