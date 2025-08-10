from setuptools import setup, find_packages

setup(
    name="midi_generator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "matplotlib>=3.7.0",
        "pretty-midi>=0.2.10",
        "pydantic>=1.8.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.60.0",
        "scipy>=1.7.0"
        "setuptools>=80.9.0",
    ],
    python_requires=">=3.9,<3.13",
)
