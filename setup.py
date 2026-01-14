"""Setup script for MedGemma Audio package."""

from setuptools import setup, find_packages

setup(
    name="medgemma-audio",
    version="0.1.0",
    description="Fine-tune MedGemma for audio understanding using masked autoencoders",
    author="",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "datasets>=2.18.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "full": [
            "unsloth>=2024.4",
            "bitsandbytes>=0.43.0",
            "tensorflow>=2.15.0",
            "wandb>=0.16.0",
            "nltk>=3.8.0",
            "pycocoevalcap>=1.2",
        ],
    },
)
