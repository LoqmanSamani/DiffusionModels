from setuptools import setup, find_packages
from pathlib import Path

readme_files = ["README_PYPI.md", "README.md"]
long_description = ""

for readme_file in readme_files:
    readme_path = Path(readme_file)
    if readme_path.exists():
        long_description = readme_path.read_text(encoding="utf-8")
        break

if not long_description:
    long_description = "A PyTorch-based library for diffusion models"

#long_description = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""

setup(
    name="TorchDiff",
    version="2.1.0",
    description="A PyTorch-based library for diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Loghman Samani",
    author_email="samaniloqman91@gmail.com",
    url="https://github.com/LoqmanSamani/TorchDiff",
    project_urls={
        "Homepage": "https://loqmansamani.github.io/torchdiff",
        "Documentation": "https://torchdiff.readthedio",
        "Source": "https://github.com/LoqmanSamani/TorchDiff",
    },
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "lpips>=0.1.4",
        "pytorch-fid>=0.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.60.0",
        "transformers>=4.20.0",
        "torchmetrics>=1.0.0"
    ],
    extras_require={
        "test": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
        "dev": ["black", "flake8", "mypy"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords=["diffusion models", "pytorch", "machine learning", "deep learning"],
)
