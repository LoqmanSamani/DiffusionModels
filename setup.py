from setuptools import setup, find_packages

setup(
    name="TorchDiff",
    version="2.0.0",
    description="A PyTorch-based library for diffusion models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Loghman Samani",
    author_email="samaniloqman91@gmail.com",
    url="https://github.com/LoqmanSamani/TorchDiff",
    project_urls={
        "Homepage": "https://loqmansamani.github.io/torchdiff",
        "Documentation": "https://torchdiff.readthedocs.io",
        "Source": "https://github.com/LoqmanSamani/TorchDiff",
    },
    packages=find_packages(),
    install_requires=[
        "lpips>=0.1.4",
        "pytorch-fid>=0.3.0",
        "torch>=2.3.0,<3.0.0",  # Adjusted to a realistic version range
        "torchvision>=0.18.0,<0.19.0",  # Aligned with torch version
        "tqdm>=4.67.1",
        "transformers>=4.44.2",
    ],
    extras_require={
        "test": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    include_package_data=True,
    package_data={
        "torchdiff": ["LICENSE", "data/*.txt", "models/*.pt"],
    },
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
