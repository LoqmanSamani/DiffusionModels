from setuptools import setup, find_packages

setup(
    name="torchdiff",
    version="1.0.0",
    description="A PyTorch-based library for diffusion models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Loghman Samani",
    author_email="samaniloqman91@gmail.com",
    url="https://github.com/LoqmanSamani/DiffusionModels",
    packages=find_packages(),
    install_requires=[
        "lpips>=0.1.4",
        "pytorch-fid>=0.3.0",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "tqdm>=4.67.1",
        "transformers>=4.44.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)