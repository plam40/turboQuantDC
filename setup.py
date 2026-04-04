from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

_base = [
    "torch>=2.0.0",
    "scipy>=1.10.0,<1.15.0",
]

_hf = [
    "transformers>=4.40.0",
    "accelerate>=0.25.0",
]

_bnb = [
    "bitsandbytes>=0.43.0",
]

_triton = [
    "triton>=3.0.0",
]

_benchmark = [
    "datasets",
    "matplotlib",
    "tqdm",
]

_dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

setup(
    name="turboquantdc",
    version="0.2.0",
    author="TurboQuantDC Contributors",
    description="TurboQuant: 3-bit KV cache compression for LLMs with <0.5% attention quality loss",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/turboquantdc/turboquantdc",
    packages=find_packages(exclude=["tests*", "benchmarks*", "reference*", "docs*", "warroom*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    python_requires=">=3.10",
    install_requires=_base,
    extras_require={
        "base": _base,
        "hf": _hf,
        "bnb": _bnb,
        "triton": _triton,
        "benchmark": _benchmark,
        "all": _base + _hf + _bnb + _triton + _benchmark,
        "dev": _dev,
    },
    keywords=[
        "llm", "kv-cache", "quantization", "compression",
        "transformer", "attention", "cuda", "pytorch",
    ],
)
