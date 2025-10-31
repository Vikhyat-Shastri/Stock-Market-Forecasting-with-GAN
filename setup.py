"""
Setup configuration for Stock Market Forecasting with GAN package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="stock-market-gan",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready GAN for stock market forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Stock-Market-Forecasting-with-GAN",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/Stock-Market-Forecasting-with-GAN/issues",
        "Documentation": "https://github.com/yourusername/Stock-Market-Forecasting-with-GAN/wiki",
        "Source Code": "https://github.com/yourusername/Stock-Market-Forecasting-with-GAN",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-gan-train=scripts.train:main",
            "stock-gan-predict=scripts.predict:main",
            "stock-gan-optimize=scripts.optimize:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
