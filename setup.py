#!/usr/bin/env python3
"""
Setup script for Intelligent Document Search System
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read README for long description
def read_readme():
    """Read README.md for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="intelligent-document-search",
    version="1.0.0",
    author="unnanu-inc",
    author_email="y21co021@rvrjc.ac.in",
    description="AI-powered document search system with RTX 4090 GPU optimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/indra7777/document-search-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0",
            "faiss-gpu>=1.7.4",
        ]
    },
    entry_points={
        "console_scripts": [
            "document-search=search_interface:main",
            "process-documents=document_processor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords="ai, nlp, document-search, llm, gpu, rtx-4090, rag, semantic-search",
)