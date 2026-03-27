from setuptools import setup, find_packages

setup(
    name="pyciphod",
    version="0.1",
    author="Charles Assaad, Federico Baldo, Simon Ferreira, Timothée Loranchet",
    author_email="charles.assaad@inserm.fr",
    description="A Python package for causal discovery, causal inference, and root cause analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "networkx>0",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],
)