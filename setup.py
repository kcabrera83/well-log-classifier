from setuptools import setup, find_packages

setup(
    name="well-log-classifier",
    version="1.0.0",
    author="Ing. Kelvin Cabrera",
    description="ML-based lithology classification and porosity estimation from well logs",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "flask>=2.3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
)
