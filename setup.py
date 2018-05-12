from setuptools import setup, find_packages


setup(
    name="suplearn_clone_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "keras",
        "pyyaml",
        "h5py",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "sqlalchemy",
    ]
)
