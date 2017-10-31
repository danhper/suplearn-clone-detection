from setuptools import setup, find_packages


setup(
    name="suplearn_clone_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "bigcode-embeddings",
        "numpy",
        "keras",
        "pyyaml",
        "h5py",
    ]
)
