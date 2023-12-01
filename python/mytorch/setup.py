from setuptools import setup, find_packages

setup(
    name='mytorch',
    version='1.0.0',
    packages=find_packages(),
    extras_require={
        'with_cupy': ['cupy>=11.0.0'],
        'with_numpy': ['numpy>=1.20.0'],
    },
)
