from setuptools import setup, find_packages

setup(
    name='neural_network_base',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)