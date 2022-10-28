from setuptools import find_packages, setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='blitzml',
    packages=find_packages(include=['blitzml', 'blitzml.*']),
    version='0.3.0',
    description='A low-code library for machine learning pipelines',
    author='AI Team',
    license='MIT',
    install_requires=[
        'joblib>=1.2.0',
        'numpy>=1.23.4',
        'pandas>=1.5.1',
        'scikit-learn>=1.1.3'
        ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)


# Great resources
# https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
# https://realpython.com/pypi-publish-python-package/#publish-your-package-to-pypi
# https://packaging.python.org/en/latest/overview/