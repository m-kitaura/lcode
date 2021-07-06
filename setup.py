import sys
import os
from setuptools import setup
from setuptools import find_packages

sys.path.append('./lcode')
sys.path.append('./tests')

def read_requirements():
    """ Parse requirements from requirements.txt. """
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='lcode',
    version='0.0.1',
    description='L-CODE: Labelless Concept Drift Detection and Explanation',
    author='M Kitaura',
    author_email='mail@example.com',
    url='https://github.com/m-kitaura/lcode',
    python_requires=">=3.8.1",
    install_requires=read_requirements(),
    license='',
    package_dir={'': 'lcode'},
    packages=find_packages(where=('lcode')),
    test_suite='tests',
)
