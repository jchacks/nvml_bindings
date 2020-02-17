from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nvml',
    version='0.1',
    description='Object Orientated NVML bindings in python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jchacks/nvml_bindings',
    author='jchacks',
    packages=['nvml'],
    install_requires=[
        'numpy',
        'py3nvml'
    ],
    include_package_data=True,
    zip_safe=False
)
