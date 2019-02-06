# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fastfeatures',
    version='0.0.1',
    description='FastFeatures',
    long_description=readme,
    author='Felix Neutatz',
    author_email='neutatz@gmail.com',
    url='https://github.com/BigDaMa/FastFeatures',
    license=license,
    package_data={'config': ['fastsklearnfeature/configuration/resources']},
    include_package_data=True,
    packages=find_packages(exclude=('tests', 'docs'))
)

