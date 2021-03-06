#!/usr/bin/env python

"""
The setup script for pip. Allows for `pip install -e .` installation.
"""

from setuptools import setup, find_packages

requirements = ['numpy', 'matplotlib', 'torch', 'PyYAML',
            'scikit-learn', 'pandas', 'cycler', 'tables']
setup_requirements = []
tests_requirements = ['pytest']

setup(
    author='L. Cheng',
    author_email='lionel.cheng@hotmail.fr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8'
    ],
    description='proteinclass: ',
    install_requires=requirements,
    license='GNU General Public License v3',
    long_description='\n\n',
    include_package_data=True,
    keywords='biology protein calssifier',
    name='proteinclass',
    packages=find_packages(include=['proteinclass']),
    setup_requires=setup_requirements,

    test_suite='tests',
    tests_require=tests_requirements,
    version='0.1',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'train_network=proteinclass.train:main',
            'train_networks=proteinclass.multiple_train:main',
            'plot_metrics=proteinclass.pproc:main',
            'predict=proteinclass.predict:main'
        ],
    },
)
