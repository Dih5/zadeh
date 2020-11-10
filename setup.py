#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script"""

from setuptools import setup, find_packages

setup(author="Dih5",
      author_email='dihedralfive@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
           # 'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      description='Python package to build fuzzy inference systems',
      extras_require={
        "docs": ["nbsphinx", "sphinx-rtd-theme", "IPython"],
        "test": ["pytest"],
      },
      keywords=[],
      name='zadeh',
      packages=find_packages(include=['zadeh'], exclude=["demos", "tests", "docs"]),
      install_requires=[],
      url='https://github.com/dih5/zadeh',
      version='0.1.0',

      )
