# zadeh
[![Github release](https://img.shields.io/github/release/dih5/zadeh.svg)](https://github.com/dih5/zadeh/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/zadeh.svg)](https://pypi.python.org/pypi/zadeh)

[![license MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/Dih5/zadeh/master/LICENSE.txt)

[![Build Status](https://travis-ci.org/Dih5/zadeh.svg?branch=master)](https://travis-ci.org/Dih5/zadeh)
[![Documentation Status](https://readthedocs.org/projects/zadeh/badge/?version=latest)](http://zadeh.readthedocs.io/en/latest/?badge=latest)

Python package to build fuzzy inference systems


## Installation
Assuming you have a [Python3](https://www.python.org/) distribution with [pip](https://pip.pypa.io/en/stable/installing/), to install a development version, cd to the directory with this file and:

```
pip3 install -e .
```
As an alternative, a virtualenv might be used to install the package:
```
# Prepare a clean virtualenv and activate it
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
# Install the package
pip3 install -e .
```

To install also the dependencies to run the tests or to generate the documentation install some of the extras like
```
pip3 install -e '.[docs,test]'
```
Mind the quotes.

## Documentation
To generate the documentation, the *docs* extra dependencies must be installed.

To generate an html documentation with sphinx run:
```
make docs
```

To generate a PDF documentation using LaTeX:
```
make pdf
```



## Test
To run the unitary tests:
```
make test
```
