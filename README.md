# zadeh
[![Github release](https://img.shields.io/github/release/dih5/zadeh.svg)](https://github.com/dih5/zadeh/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/zadeh.svg)](https://pypi.python.org/pypi/zadeh)

[![license MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/Dih5/zadeh/master/LICENSE.txt)

[![Build Status](https://travis-ci.org/Dih5/zadeh.svg?branch=master)](https://travis-ci.org/Dih5/zadeh)
[![Documentation Status](https://readthedocs.org/projects/zadeh/badge/?version=latest)](http://zadeh.readthedocs.io/en/latest/?badge=latest)

Python package to build fuzzy inference systems


## Installation
Assuming you have a [Python3](https://www.python.org/) distribution with [pip](https://pip.pypa.io/en/stable/installing/), the latest pypi release can be installed with:
```
pip3 install zadeh
```
To install the recommended optional dependencies you can run
```
pip3 install 'zadeh[extras]'
```
Mind the quotes.

## Developer information
### Development installation

To install a development version, clone the repo, cd to the directory with this file and:

```
pip3 install -e '.[test]'
```
Consider using a virtualenv if needed:
```
# Prepare a clean virtualenv and activate it
virtualenv venv
source venv/bin/activate
# Install the package
pip3 install -e '.[test]'
```

### Documentation

To generate the documentation, the *docs* extra dependencies must be installed.

To generate an html documentation with sphinx run:
```
make docs
```

To generate a PDF documentation using LaTeX:
```
make pdf
```



### Test
To run the unitary tests:
```
make test
```
