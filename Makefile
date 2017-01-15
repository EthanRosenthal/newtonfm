PYTHON ?= python
CYTHON ?= cython

# Compilation...

CYTHONSRC= $(wildcard newtonfm/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.cpp)

inplace:
	$(PYTHON) setup.py build_ext -i

all: cython inplace

cython: $(CSRC)

clean:
	rm -f newtonfm/*.c newtonfm/*.cpp newtonfm/*.html
	rm -f `find newtonfm -name "*.pyc"`
	rm -f `find newtonfm -name "*.so"`

%.cpp: %.pyx
	$(CYTHON) --cplus $<

# Tests...
#
# test-code: inplace
# 	$(NOSETESTS) -s newtonfm

# test-coverage:
#	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
#	--cover-package=newtonfm newtonfm
