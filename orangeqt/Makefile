ORANGEQT_BUILD_DIR=build
ifndef OLD
  OLD=..
endif
ifndef PYTHON
  PYTHON=$(shell which python)
endif

PYTHON_VERSION = $(shell $(PYTHON) -c 'import sys; print "%s.%s" % sys.version_info[:2]')
OS = $(shell uname)

all:
	mkdir -p $(ORANGEQT_BUILD_DIR)
	cd $(ORANGEQT_BUILD_DIR); cmake -DCMAKE_BUILD_TYPE=Release -DORANGE_LIB_DIR=$(abspath $(OLD)) -DPYTHON_EXECUTABLE=$(PYTHON) -DCMAKE_USE_PYTHON_VERSION=$(PYTHON_VERSION) $(EXTRA_ORANGEQT_CMAKE_ARGS) ..
	if ! $(MAKE) $@ -C $(ORANGEQT_BUILD_DIR); then exit 1; fi;
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orangeqt.so $(OLD)/orangeqt.so
endif
	

cleantemp:
	rm -rf $(ORANGEQT_BUILD_DIR)

clean: cleantemp
	rm -f $(OLD)/orangeqt.so
