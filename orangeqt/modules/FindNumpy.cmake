# This file is copied from Avogadro
# Home page at http://avogadro.openmolecules.net
# Code repository at https://github.com/cryos/avogadro


# - Find numpy
# Find the native numpy includes
# This module defines
#  NUMPY_INCLUDE_DIR, where to find numpy/arrayobject.h, etc.
#  NUMPY_FOUND, If false, do not try to use numpy headers.

#if (NUMPY_INCLUDE_DIR)
  # in cache already
#  set (NUMPY_FIND_QUIETLY TRUE)
#endif (NUMPY_INCLUDE_DIR)

EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c 
    "import numpy; print numpy.get_include()"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)


if (NUMPY_INCLUDE_DIR)
  if(EXISTS ${NUMPY_INCLUDE_DIR}/numpy/arrayobject.h)
    # successful
    set (NUMPY_FOUND TRUE)
    set (NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR} CACHE STRING "Numpy include path")
  else()
    set(NUMPY_FOUND FALSE)
  endif()
else (NUMPY_INCLUDE_DIR)
  # Did not successfully include numpy
  set(NUMPY_FOUND FALSE)
endif (NUMPY_INCLUDE_DIR)

if (NUMPY_FOUND)
  if (NOT NUMPY_FIND_QUIETLY)
    message (STATUS "Numpy headers found")
  endif (NOT NUMPY_FIND_QUIETLY)
else (NUMPY_FOUND)
  if (NUMPY_FIND_REQUIRED)
    message (FATAL_ERROR "Numpy headers missing")
  endif (NUMPY_FIND_REQUIRED)
endif (NUMPY_FOUND)

MARK_AS_ADVANCED (NUMPY_INCLUDE_DIR)
