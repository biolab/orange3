# Find Python
# ~~~~~~~~~~~
# Find the Python interpreter and related Python directories.
#
# This file defines the following variables:
#
# PYTHON_EXECUTABLE - The path and filename of the Python interpreter.
#
# PYTHON_SHORT_VERSION - The version of the Python interpreter found,
#     excluding the patch version number. (e.g. 2.5 and not 2.5.1))
#
# PYTHON_LONG_VERSION - The version of the Python interpreter found as a human
#     readable string.
#
# PYTHON_SITE_PACKAGES_INSTALL_DIR - this cache variable can be used for installing 
#                              own python modules. You may want to adjust this to be the
#                              same as ${PYTHON_SITE_PACKAGES_DIR}, but then admin
#                              privileges may be required for installation.
#
# PYTHON_SITE_PACKAGES_DIR - Location of the Python site-packages directory.
#
# PYTHON_INCLUDE_PATH - Directory holding the python.h include file.
#
# PYTHON_LIBRARY, PYTHON_LIBRARIES- Location of the Python library.

# Copyright (c) 2007, Simon Edwards <simon@simonzone.com>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.



#include(CMakeFindFrameworks)

if(EXISTS PYTHON_LIBRARY)
  # Already in cache, be silent
  set(PYTHONLIBRARY_FOUND TRUE)
else(EXISTS PYTHON_LIBRARY)

  find_package(PythonInterp)
  find_package(PythonLibs)
  
  if(PYTHONLIBS_FOUND)
  	set(PYTHONLIBRARY_FOUND TRUE)
  endif(PYTHONLIBS_FOUND)
  
  if(PYTHONINTERP_FOUND)

    # get the directory of the current file, used later on in the file
    get_filename_component( _py_cmake_module_dir  ${CMAKE_CURRENT_LIST_FILE} PATH)
    if(NOT EXISTS "${_py_cmake_module_dir}/FindLibPython.py")
      message(FATAL_ERROR "The file FindLibPython.py does not exist in ${_py_cmake_module_dir} (the directory where FindPythonLibrary.cmake is located). Check your installation.")
    endif(NOT EXISTS "${_py_cmake_module_dir}/FindLibPython.py")

    execute_process(COMMAND ${PYTHON_EXECUTABLE}  "${_py_cmake_module_dir}/FindLibPython.py" OUTPUT_VARIABLE python_config)
    if(python_config)
      string(REGEX REPLACE ".*exec_prefix:([^\n]+).*$" "\\1" PYTHON_PREFIX ${python_config})
      string(REGEX REPLACE ".*\nshort_version:([^\n]+).*$" "\\1" PYTHON_SHORT_VERSION ${python_config})
      string(REGEX REPLACE ".*\nlong_version:([^\n]+).*$" "\\1" PYTHON_LONG_VERSION ${python_config})

      string(REGEX REPLACE ".*\npy_inc_dir:([^\n]+).*$" "\\1" _TMP_PYTHON_INCLUDE_PATH ${python_config})
      string(REGEX REPLACE ".*\nsite_packages_dir:([^\n]+).*$" "\\1" _TMP_PYTHON_SITE_PACKAGES_DIR ${python_config})

      # Put these two variables in the cache so they are visible for the user, but read-only:
      set(PYTHON_INCLUDE_PATH "${_TMP_PYTHON_INCLUDE_PATH}" CACHE PATH "The python include directory" FORCE)
      set(PYTHON_SITE_PACKAGES_DIR "${_TMP_PYTHON_SITE_PACKAGES_DIR}" CACHE PATH "The python site packages dir" FORCE)

      # This one is intended to be used and changed by the user for installing own modules:
      set(PYTHON_SITE_PACKAGES_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/python${PYTHON_SHORT_VERSION}/site-packages CACHE PATH "The directory where python modules will be installed to.")

      string(REGEX REPLACE "([0-9]+).([0-9]+)" "\\1\\2" PYTHON_SHORT_VERSION_NO_DOT ${PYTHON_SHORT_VERSION})
#      set(PYTHON_LIBRARY_NAMES python${PYTHON_SHORT_VERSION} python${PYTHON_SHORT_VERSION_NO_DOT})
      if(WIN32)
          string(REPLACE "\\" "/" PYTHON_SITE_PACKAGES_DIR ${PYTHON_SITE_PACKAGES_DIR})
      endif(WIN32)
#      find_library(PYTHON_LIBRARY NAMES ${PYTHON_LIBRARY_NAMES} PATHS ${PYTHON_PREFIX}/lib ${PYTHON_PREFIX}/libs NO_DEFAULT_PATH)
#      set(PYTHONLIBRARY_FOUND TRUE)
    endif(python_config)

    # adapted from cmake's builtin FindPythonLibs
#    if(APPLE)
#      cmake_find_frameworks(Python)
#      set(PYTHON_FRAMEWORK_INCLUDES)
#      if(Python_FRAMEWORKS)
#        # If a framework has been selected for the include path,
#        # make sure "-framework" is used to link it.
#        if("${PYTHON_INCLUDE_PATH}" MATCHES "Python\\.framework")
#          set(PYTHON_LIBRARY "")
#          set(PYTHON_DEBUG_LIBRARY "")
#        endif("${PYTHON_INCLUDE_PATH}" MATCHES "Python\\.framework")
#        if(NOT PYTHON_LIBRARY)
#          set (PYTHON_LIBRARY "-framework Python" CACHE FILEPATH "Python Framework" FORCE)
#        endif(NOT PYTHON_LIBRARY)
#        set(PYTHONLIBRARY_FOUND TRUE)
#      endif(Python_FRAMEWORKS)
#    endif(APPLE)
  endif(PYTHONINTERP_FOUND)

  if(PYTHONLIBRARY_FOUND)
#    set(PYTHON_LIBRARIES ${PYTHON_LIBRARY})
    if(NOT PYTHONLIBRARY_FIND_QUIETLY)
      message(STATUS "Found Python executable: ${PYTHON_EXECUTABLE}")
      message(STATUS "Found Python version: ${PYTHON_LONG_VERSION}")
      message(STATUS "Found Python library: ${PYTHON_LIBRARIES}")
    endif(NOT PYTHONLIBRARY_FIND_QUIETLY)
  else(PYTHONLIBRARY_FOUND)
    if(PYTHONLIBRARY_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find Python")
    endif(PYTHONLIBRARY_FIND_REQUIRED)
  endif(PYTHONLIBRARY_FOUND)

endif (EXISTS PYTHON_LIBRARY)
