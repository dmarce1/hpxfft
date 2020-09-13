# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)

# Libraries linked via full path no longer produce linker search paths.
cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0074 NEW)

project(hpxfft CXX C)

# Search path for CMake modules to be loaded by include() and find_package()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(Boost REQUIRED)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "-g  -ffast-math -march=native")

set(hpxfft_source_files
	src/fft.cu
	src/fft.cpp
	src/fft3d.cpp
)

set(hpxfft_header_files
	hpxfft/fft.hpp
	hpxfft/fft3d.hpp
)


add_hpx_executable(
  hpxfft
  DEPENDENCIES
    Silo::silo Boost::boost
  SOURCES
    ${hpxfft_source_files}
  HEADERS
    ${hpxfft_header_files}
 )

target_link_libraries (hpxfft PUBLIC boost_program_options tcmalloc siloh5 hwloc dl z hdf5)


target_include_directories(hpxfft PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(hpxfft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --gpu-architecture=compute_75;
						-Xptxas -O3 --use_fast_math ;
						-v
                        >)

