# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(DEV_MODULE Development.Module)

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

find_package(pybind11 REQUIRED)

file(GLOB BINDINGS_SOURCES ./*.cpp)
file(GLOB BINDINGS_HEADERS ./*.hpp)

pybind11_add_module(pybind11_model_api ${BINDINGS_SOURCES} ${BINDINGS_HEADERS})

target_link_libraries(pybind11_model_api PRIVATE model_api)
