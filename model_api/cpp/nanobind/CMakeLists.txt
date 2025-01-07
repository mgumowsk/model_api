# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(DEV_MODULE Development.Module)

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


file(GLOB BINDINGS_SOURCES ./*.cpp)
file(GLOB BINDINGS_HEADERS ./*.hpp)

nanobind_add_module(nanobind_model_api NB_STATIC STABLE_ABI LTO ${BINDINGS_SOURCES} ${BINDINGS_HEADERS})

target_link_libraries(nanobind_model_api PRIVATE model_api)
#
#nanobind_add_stub(
    #nanobind_model_api_stub
    #MODULE nanobind_model_api
    #OUTPUT nanobind_model_api.pyi
    #PYTHON_PATH $<TARGET_FILE_DIR:nanobind_model_api>
    #DEPENDS nanobind_model_api
#)
#