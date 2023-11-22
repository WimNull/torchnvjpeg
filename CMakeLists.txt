cmake_minimum_required(VERSION 3.4)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(torchnvjpeg LANGUAGES CXX CUDA)
set(CUDA 10.1 REQUIRED)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


EXECUTE_PROCESS(COMMAND which python OUTPUT_VARIABLE PythonDir)
message(STATUS "Python path: ${PythonDir}")
get_filename_component(PythonDir "${PythonDir}" DIRECTORY)
set(CMAKE_PREFIX_PATH ${PythonDir})
find_package(PythonLibs)
message(STATUS "PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS}, PYTHON_LIBRARIES ${PYTHON_LIBRARIES} ${PYTHON_BINARY}")
set(CMAKE_PREFIX_PATH "${PythonDir}/lib")
link_directories("${PythonDir}/lib")

EXECUTE_PROCESS(COMMAND python -c "import site; print(site.getsitepackages()[0], end='')" OUTPUT_VARIABLE PythonSiteDir)
# libtorch
set(CMAKE_PREFIX_PATH "${PythonSiteDir}/torch/share/cmake")

find_package(Torch REQUIRED)
if (NOT TORCH_FOUND)
    message(FATAL_ERROR "Torch not found.")
else ()
    message(STATUS "Find Torch VERSION ${Torch_VERSION},
    TORCH_CXX_FLAGS ${TORCH_CXX_FLAGS}
    TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS},
    TORCH_LIBRARIES ${TORCH_LIBRARIES}, ")
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${PYTHON_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
link_directories("/usr/local/cuda/lib64")

add_executable(torchnvjpeg csrc/torchnvjpeg.cpp)


#find_package(OpenMP)
#if (OpenMP_CXX_FOUND)
#    target_compile_definitions(torchnvjpeg PUBLIC -DOPENMP)
#    target_link_libraries(torchnvjpeg PUBLIC OpenMP::OpenMP_CXX)
#else ()
#    message("WARNING" "OpenMP library could not be found. ")
#endif ()

target_link_libraries(torchnvjpeg PUBLIC nvjpeg ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
#target_compile_definitions(torchnvjpeg PUBLIC -DPYBIND)
