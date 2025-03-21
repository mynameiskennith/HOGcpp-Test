cmake_minimum_required(VERSION 3.10)
project(HOGcpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include LIBSVM headers
include_directories(/home/minato/Dev/libsvm-3.35)

# Include project headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Print OpenCV version and libraries (for debugging)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Copy feature_extraction.h to the build directory (if needed)
configure_file(
    ${PROJECT_SOURCE_DIR}/include/feature_extraction.h
    ${PROJECT_BINARY_DIR}/include/feature_extraction.h
    COPYONLY
)

# Define the HOGcpp library (as an interface library since it's header-only)
add_library(HOGcpp INTERFACE)
target_include_directories(HOGcpp INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Add executable targets
add_executable(${PROJECT_NAME} src/train.cpp)
add_executable(detect_screws src/detect_screws.cpp)

# Link OpenCV and LIBSVM libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} /home/minato/Dev/libsvm-3.35/libsvm.so HOGcpp)
target_link_libraries(detect_screws ${OpenCV_LIBS} /home/minato/Dev/libsvm-3.35/libsvm.so HOGcpp)

# Enable all compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
    target_compile_options(detect_screws PRIVATE -Wall -Wextra)
endif()