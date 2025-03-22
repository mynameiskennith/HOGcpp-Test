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
add_library(HOGcpp_lib INTERFACE)
target_include_directories(HOGcpp_lib INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Add executable targets
add_executable(train_app src/train.cpp)
add_executable(detect_app src/detect_screws.cpp)

# Link OpenCV and LIBSVM libraries
target_link_libraries(train_app ${OpenCV_LIBS} /home/minato/Dev/libsvm-3.35/libsvm.so HOGcpp_lib)
target_link_libraries(detect_app ${OpenCV_LIBS} /home/minato/Dev/libsvm-3.35/libsvm.so HOGcpp_lib)

# Enable all compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(train_app PRIVATE -Wall -Wextra)
    target_compile_options(detect_app PRIVATE -Wall -Wextra)
endif()