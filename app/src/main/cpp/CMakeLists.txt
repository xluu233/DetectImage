# 设定所要求能运行的最低版本
cmake_minimum_required(VERSION 3.10.2)

#必须配置
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -static-openmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -static-openmp")

include_directories(#将头文件所在目录告诉编译器，在里面添加
        ${CMAKE_SOURCE_DIR}/../cpp/include
        )


add_library( # Sets the name of the library.
        detect-opencv
        SHARED
        main.cpp
        main.h

        detect/PageDet.cpp
        detect/PageDet.h
        detect/tranforms.cpp

        )

#mnn
include_directories(
        ${CMAKE_SOURCE_DIR}/../cpp/include/mnn
        ${CMAKE_SOURCE_DIR}/../cpp/include/mnn/expr
        ${CMAKE_SOURCE_DIR}/../cpp/include/mnn/plugin)

add_library(libMNN STATIC IMPORTED)
add_library(libMNN_CL STATIC IMPORTED)
add_library(libMNN_Express STATIC IMPORTED)
set_target_properties(
        libMNN
        libMNN_CL
        libMNN_Express
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN_CL.so
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN_Express.so
)

#opencv
include_directories(
        ${CMAKE_SOURCE_DIR}/../cpp/include/opencv2)

add_library(lib_opencv STATIC IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java4.so)

find_library( # Sets the name of the path variable.
        log-lib
        log)


target_link_libraries( # Specifies the target library.
        detect-opencv

        lib_opencv
        libMNN
        libMNN_CL
        libMNN_Express

        android
        jnigraphics
        ${log-lib})
