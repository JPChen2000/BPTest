cmake_minimum_required(VERSION 3.11)
project(bp_network)


include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/rcsc RCSC_SOURCES)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/utils UTILS_SOURCES)
message("source :" ${RCSC_SOURCES} ${UTILS_SOURCES})
add_executable(bp_network ${RCSC_SOURCES}
                          ${UTILS_SOURCES}
                          main.cpp)
