find_path(
  GLOO_INCLUDE_DIR gloo/allreduce.h
  PATHS ${GLOO_INC_DIR}
  PATH_SUFFIXES include)

find_library(
  GLOO_LIBRARY gloo
  PATHS ${GLOO_LIB_DIR}
  PATH_SUFFIXES lib
  HINTS GLOO)

find_library(
  UV_LIBRARY uv
  PATHS ${UV_LIB_DIR}
  PATH_SUFFIXES lib
  HINTS UV)

message(STATUS "GLOO LIB <${GLOO_LIBRARY}>")
message(STATUS "GLOO INC <${GLOO_INCLUDE_DIR}>")
message(STATUS "UV   LIB <${UV_LIB_DIR}>")

target_sources(mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gloo.cpp)
target_link_libraries(mlx PUBLIC ${GLOO_LIBRARY})
target_link_libraries(mlx PUBLIC ${UV_LIBRARY})
target_include_directories(mlx PRIVATE ${GLOO_INCLUDE_DIR})
