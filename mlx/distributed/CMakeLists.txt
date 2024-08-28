target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
)

if (MLX_BUILD_CPU)
    if (MLX_CUSTOM_DISTRIBUTED)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sockets)
    elseif (MPI_FOUND)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mpi)
    else()
      target_sources(
        mlx
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/no_distributed.cpp
      )
    endif()
endif()
