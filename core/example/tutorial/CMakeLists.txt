add_subdirectory(01_hello_world)
add_subdirectory(02_tuple)
add_subdirectory(03_struct_of_arrays)

if(Cabana_ENABLE_Serial OR Cabana_ENABLE_OpenMP OR Cabana_ENABLE_Pthread)
  add_subdirectory(04_aosoa)
  add_subdirectory(05_slice)
  add_subdirectory(07_sorting)
  add_subdirectory(08_linked_cell_list)
  add_subdirectory(09_verlet_list)
endif()

if(Cabana_ENABLE_Cuda AND (Cabana_ENABLE_Serial OR Cabana_ENABLE_OpenMP OR Cabana_ENABLE_Pthread))
  add_subdirectory(06_deep_copy)
endif()

if(Cabana_ENABLE_OpenMP)
  add_subdirectory(10_advanced_slice_openmp)
endif()

if(Cabana_ENABLE_Cuda)
  add_subdirectory(10_advanced_slice_cuda)
endif()