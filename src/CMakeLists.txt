add_executable(main
    main.cu
    vdbExporter.cpp
)

set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)

target_link_libraries(main PRIVATE
    OpenVDB::openvdb
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenVDB_INCLUDE_DIRS}
)