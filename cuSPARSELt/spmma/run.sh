mkdir -p build
cd build
cmake .. -DWITH_INT8=ON -DWITH_CUP04=ON -DCUSPARSELT_DIR=../libcusparse_lt/
./spmma_example_static
./spmma_example
