mkdir -p build
cd build
cmake .. -DWITH_INT8=ON -DWITH_CUP04=ON -DCUSPARSELT_DIR=/usr/lib/x86_64-linux-gnu/
./spmma_example 32 32 32
