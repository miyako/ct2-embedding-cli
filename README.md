# ct2-embedding-cli
CLI to generate embeddings

### Apple

* Build Ctranslate2 with optimisation

```
git clone https://github.com/OpenNMT/CTranslate2 --recursive
cd CTranslate2
cmake -S . -B build_arm \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_ACCELERATE=ON \
    -DWITH_MKL=OFF \
    -DENABLE_CPU_DISPATCH=OFF \
    -DWITH_OPENBLAS=ON \
    -DOPENMP_RUNTIME=COMP \
    -DOpenMP_CXX_INCLUDE_DIR="..." \
    -DOpenMP_CXX_INCLUDE_DIR="..." \
    -DOpenMP_libomp_LIBRARY="..." \
    -DOPENBLAS_INCLUDE_DIR="..." \
    -DOPENBLAS_LIBRARY="..."
cmake --build build_arm --config Release
cargo build --release --target aarch64-apple-darwin
``` 

remove `'-Xclang -fopenmp'` from other C++ flags.

### Intel

```
git clone https://github.com/OpenNMT/CTranslate2 --recursive
cd CTranslate2
cmake -S . -B build_amd \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_ACCELERATE=OFF \
    -DWITH_MKL=OFF \
    -DENABLE_CPU_DISPATCH=OFF \
    -DWITH_OPENBLAS=ON \
    -DOPENMP_RUNTIME=COMP \
    -DOpenMP_CXX_INCLUDE_DIR="..." \
    -DOpenMP_CXX_INCLUDE_DIR="..." \
    -DOpenMP_libomp_LIBRARY="..." \
    -DOPENBLAS_INCLUDE_DIR="..." \
    -DOPENBLAS_LIBRARY="..."
cmake --build build_amd --config Release
cargo build --release --target x86_64-apple-darwin
```

### Windows

```
git clone https://github.com/OpenNMT/CTranslate2 --recursive
cd CTranslate2
cmake -S . -B build -A x64 \
 -DBUILD_SHARED_LIBS=OFF \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
 -DWITH_DNNL=OFF \
 -DWITH_OPENMP=ON \
 -DWITH_MKL=OFF \
 -DENABLE_DNNL=OFF \
 -DDNNL_CPU_RUNTIME=OMP \
 -DWITH_CPU_DISPATCH=OFF \
 -DCTRANSLATE2_BUILD_SINGLE_ISA=OFF \
 -DOPENMP_RUNTIME=COMP
cmake --build build --config Release
set CXXFLAGS=/MT /EHsc
cargo build --release --target x86_64-pc-windows-msvc
```
