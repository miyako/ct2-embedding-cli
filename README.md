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
    -DENABLE_CPU_DISPATCH=ON \
    -DWITH_OPENBLAS=ON \
    -DOPENMP_RUNTIME=COMP \
    -DOPENMP_C_INCLUDE_DIR="..." \
    -DOPENMP_CXX_INCLUDE_DIR="..." \
    -DOPENMP_LIBOMP_LIBRARY="..." \
    -DOPENBLAS_INCLUDE_DIR="..."
``` 
