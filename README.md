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

First select a model on Hugging Face. 

Here are some suggestions:

|Size|Dimesnions|Model|
|-|-:|-|
|Large|1024|[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)|
|Medium|768|[sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)|
|Small|384|[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)|

## Models

You need to first convert the model to `ct2` format. Optionally you might want to quantise the weights.

Setup `ctranslate2` in `venv`.

```
cd ~/
mkdir ctranslate2 
cd ctranslate2
python3 -m venv .venv
source .venv/bin/activate
pip install torch
pip install ctranslate2 transformers
```
For the small, medium, large models listed above:

```
mkdir small
ct2-transformers-converter \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --output_dir small \
  --force \
  --quantization int8
```

```
mkdir medium
ct2-transformers-converter \
  --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --output_dir medium \
  --force \
  --quantization int8
```

```
mkdir large
ct2-transformers-converter \
  --model intfloat/multilingual-e5-large \
  --output_dir large \
  --force \
  --quantization int8
```

## CLI

You can now convert a text to vectors like so

```
ct2-embedding-cli \
   -m /Users/miyako/Documents/GitHub/ct2-embedding-cli/models/large
   -t "If I am not for myself, who will be for me? And if I am only for myself, what am I? And if not now, when?"
Generated Embedding Vector (Size: 34816):
[0.9075532, 0.73017204, -0.96385884, -1.0773305, 0.56636286, -0.6996522, -0.99128413, 1.0444531, 2.2073429, -1.0438899]
```

```
ct2-embedding-cli \
   -m /Users/miyako/Documents/GitHub/ct2-embedding-cli/models/medium
   -t "If I am not for myself, who will be for me? And if I am only for myself, what am I? And if not now, when?"
Generated Embedding Vector (Size: 26112):
[0.13720347, -0.040625297, -0.0046865623, 0.069453046, -0.038324688, -0.047868043, 0.011250849, -0.18425152, 0.17645219, -0.009526617]
```

```
ct2-embedding-cli -m /Users/miyako/Documents/GitHub/ct2-embedding-cli/models/small
   -t "If I am not for myself, who will be for me? And if I am only for myself, what am I? And if not now, when?"
Generated Embedding Vector (Size: 384):
[0.016460087, -0.11727273, 0.025516136, 0.033093385, 0.02568352, 0.015095918, 0.0033053188, 0.009695337, -0.06317781, 0.080003306]
```
