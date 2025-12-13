# ct2-embedding-cli
CLI to generate embeddings

## Abstract

**CTranslate2** is an engine highly optimised for fast local inference, especially **quantised transformer-based models**. For a simple task like generating embeddings for the purpose of semantic database search, a chat-oriented LLM frameworks like llama.cpp might be an overkill. This CLI tool leverages `ct2` for such use cases.

```
Usage: ct2-embedding-cli [OPTIONS] --model <MODEL>

Options:
  -m, --model <MODEL>    
  -t, --text <TEXT>      
  -d, --device <DEVICE>  [default: cpu]
      --server           Run in HTTP server mode
      --port <PORT>      Port to listen on (only used in server mode) [default: 3000]
  -h, --help             Print help
  -V, --version          Print version
```
  
> [!NOTE]
> You can omit `--text` and use `stdIn` instead.

## Design

Rust is used to tokenise input text. C++ is used to encode tokens and mean pool embeddings. Rust is also used to run a simple HTTP server.

## Usage

Find a transformer based model on Hugging Face. 

|Scale|Dimesnions|Model|Size on Disk|
|-|-:|-|-:|
|Large|`1024`|[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)|`587.1 MB`|
|Medium|`768`|[sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)|`296.2 MB`| 
|Small|`384`|[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)|`135.4 MB`| 

> [!TIP]
> Converted model are available in releases.

You need to convert the model to `ct2` format. Optionally you might want to quantise the weights.

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

* nomic-ai/nomic-bert-2048=ERROR
* google/embeddinggemma300m=ERROR 
* all-MiniLM-L6-v2

```
mkdir ct2
cd ct2     
python3 -m venv venv 
source venv/bin/activate    
pip install torch
pip install ctranslate2 transformers torch
ct2-transformers-converter --model sentence-transformers/all-MiniLM-L6-v2 \
    --output_dir all-MiniLM-L6-v2_f16 \
    --quantization float16 \
    --force
```

The converter script generates `3` files. You need to download `2` more:

* tokenizer_config.json
* tokenizer.json

Now you can use the CLI.

---

## Developer Notes

### Apple Silicon

* Build Ctranslate2 with accelerate framework

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

> [!WARNING]
> brew static lib seems broken (ARM ABI includes `___emutls_get_address`); build OpenBLAS from source. 

```
# Clean clone/source directory
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
# Use explicit target for Apple Silicon and force use of the native Apple Clang compiler
# TARGET=ARMV8 works well for M-series chips.
make DYNAMIC_ARCH=0 TARGET=ARMV8 CC=clang FC=gfortran
```
