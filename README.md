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

### Models

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

---

## Examples

```
ct2-embedding-cli -m /Users/miyako/Documents/GitHub/ct2-embedding-cli/models/small
   -t "If I am not for myself, who will be for me? And if I am only for myself, what am I? And if not now, when?"
{
  "text": "If I am not for myself, who will be for me? And if I am only for myself, what am I? And if not now, when?",
  "ids": [
    0,
    4263,
    87,
    444,
    959,
    100,
    35978,
    4,
    2750,
    1221,
    186,
    100,
    163,
    32,
    3493,
    2174,
    87,
    444,
    4734,
    100,
    35978,
    4,
    2367,
    444,
    87,
    32,
    3493,
    2174,
    959,
    5036,
    4,
    3229,
    32,
    2
  ],
  "embeddings": [
    0.05623773,
    0.32553634,
    0.17356923,
    0.29647267,
    0.35619104,
    0.077352,
    0.46401682,
    0.03654891,
    -0.16268198,
    -0.05699,
    0.27192265,
    -0.41391996,
    -0.070286945,
    -0.059670933,
    -0.12597905,
    0.0893452,
    0.08727617,
    -0.121742725,
    -0.2024223,
    0.13778014,
    0.03059977,
    0.070499785,
    -0.28640753,
    -0.09934541,
    0.12525691,
    -0.061339814,
    0.225425,
    0.27471495,
    -0.048570573,
    0.035768077,
    0.00097455503,
    -0.47201797,
    -0.023264064,
    0.18261191,
    0.27739003,
    0.1242388,
    0.087270655,
    -0.097840995,
    -0.0075118383,
    0.048323285,
    0.13033757,
    -0.14105853,
    -0.013920448,
    0.083004124,
    0.122369684,
    0.034445304,
    -0.13558093,
    0.001969011,
    -0.17339274,
    -0.045723602,
    0.2197878,
    0.11169797,
    -0.20440672,
    0.04368761,
    0.2281082,
    0.16644947,
    -0.067670636,
    0.0006452256,
    0.12112611,
    0.055080414,
    -0.022057246,
    -0.16782287,
    -0.15697795,
    0.20582314,
    0.055253956,
    0.15274592,
    0.07933995,
    0.039044984,
    -0.2377664,
    -0.08326227,
    0.31036896,
    0.22155274,
    -0.13162702,
    -0.054556794,
    0.10334597,
    -0.06276899,
    0.3677059,
    -0.042780288,
    0.30225047,
    0.21981056,
    -0.05382813,
    0.059674304,
    0.06760905,
    -0.22702336,
    -0.2270057,
    0.16021866,
    0.074576706,
    0.3635217,
    -0.08954256,
    -0.10264436,
    -0.523648,
    0.1812364,
    0.26125827,
    -0.03739145,
    0.292249,
    -0.047770966,
    0.039882522,
    -0.017542847,
    -0.10985823,
    0.28751114,
    -0.043315686,
    -0.31477344,
    -0.23531061,
    0.018925373,
    0.28825772,
    -0.07781274,
    -0.22251244,
    -0.091012284,
    -0.11636054,
    -0.23538654,
    -0.13237825,
    0.03442477,
    0.09467799,
    0.018459199,
    0.045857284,
    -0.3777568,
    0.030646062,
    0.40598956,
    -0.10182051,
    0.05288895,
    -0.0854194,
    0.11493546,
    0.009863842,
    -0.07295779,
    0.085251555,
    -0.24074395,
    0.020745084,
    0.006254593,
    -0.17417951,
    0.042399872,
    0.17959853,
    0.07618114,
    -0.32835093,
    -0.12913856,
    0.052808832,
    -0.09312059,
    -0.22218522,
    -0.0017446704,
    0.09687252,
    0.08980171,
    -0.27431202,
    0.6638357,
    -0.18519776,
    0.2650151,
    0.025322365,
    -0.36475775,
    0.35328066,
    0.13545674,
    -0.12197426,
    0.0046623508,
    -0.2136255,
    -0.2907609,
    -0.26568696,
    0.13651793,
    -0.07906788,
    -0.19289574,
    -0.30315486,
    -0.19453394,
    -0.087540105,
    -0.0435408,
    -0.016961025,
    0.063853264,
    -0.24145308,
    -0.0017822302,
    -0.021698914,
    0.13658081,
    -0.15184186,
    0.026558438,
    -0.018635405,
    0.028930232,
    0.40034863,
    -0.15231729,
    0.05744221,
    -0.119073756,
    0.16301298,
    0.1795661,
    -0.45850268,
    -0.013799785,
    -0.1584941,
    0.020647546,
    -0.15812063,
    0.08051419,
    -0.26125914,
    -0.38681668,
    -0.49105477,
    -0.1105848,
    -0.26652217,
    0.082904816,
    -0.14441366,
    -0.09835468,
    -0.02849165,
    0.15808952,
    -0.23979346,
    0.20121261,
    0.0048878565,
    -0.20819098,
    0.37128192,
    0.2056784,
    0.027127355,
    0.10230378,
    -0.22420006,
    0.09703667,
    0.5627573,
    -0.10258156,
    0.07113099,
    0.16297337,
    -0.1300498,
    0.12830421,
    0.10490283,
    0.0361912,
    -0.13596848,
    0.2387575,
    0.23168519,
    -0.11241614,
    0.11571648,
    -0.26724046,
    0.19735514,
    0.39664015,
    0.11907042,
    0.019061206,
    0.008521213,
    -0.12479124,
    -0.15436672,
    -0.103336856,
    0.24008468,
    0.014284574,
    0.06544657,
    -0.08701249,
    0.24279737,
    -0.37874642,
    0.5128616,
    -0.08267632,
    -0.23328047,
    0.028661357,
    -0.1269811,
    -0.38756233,
    0.7108594,
    0.06297616,
    -0.55351156,
    0.11743427,
    -0.2888317,
    -0.38684043,
    0.16567385,
    -0.07834213,
    -0.057090268,
    0.5098758,
    -0.3214789,
    -0.24287227,
    0.37057498,
    0.15475164,
    0.5239109,
    -0.1932068,
    0.16083848,
    0.10978786,
    -0.17958945,
    0.11431927,
    -0.43558338,
    -0.04590388,
    0.093369514,
    -0.059784587,
    -0.4063162,
    -0.23101316,
    0.085292645,
    -0.20707078,
    -0.1948159,
    0.082962625,
    0.04559434,
    0.031138014,
    0.03504769,
    -0.09761996,
    0.1943414,
    0.042259548,
    0.377627,
    0.24283656,
    -0.53213996,
    -0.34408736,
    0.32528856,
    0.18408868,
    0.21276544,
    -0.16105327,
    -0.16510966,
    -0.19387995,
    0.4767013,
    0.33583012,
    -0.17383347,
    0.119039126,
    0.3897591,
    0.49881974,
    -0.049543336,
    -0.1458955,
    -0.286502,
    0.38479587,
    0.04099434,
    0.14045899,
    0.008453883,
    -0.4798436,
    0.038791228,
    -0.047301933,
    -0.11868439,
    0.078101665,
    0.19099236,
    0.29207182,
    0.14053945,
    -0.021845726,
    0.4057107,
    0.120835304,
    -0.071291745,
    -0.021940367,
    -0.32932994,
    -0.061094735,
    0.0480097,
    0.07266048,
    0.08325707,
    0.2523495,
    -0.15318187,
    0.002241703,
    -0.47334218,
    -0.23652697,
    -0.12888469,
    -0.06261894,
    0.15758133,
    0.20325288,
    -0.47746092,
    0.20382531,
    -0.019693248,
    0.15806548,
    -0.006948217,
    -0.6706485,
    0.10287358,
    0.16254531,
    -0.019455144,
    0.01428679,
    0.32025093,
    -0.16574307,
    -0.19633074,
    0.1509595,
    -0.27205285,
    -0.16687614,
    -0.25410593,
    -0.04226725,
    0.3211888,
    -0.053161085,
    -0.28894684,
    0.05105385,
    -0.09273695,
    -0.20046759,
    -0.19887526,
    -0.5102331,
    -0.23082255,
    0.16498291,
    0.11135728,
    0.17800646,
    -0.39425963,
    -0.10167863,
    0.12136574,
    -0.058361508,
    -0.17772259,
    -0.07146793,
    -0.07379197,
    -0.08616698,
    0.22793005,
    0.24311543,
    -0.11802287,
    0.16068198,
    -0.10813414,
    -0.060063455,
    0.25800207,
    -0.26533344,
    -0.009577856,
    -0.40219426,
    0.102059856,
    0.14254461,
    0.31682634,
    -0.16671635,
    0.19845213,
    0.3807657,
    0.011370691,
    0.26937127,
    -0.085606486,
    0.016104154,
    0.40326038,
    0.03226514,
    -0.189299,
    -0.47890276
  ]
}
```
