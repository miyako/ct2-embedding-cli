fn main() {
    // 1. Link C++ Standard Library
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-lc++");
        // Link OpenMP on Mac (Homebrew path usually)
        println!("cargo:rustc-link-search=lib");
        println!("cargo:rustc-link-lib=static=omp");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
    }
    // Windows connects MSVC OpenMP automatically usually.

    // 2. Compile our C++ Bridge
    cxx_build::bridge("src/main.rs")
        .file("src/cpp/embedding_bridge.cc")
        .flag_if_supported("-std=c++17")
        .include("include") 
        .include("src/cpp") 
        .compile("ct2-bridge");

    // 3. Link CTranslate2
    // ideally, you should download/compile ctranslate2 separately and set CTRANSLATE2_ROOT.
    // For this example, we assume libraries are in "lib" folder or installed globally.
    println!("cargo:rustc-link-search=native=lib");
    println!("cargo:rustc-link-lib=static=ctranslate2");
    
    // Dependencies of CTranslate2 (OpenBLAS/MKL/etc) need to be linked here.
    // Example for OpenBLAS:
    println!("cargo:rustc-link-lib=static=openblas");
    println!("cargo:rustc-link-lib=static=spdlog");
    // println!("cargo:rustc-link-lib=static=cpu_features");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/cpp/embedding_bridge.cc");
    println!("cargo:rerun-if-changed=src/cpp/embedding_bridge.h");
}
