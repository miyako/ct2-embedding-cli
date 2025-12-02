fn main() {

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-lc++");
        println!("cargo:rustc-link-search=lib");
        println!("cargo:rustc-link-lib=static=omp");
        println!("cargo:rustc-link-lib=static=ctranslate2"); 
        println!("cargo:rustc-link-lib=static=openblas");
        println!("cargo:rustc-link-lib=static=spdlog");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Accelerate");        
    } 
    
    if cfg!(target_env = "msvc") {
        println!("cargo:rustc-link-search=msvc");
        println!("cargo:rustc-link-lib=static=cpu_features"); 
        println!("cargo:rustc-link-lib=static=ctranslate2"); 
        println!("cargo:rustc-link-lib=static=utils"); 
    }

    cxx_build::bridge("src/main.rs")
        .file("src/cpp/embedding_bridge.cc")
        .flag_if_supported("-std=c++17")
        .include("include") 
        .include("src/cpp") 
        .compile("ct2-bridge");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/cpp/embedding_bridge.cc");
    println!("cargo:rerun-if-changed=src/cpp/embedding_bridge.h");
}
