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
    
    let mut bridge = cxx_build::bridge("src/main.rs");
    
    // --- START: Add the /MT flag for static runtime linking ---
    if cfg!(target_env = "msvc") {
        // Use the /MT flag for Release builds (Multi-threaded Static)
        // or /MTd for Debug builds (Multi-threaded Debug Static).
        // Since cargo build defaults to Release, we use /MT.
        // If you need debug builds, you'd check for target_cfg = "debug"
        // and set the flag to /MTd.
        bridge.flag("/MT");
    }

    bridge.file("src/cpp/embedding_bridge.cc")
        .flag_if_supported("-std=c++17")
        .include("include") 
        .include("src/cpp") 
        .compile("ct2-bridge");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/cpp/embedding_bridge.cc");
    println!("cargo:rerun-if-changed=src/cpp/embedding_bridge.h");
}
