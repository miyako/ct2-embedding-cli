/*
#pragma once
#include "rust/cxx.h"
#include <memory>
#include <vector>
#include <string>

// Forward declaration
namespace ctranslate2 {
    class Encoder;
}

class EmbeddingModel {
public:
    EmbeddingModel(rust::Str model_path, rust::Str device);
    ~EmbeddingModel();

    // Use rust::Vec instead of std::vector for the interface
    rust::Vec<float> encode(const rust::Vec<uint32_t>& input_ids, const rust::Vec<size_t>& lengths) const;

private:
    std::shared_ptr<ctranslate2::Encoder> encoder_;
};

// Factory function using rust::Str
std::unique_ptr<EmbeddingModel> new_embedding_model(rust::Str model_path, rust::Str device);
*/
#pragma once
#include "rust/cxx.h"
#include <memory>
#include <vector>
#include <string>
#include <ctranslate2/devices.h> // Include for ctranslate2::Device

// Forward declaration
namespace ctranslate2 {
    class Encoder;
}

class EmbeddingModel {
public:
    // Constructor uses rust::Str, implementation converts to std::string
    EmbeddingModel(rust::Str model_path, rust::Str device);
    ~EmbeddingModel();

    // The FFI signature remains the same as expected by Rust
    rust::Vec<float> encode(const rust::Vec<uint32_t>& input_ids, const rust::Vec<size_t>& lengths) const;

private:
    std::shared_ptr<ctranslate2::Encoder> encoder_;
    // Store the device, as encoder does not expose a public device() method
    ctranslate2::Device device_;
};

// Factory function
std::unique_ptr<EmbeddingModel> new_embedding_model(rust::Str model_path, rust::Str device);
