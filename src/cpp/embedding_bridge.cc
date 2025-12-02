/*
#include "embedding_bridge.h"
#include <ctranslate2/encoder.h>
#include <ctranslate2/storage_view.h>
#include <iostream>
#include <algorithm> // for std::copy

EmbeddingModel::EmbeddingModel(rust::Str model_path, rust::Str device) {
    // Convert rust::Str to std::string
    std::string path_str(model_path.data(), model_path.length());
    std::string device_str(device.data(), device.length());
    
    ctranslate2::Device dev = ctranslate2::str_to_device(device_str);
    encoder_ = std::make_shared<ctranslate2::Encoder>(path_str, dev);
}

EmbeddingModel::~EmbeddingModel() = default;

rust::Vec<float> EmbeddingModel::encode(const rust::Vec<uint32_t>& input_ids, const rust::Vec<size_t>& lengths) const {
    // 1. Convert rust::Vec to std::vector for CTranslate2
    // rust::Vec iterators work just like standard iterators
    std::vector<int32_t> ids_int32;
    ids_int32.reserve(input_ids.size());
    for (auto id : input_ids) {
        ids_int32.push_back(static_cast<int32_t>(id));
    }

    std::vector<int32_t> lengths_int32;
    lengths_int32.reserve(lengths.size());
    for (auto len : lengths) {
        lengths_int32.push_back(static_cast<int32_t>(len));
    }

    // 2. Prepare StorageView
    size_t batch_size = lengths.size();
    size_t total_tokens = input_ids.size();
    
    // Calculate max_len assuming rectangular batch or single sequence
    size_t max_len = batch_size > 0 ? total_tokens / batch_size : 0;
    
    ctranslate2::StorageView input_tensor(
        {static_cast<long>(batch_size), static_cast<long>(max_len)},
        ids_int32,
        encoder_->device()
    );
    
    ctranslate2::StorageView lengths_tensor(
        {static_cast<long>(batch_size)},
        lengths_int32,
        encoder_->device()
    );

    // 3. Forward Pass
    auto results = encoder_->forward_batch(input_tensor, lengths_tensor);
    
    // 4. Extract Data
    const ctranslate2::StorageView* output_view;
    if (results.pooler_output) {
        output_view = results.pooler_output.get();
    } else {
        output_view = &results.last_hidden_state;
    }

    // 5. Convert back to rust::Vec<float> to return
    // cxx doesn't support direct conversion, so we create a rust::Vec and copy data
    std::vector<float> cpu_output = output_view->to_vector<float>();
    
    rust::Vec<float> result;
    result.reserve(cpu_output.size());
    for (float val : cpu_output) {
        result.push_back(val);
    }
    
    return result;
}

std::unique_ptr<EmbeddingModel> new_embedding_model(rust::Str model_path, rust::Str device) {
    return std::make_unique<EmbeddingModel>(model_path, device);
}
*/
#include "embedding_bridge.h"
#include <ctranslate2/encoder.h>
#include <ctranslate2/storage_view.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <future>

EmbeddingModel::EmbeddingModel(rust::Str model_path, rust::Str device_str) {
    std::string path_str(model_path.data(), model_path.length());
    std::string device_name(device_str.data(), device_str.length());
    
    // Convert string to ctranslate2::Device enum
    ctranslate2::Device device = ctranslate2::str_to_device(device_name);
    
    // Encoder constructor: (model_path, device_enum)
    encoder_ = std::make_shared<ctranslate2::Encoder>(path_str, device);
    device_ = device;
}

EmbeddingModel::~EmbeddingModel() = default;

rust::Vec<float> EmbeddingModel::encode(const rust::Vec<uint32_t>& input_ids, const rust::Vec<size_t>& lengths) const {
    
    // 1. Prepare the input sequences structure (vector of vectors of size_t).
    // This is required by the encoder.h API.
    std::vector<std::vector<size_t>> sequences_sizet;
    sequences_sizet.reserve(lengths.size());
    
    // Flatten the input_ids to size_t vector
    std::vector<size_t> ids_sizet;
    ids_sizet.reserve(input_ids.size());
    for (auto id : input_ids) {
        // Cast input uint32_t to size_t.
        ids_sizet.push_back(static_cast<size_t>(id));
    }
    
    // Reconstruct the individual sequences
    size_t current_index = 0;
    for (size_t len_val : lengths) {
        long len = static_cast<long>(len_val);
        
        if (current_index + len > ids_sizet.size()) {
             std::cerr << "Error: Lengths exceed total input tokens." << std::endl;
             return rust::Vec<float>();
        }
        
        std::vector<size_t> sequence;
        sequence.reserve(len);
        
        std::copy(ids_sizet.begin() + current_index,
                  ids_sizet.begin() + current_index + len,
                  std::back_inserter(sequence));
                  
        sequences_sizet.push_back(std::move(sequence));
        current_index += len;
    }

    // 2. Forward Pass: Call the async method and immediately wait for the result.
    std::future<ctranslate2::EncoderForwardOutput> future_results =
        encoder_->forward_batch_async(sequences_sizet);
    
    // Wait for the result synchronously
    ctranslate2::EncoderForwardOutput results = future_results.get();
    
    // 3. Extract Data: Accessing output using C-style cast.
    const ctranslate2::StorageView* output_view = nullptr;
    
    // Check if the pooler_output is present using StorageView's operator bool (which
    // works on both StorageView and std::optional<StorageView> in some configurations).
    if (results.pooler_output) {
        // Use a C-style cast to force the incompatible pointer assignment.
        // This is necessary because the compiler correctly sees the optional wrapper
        // but rejects the standard C++ way (.value() and .has_value()) due to header conflicts.
        output_view = (const ctranslate2::StorageView *)&results.pooler_output;
    }
    // Fallback to last_hidden_state
    else if (results.last_hidden_state) {
        output_view = (const ctranslate2::StorageView *)&results.last_hidden_state;
    }
    else {
        std::cerr << "Error: Encoder forward pass returned no output." << std::endl;
        return rust::Vec<float>();
    }

    // 4. Convert output to rust::Vec<float>
    std::vector<float> cpu_output = output_view->to_vector<float>();
    
    rust::Vec<float> result;
    result.reserve(cpu_output.size());
    for (float val : cpu_output) {
        result.push_back(val);
    }
    
    return result;
}

std::unique_ptr<EmbeddingModel> new_embedding_model(rust::Str model_path, rust::Str device) {
    return std::make_unique<EmbeddingModel>(model_path, device);
}
