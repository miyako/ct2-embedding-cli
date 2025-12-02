#include "embedding_bridge.h"
#include <ctranslate2/encoder.h>
#include <ctranslate2/storage_view.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <future>
#include <numeric>

// Helper function to manually perform Mean Pooling on the last hidden state.
std::vector<float> mean_pool_last_hidden_state(
    const ctranslate2::StorageView& last_hidden_state,
    const std::vector<size_t>& lengths)
{
    // Shape: [Batch_size, Max_len, Hidden_dim]
    const auto& shape = last_hidden_state.shape();
    if (shape.size() != 3) {
        std::cerr << "Error: Last hidden state shape is not 3D." << std::endl;
        return {};
    }
    
    long batch_size = shape[0];
    long max_len = shape[1];
    long hidden_dim = shape[2];
    
    // Transfer data to CPU vector for arithmetic operations
    std::vector<float> flat_data = last_hidden_state.to_vector<float>();
    
    std::vector<float> pooled_output;
    pooled_output.reserve(batch_size * hidden_dim);
    size_t data_offset = 0;

    for (long i = 0; i < batch_size; ++i) {
        long seq_len = lengths[i];
        std::vector<float> sentence_sum(hidden_dim, 0.0f);

        for (long t = 0; t < seq_len; ++t) {
            size_t token_start_index = data_offset + (t * hidden_dim);
            for (long d = 0; d < hidden_dim; ++d) {
                size_t index = token_start_index + d;
                sentence_sum[d] += flat_data[index];
            }
        }
        
        if (seq_len > 0) {
            float inv_seq_len = 1.0f / static_cast<float>(seq_len);
            for (float& val : sentence_sum) {
                val *= inv_seq_len;
                pooled_output.push_back(val);
            }
        } else {
             for (long d = 0; d < hidden_dim; ++d) {
                 pooled_output.push_back(0.0f);
             }
        }
        
        data_offset += max_len * hidden_dim;
    }

    return pooled_output;
}

// -----------------------------------------------------------------------------

EmbeddingModel::EmbeddingModel(rust::Str model_path, rust::Str device_str) {
    std::string path_str(model_path.data(), model_path.length());
    std::string device_name(device_str.data(), device_str.length());
    
    ctranslate2::Device device = ctranslate2::str_to_device(device_name);
    
    encoder_ = std::make_shared<ctranslate2::Encoder>(path_str, device);
    device_ = device;
}

// FIX: Explicitly define the destructor to ensure the symbol is generated for the linker.
EmbeddingModel::~EmbeddingModel() {
    // The shared_ptr encoder_ will be automatically cleaned up.
}

rust::Vec<float> EmbeddingModel::encode(const rust::Vec<uint32_t>& input_ids, const rust::Vec<size_t>& lengths) const {
    
    // 1. Prepare inputs
    std::vector<std::vector<size_t>> sequences_sizet;
    sequences_sizet.reserve(lengths.size());
    
    std::vector<size_t> ids_sizet;
    ids_sizet.reserve(input_ids.size());
    for (auto id : input_ids) {
        ids_sizet.push_back(static_cast<size_t>(id));
    }
    
    std::vector<size_t> lengths_sizet;
    lengths_sizet.reserve(lengths.size());

    size_t current_index = 0;
    for (size_t len_val : lengths) {
        long len = static_cast<long>(len_val);
        lengths_sizet.push_back(len_val);
        
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

    // 2. Forward Pass
    std::future<ctranslate2::EncoderForwardOutput> future_results =
        encoder_->forward_batch_async(sequences_sizet);
    
    ctranslate2::EncoderForwardOutput results = future_results.get();
    
    // 3. Extract and Pool Data
    std::vector<float> cpu_output;

    // We skip results.pooler_output to enforce Mean Pooling, which is standard for Sentence Transformers.
    if (results.last_hidden_state) {
        // Perform Mean Pooling on the last hidden state
        cpu_output = mean_pool_last_hidden_state(
            results.last_hidden_state,
            lengths_sizet
        );
    }
    else {
        std::cerr << "Error: Encoder forward pass returned no output." << std::endl;
        return rust::Vec<float>();
    }

    // 4. Convert output to rust::Vec<float>
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
