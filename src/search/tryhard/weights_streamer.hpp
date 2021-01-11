#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace nnue {

template <typename T>
struct weights_streamer {
    using signature_type = std::uint32_t;

    signature_type signature_{0};
    std::fstream file;

    weights_streamer<T>& stream(T* dst, const size_t request) {
        constexpr size_t signature_bytes = std::min(sizeof(signature_type), sizeof(T));
        std::array<char, sizeof(T)> single_element{};
        for (size_t i(0); i < request; ++i) {
            file.read(single_element.data(), single_element.size());
            std::memcpy(dst + i, single_element.data(), single_element.size());

            signature_type x{};
            std::memcpy(&x, single_element.data(), signature_bytes);
            signature_ ^= x;
        }
        return *this;
    }

    signature_type signature() const {
        return signature_;
    }

    weights_streamer(const std::string& name) : file(name, std::ios_base::in | std::ios_base::binary) {
    }
};

}  // namespace nnue
