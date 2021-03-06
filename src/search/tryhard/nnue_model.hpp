#pragma once

#include <cstdint>
#include <string>
#include "nnue_util.hpp"
#include "weights_streamer.hpp"

namespace nnue {

constexpr size_t half_ka_numel = 49 * 2;
constexpr size_t base_dim = 32;

template <typename T>
struct weights {
    typename weights_streamer<T>::signature_type signature_{0};
    big_affine<T, half_ka_numel, base_dim> w{};
    big_affine<T, half_ka_numel, base_dim> b{};
    stack_affine<T, 2 * base_dim, 32> fc0{};
    stack_affine<T, 32, 32> fc1{};
    stack_affine<T, 64, 1> fc2{};

    size_t signature() const {
        return signature_;
    }

    size_t num_parameters() const {
        return w.num_parameters() + b.num_parameters() + fc0.num_parameters() + fc1.num_parameters() +
               fc2.num_parameters();
    }

    weights<T>& load(weights_streamer<T>& ws) {
        w.load_(ws);
        b.load_(ws);
        fc0.load_(ws);
        fc1.load_(ws);
        fc2.load_(ws);
        signature_ = ws.signature();
        return *this;
    }

    weights<T>& load(const std::string& path) {
        auto ws = weights_streamer<T>(path);
        return load(ws);
    }
};

template <typename T>
struct feature_transformer {
    const big_affine<T, half_ka_numel, base_dim>* weights_;
    stack_vector<T, base_dim> active_;

    constexpr stack_vector<T, base_dim> active() const {
        return active_;
    }

    void clear() {
        active_ = stack_vector<T, base_dim>::from(weights_->b);
    }

    void insert(const size_t idx) {
        weights_->insert_idx(idx, active_);
    }

    void erase(const size_t idx) {
        weights_->erase_idx(idx, active_);
    }

    feature_transformer(const big_affine<T, half_ka_numel, base_dim>* src) : weights_{src} {
        clear();
    }
};

template <typename T>
struct eval {
    const weights<T>* weights_;
    feature_transformer<T> white;
    feature_transformer<T> black;

    constexpr T propagate(const bool pov) const {
        const auto w_x = white.active();
        const auto b_x = black.active();
        const auto x0 = pov ? splice(w_x, b_x).apply_(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
        const auto x1 = (weights_->fc0).forward(x0).apply_(relu<T>);
        const auto x2 = splice(x1, (weights_->fc1).forward(x1).apply_(relu<T>));
        const T val = (weights_->fc2).forward(x2).item();
        return val;
    }

    constexpr int evaluate(const bool pov) const {
        const T value = 600.0 * propagate(pov);
        return static_cast<int>(value);
    }

    eval(const weights<T>* src) : weights_{src}, white{&(src->w)}, black{&(src->b)} {
    }
};

}  // namespace nnue
