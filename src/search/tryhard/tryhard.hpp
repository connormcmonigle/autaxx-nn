#ifndef SEARCH_TRYHARD_HPP
#define SEARCH_TRYHARD_HPP

#include <libataxx/move.hpp>
#include <libataxx/position.hpp>
#include "../../utils.hpp"
#include "../pv.hpp"
#include "../search.hpp"
#include "../tt.hpp"
#include "nnue_model.hpp"
#include "ttentry.hpp"

namespace search {

namespace tryhard {

constexpr int mate_score = 10000;
constexpr int max_depth = 128;

constexpr int eval_to_tt(const int eval, const int ply) {
    if (eval > mate_score - max_depth) {
        return eval + ply;
    }
    if (eval < -mate_score + max_depth) {
        return eval - ply;
    }
    return eval;
}

constexpr int eval_from_tt(const int eval, const int ply) {
    if (eval > mate_score - max_depth) {
        return eval - ply;
    }
    if (eval < -mate_score + max_depth) {
        return eval + ply;
    }
    return eval;
}

class Tryhard : public Search {
   public:
    struct Stack {
        int ply;
        PV pv;
        libataxx::Move killer;
        bool nullmove;
    };

    Tryhard(const unsigned int mb, const std::string &weights_path) : tt_{mb} {
        weights_.load(weights_path);
    }

    void go(const libataxx::Position pos, const Settings &settings) override {
        stop();
        search_thread_ = std::thread(&Tryhard::root, this, pos, settings);
    }

    void clear() noexcept override {
        tt_.clear();
        for (int i = 0; i < max_depth + 1; ++i) {
            stack_[i].ply = i;
            stack_[i].pv.clear();
            stack_[i].killer = libataxx::Move::nomove();
            stack_[i].nullmove = true;
        }
    }

    [[nodiscard]] static int eval(
        const libataxx::Position &pos,
        const nnue::weights<float> &weights) noexcept {
        auto evaluator = nnue::eval<float>{&weights};

        for (const auto &sq : pos.white()) {
            evaluator.white.insert(sq.index());
            evaluator.black.insert(7 * 7 + sq.index());
        }

        for (const auto &sq : pos.black()) {
            evaluator.black.insert(sq.index());
            evaluator.white.insert(7 * 7 + sq.index());
        }

        const int score = evaluator.evaluate(static_cast<bool>(pos.turn()));
        return score;
    }

   private:
    void root(const libataxx::Position pos, const Settings &settings) noexcept;

    [[nodiscard]] int search(Stack *stack,
                             const libataxx::Position &pos,
                             int alpha,
                             int beta,
                             int depth);

    Stack stack_[max_depth + 1];
    TT<TTEntry> tt_;
    nnue::weights<float> weights_{};
};

}  // namespace tryhard

}  // namespace search

#endif
