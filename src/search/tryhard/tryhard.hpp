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

    Tryhard(const unsigned int mb, const nnue::weights<float> &weights)
        : tt_{mb}, evaluator_{&weights}, turn_{false} {
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

    void init_pos(const libataxx::Position &pos) noexcept {
        evaluator_.white.clear();
        evaluator_.black.clear();

        for (const auto &sq : pos.white()) {
            evaluator_.white.insert(sq.index());
            evaluator_.black.insert(7 * 7 + sq.index());
        }

        for (const auto &sq : pos.black()) {
            evaluator_.black.insert(sq.index());
            evaluator_.white.insert(7 * 7 + sq.index());
        }

        turn_ = static_cast<bool>(pos.turn());
    }

    void update(const libataxx::Position &pos, const libataxx::Move &move) {
        turn_ = !turn_;

        // Handle nullmove
        if (move == libataxx::Move::nullmove()) {
            return;
        }

        const auto to_bb = libataxx::Bitboard{move.to()};
        const auto from_bb = libataxx::Bitboard{move.from()};
        const auto them_unset = to_bb.singles() & pos.them();
        const auto us_set = them_unset | to_bb;
        const auto us_unset = from_bb & (~to_bb);

        if (pos.turn() == libataxx::Side::White) {
            for (const auto sq : us_set) {
                evaluator_.white.insert(sq.index());
                evaluator_.black.insert(7 * 7 + sq.index());
            }

            for (const auto sq : us_unset) {
                evaluator_.white.erase(sq.index());
                evaluator_.black.erase(7 * 7 + sq.index());
            }

            for (const auto sq : them_unset) {
                evaluator_.black.erase(sq.index());
                evaluator_.white.erase(7 * 7 + sq.index());
            }
        } else {
            for (const auto sq : us_set) {
                evaluator_.black.insert(sq.index());
                evaluator_.white.insert(7 * 7 + sq.index());
            }

            for (const auto sq : us_unset) {
                evaluator_.black.erase(sq.index());
                evaluator_.white.erase(7 * 7 + sq.index());
            }

            for (const auto sq : them_unset) {
                evaluator_.white.erase(sq.index());
                evaluator_.black.erase(7 * 7 + sq.index());
            }
        }
    }

    void downdate(const libataxx::Position &pos, const libataxx::Move &move) {
        turn_ = !turn_;

        // Handle nullmove
        if (move == libataxx::Move::nullmove()) {
            return;
        }

        const auto to_bb = libataxx::Bitboard{move.to()};
        const auto from_bb = libataxx::Bitboard{move.from()};
        const auto them_unset = to_bb.singles() & pos.them();
        const auto us_set = them_unset | to_bb;
        const auto us_unset = from_bb & (~to_bb);

        if (pos.turn() == libataxx::Side::White) {
            for (const auto sq : us_set) {
                evaluator_.white.erase(sq.index());
                evaluator_.black.erase(7 * 7 + sq.index());
            }

            for (const auto sq : us_unset) {
                evaluator_.white.insert(sq.index());
                evaluator_.black.insert(7 * 7 + sq.index());
            }

            for (const auto sq : them_unset) {
                evaluator_.black.insert(sq.index());
                evaluator_.white.insert(7 * 7 + sq.index());
            }
        } else {
            for (const auto sq : us_set) {
                evaluator_.black.erase(sq.index());
                evaluator_.white.erase(7 * 7 + sq.index());
            }

            for (const auto sq : us_unset) {
                evaluator_.black.insert(sq.index());
                evaluator_.white.insert(7 * 7 + sq.index());
            }

            for (const auto sq : them_unset) {
                evaluator_.white.insert(sq.index());
                evaluator_.black.insert(7 * 7 + sq.index());
            }
        }
    }

    [[nodiscard]] int eval() noexcept {
        return evaluator_.evaluate(turn_);
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

    nnue::eval<float> evaluator_;
    bool turn_;

   private:
    void root(const libataxx::Position pos, const Settings &settings) noexcept;

    [[nodiscard]] int search(Stack *stack,
                             const libataxx::Position &pos,
                             int alpha,
                             int beta,
                             int depth);

    Stack stack_[max_depth + 1];
    TT<TTEntry> tt_;
};

}  // namespace tryhard

}  // namespace search

#endif
