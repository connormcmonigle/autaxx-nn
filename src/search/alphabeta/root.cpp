#include <cassert>
#include <iostream>
#include "alphabeta.hpp"

using namespace std::chrono;

namespace search {

namespace alphabeta {

void Alphabeta::root(const libataxx::Position pos,
                     const Settings &settings) noexcept {
    // Clear
    stats_.clear();
    for (int i = 0; i < max_depth + 1; ++i) {
        stack_[i].ply = i;
        stack_[i].pv.clear();
        stack_[i].killer = libataxx::Move::nomove();
        stack_[i].nullmove = true;
    }

    PV pv;
    const auto start_time = high_resolution_clock::now();
    int depth = max_depth;

    controller_.max_nodes = std::numeric_limits<std::uint64_t>::max();
    controller_.end_time = start_time + hours(1);

    // Time management
    switch (settings.type) {
        case Type::Time: {
            int search_time = 0;

            // Calculate time usage
            if (pos.turn() == libataxx::Side::Black) {
                search_time = settings.btime / 30;
            } else {
                search_time = settings.wtime / 30;
            }

            // Minimum time
            if (search_time < 1) {
                search_time = 1;
            }
            assert(search_time > 0);

            controller_.end_time = start_time + milliseconds(search_time);
            break;
        }
        case Type::Depth:
            depth = settings.depth;
            break;
        case Type::Nodes:
            controller_.max_nodes = settings.nodes;
            break;
        case Type::Movetime:
            controller_.end_time = start_time + milliseconds(settings.movetime);
            break;
        case Type::Infinite:
            break;
        default:
            break;
    }

    // Iterative deepening
    for (int i = 1; i <= depth; ++i) {
        const int score = alphabeta(stack_, pos, -mate_score, mate_score, i);
        const auto finish = high_resolution_clock::now();

        assert(-mate_score < score && score < mate_score);

        if (i > 1 &&
            (controller_.stop || stats_.nodes >= controller_.max_nodes ||
             high_resolution_clock::now() > controller_.end_time)) {
            break;
        }

        // Update our main pv
        pv = stack_[0].pv;
        assert(legal_pv(pos, pv));

#ifndef NDEBUG
        // The TT should always have the root position in it
        const auto ttentry = tt_.poll(pos.hash());
        assert(ttentry.hash == pos.hash());
        assert(ttentry.move == pv[0]);
        assert(pos.legal_move(ttentry.move));
        assert(tt_.poll(pos.hash()).depth >= i);
#endif

        // Send info string
        duration<double> elapsed = finish - start_time;
        std::cout << "info";
        std::cout << " depth " << i;
        std::cout << " seldepth " << stats_.seldepth;
        std::cout << " score cp " << score;
        std::cout << " time " << static_cast<int>(elapsed.count() * 1000);
        std::cout << " nodes " << stats_.nodes;
        std::cout << " tthits " << stats_.tthits;
        std::cout << " hashfull " << tt_.hashfull();
        if (elapsed.count() > 0) {
            std::cout << " nps "
                      << static_cast<std::uint64_t>(stats_.nodes /
                                                    elapsed.count());
        }
        if (pv.size() > 0) {
            std::cout << " pv";
            for (const auto &move : pv) {
                std::cout << " " << move;
            }
        }
        std::cout << std::endl;
    }

#ifndef NDEBUG
    std::uint64_t total = 0;
    for (int i = 0; i < libataxx::max_moves; ++i) {
        total += stats_.cutoffs[i];
    }
    for (int i = 0; i < 10 && total; ++i) {
        const auto percent =
            100 * static_cast<float>(stats_.cutoffs[i]) / total;
        std::cout << "info string";
        std::cout << " index " << i;
        std::cout << " cutoffs " << percent << "%";
        std::cout << std::endl;
    }
#endif

    if (pv.size() > 0) {
        std::cout << "bestmove " << pv.at(0) << std::endl;
    } else {
        std::cout << "bestmove 0000" << std::endl;
    }
}

}  // namespace alphabeta

}  // namespace search
