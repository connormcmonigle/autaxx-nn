cmake_minimum_required(VERSION 3.12)

# Project
project(Autaxx VERSION 1.0 LANGUAGES CXX)

# Base directory relative includes
include_directories(.)

# Third party includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs/libataxx/src/)

# Flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add the executable
add_executable(
    autaxx
    src/main.cpp
    src/options.cpp
    src/protocol/uai/go.cpp
    src/protocol/uai/isready.cpp
    src/protocol/uai/listen.cpp
    src/protocol/uai/moves.cpp
    src/protocol/uai/position.cpp
    src/protocol/uai/setoption.cpp
    src/protocol/uai/uainewgame.cpp
    src/protocol/uai/extension/display.cpp
    src/protocol/uai/extension/perft.cpp
    src/protocol/uai/extension/split.cpp
    src/search/search.cpp
    src/search/tryhard/classical.cpp
    src/search/tryhard/search.cpp
    src/search/tryhard/root.cpp
    src/search/mcts/eval.cpp
    src/search/mcts/root.cpp
    src/search/minimax/minimax.cpp
    src/search/minimax/eval.cpp
    src/search/minimax/root.cpp

    src/search/alphabeta/search.cpp
    src/search/alphabeta/eval.cpp
    src/search/alphabeta/root.cpp
)

target_link_libraries(autaxx "${CMAKE_CURRENT_LIST_DIR}/libs/libataxx/build/static/libataxx.a")
