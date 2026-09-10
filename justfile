build:
    cmake -B build
    cmake --build build -j8

format:
    find src tests -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

format-check:
    find src tests -name '*.hpp' -o -name '*.cpp' | xargs clang-format --dry-run --Werror

test: build
    ./build/mkx_tests

check: format-check test
