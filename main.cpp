#include <iostream>
#include "parallel/unittest.hpp"
#include "parallel/app.hpp"
#include "parallel/benchmark.hpp"

using namespace std;
int main(int argc, char **argv) {
  parallel::benchmark_run();
  return 0;
}
