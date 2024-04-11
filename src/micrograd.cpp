#include "engine.h"
#include "nn.h"
#include <format>
#include <iostream>
#include <memory>

int main()
{
  std::vector<std::vector<double>> xs = { { 2, 3, -1 }, { 3, -1, 0.5 }, { 0.5, 1, 1 }, { 1, 1, -1 } };
  std::vector<double> ys = { 1, -1, -1, 1 };
  auto mlp = gradientDescent({ 4, 4 }, xs, ys, 0.1, 1e-6, 1000);
  for (auto &x : xs) {
    std::cout << "---\n";
    auto out = mlp(x);
    for (auto &v : out) { std::cout << v->data() << " "; }
    std::cout << std::endl;
  }
  return 0;
}
