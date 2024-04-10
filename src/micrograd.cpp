#include <__format/format_functions.h>
#include <engine.h>
#include <format>
#include <iostream>
#include <memory>

int main()
{
  auto a = std::make_shared<Value>(-4.0);
  auto b = std::make_shared<Value>(2.0);
  auto c = a + b;
  auto d = a * b + pow(b, 3);
  c += c + 1;
  c += 1 + c + (-a);
  d += d * 2 + relu(b + a);
  d += 3 * d + relu(b - a);
  auto e = c - d;
  auto f = pow(e, 2);
  auto g = f / 2.0;
  g = g + 10.0 / f;
  std::cout << std::format("{:.4f}", g->data()) << std::endl;
  g->backward();
  std::cout << std::format("{:.4f}", a->grad()) << std::endl;
  std::cout << std::format("{:.4f}", b->grad()) << std::endl;


  return 0;
}
