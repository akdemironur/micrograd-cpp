#include <__format/format_functions.h>
#include <engine.h>
#include <iostream>
#include <memory>

int main()
{
  ValuePtr a = std::make_shared<Value>(5, "a");
  ValuePtr b = std::make_shared<Value>(1, "b");
  ValuePtr c = pow(a, 1) + 1 / a + b / a - b / 2 + tanh(a * b / 2 + relu(a)) - pow(a, 2 * b);
  c->backward();
  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "c: " << c << std::endl;
  return 0;
}
