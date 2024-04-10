#include "engine.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>

TEST_CASE("exp(a * b)")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = exp(a * b);
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == std::exp(3 * 4));
  REQUIRE(a->grad() == 4 * std::exp(3 * 4));
  REQUIRE(b->grad() == 3 * std::exp(3 * 4));
}

TEST_CASE("neg a")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = -a;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == -3);
  REQUIRE(a->grad() == -1);
}

TEST_CASE("a + b")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = a + b;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 3 + 4);
  REQUIRE(a->grad() == 1);
  REQUIRE(b->grad() == 1);
}

TEST_CASE("a - b")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = a - b;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 3 - 4);
  REQUIRE(a->grad() == 1);
  REQUIRE(b->grad() == -1);
}

TEST_CASE("a * b")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = a * b;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 3 * 4);
  REQUIRE(a->grad() == 4);
  REQUIRE(b->grad() == 3);
}

TEST_CASE("a / b")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = a / b;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 3.0 / 4.0);
  REQUIRE(a->grad() == 1.0 / 4.0);
  REQUIRE(b->grad() == -3.0 / 16.0);
}

TEST_CASE("a + 1")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = a + 1;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 3 + 1);
  REQUIRE(a->grad() == 1);
}

TEST_CASE("1 + a")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = 1 + a;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 1 + 3);
  REQUIRE(a->grad() == 1);
}

TEST_CASE("a - 1")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = a - 1;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 3 - 1);
  REQUIRE(a->grad() == 1);
}

TEST_CASE("1 - a")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = 1 - a;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 1 - 3);
  REQUIRE(a->grad() == -1);
}

TEST_CASE("a * 2")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = a * 2;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 3 * 2);
  REQUIRE(a->grad() == 2);
}

TEST_CASE("2 * a")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = 2 * a;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 2 * 3);
  REQUIRE(a->grad() == 2);
}

TEST_CASE("a / 2")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = a / 2;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 3.0 / 2.0);
  REQUIRE(a->grad() == 0.5);
}

TEST_CASE("2 / a")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = 2 / a;
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 2.0 / 3.0);
  REQUIRE(a->grad() == -2.0 / 9.0);
}

TEST_CASE("a + b + c")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = std::make_shared<Value>(5);
  ValuePtr d = a + b + c;
  d->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 5);
  REQUIRE(d->data() == 3 + 4 + 5);
  REQUIRE(a->grad() == 1);
  REQUIRE(b->grad() == 1);
  REQUIRE(c->grad() == 1);
}

TEST_CASE("a + b + c + d")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = std::make_shared<Value>(5);
  ValuePtr d = std::make_shared<Value>(6);
  ValuePtr e = a + b + c + d;
  e->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 5);
  REQUIRE(d->data() == 6);
  REQUIRE(e->data() == 3 + 4 + 5 + 6);
  REQUIRE(a->grad() == 1);
  REQUIRE(b->grad() == 1);
  REQUIRE(c->grad() == 1);
  REQUIRE(d->grad() == 1);
}

TEST_CASE("a * b * c")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = std::make_shared<Value>(5);
  ValuePtr d = a * b * c;
  d->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 5);
  REQUIRE(d->data() == 3 * 4 * 5);
  REQUIRE(a->grad() == 4 * 5);
  REQUIRE(b->grad() == 3 * 5);
  REQUIRE(c->grad() == 3 * 4);
}

TEST_CASE("a * b * c * d")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = std::make_shared<Value>(5);
  ValuePtr d = std::make_shared<Value>(6);
  ValuePtr e = a * b * c * d;
  e->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 5);
  REQUIRE(d->data() == 6);
  REQUIRE(e->data() == 3 * 4 * 5 * 6);
  REQUIRE(a->grad() == 4 * 5 * 6);
  REQUIRE(b->grad() == 3 * 5 * 6);
  REQUIRE(c->grad() == 3 * 4 * 6);
  REQUIRE(d->grad() == 3 * 4 * 5);
}

TEST_CASE("a / b / c")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = std::make_shared<Value>(4);
  ValuePtr d = (a / b) / c;
  d->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 4);
  REQUIRE_THAT(d->data(), Catch::Matchers::WithinAbs(3.0 / 4.0 / 4.0, 1e-6));
  REQUIRE_THAT(a->grad(), Catch::Matchers::WithinAbs(1.0 / 4.0 / 4.0, 1e-6));
  REQUIRE_THAT(b->grad(), Catch::Matchers::WithinAbs(-3.0 / 16.0 / 4.0, 1e-6));
  REQUIRE_THAT(c->grad(), Catch::Matchers::WithinAbs(-3.0 / 16.0 / 4.0, 1e-6));
}

TEST_CASE("a / b / c / d")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(4);
  ValuePtr c = std::make_shared<Value>(4);
  ValuePtr d = std::make_shared<Value>(4);
  ValuePtr e = a / b / c / d;
  e->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == 4);
  REQUIRE(c->data() == 4);
  REQUIRE(d->data() == 4);
  REQUIRE_THAT(e->data(), Catch::Matchers::WithinAbs(3.0 / 4.0 / 4.0 / 4.0, 1e-6));
  REQUIRE_THAT(a->grad(), Catch::Matchers::WithinAbs(1.0 / 4.0 / 4.0 / 4.0, 1e-6));
  REQUIRE_THAT(b->grad(), Catch::Matchers::WithinAbs(-3.0 / 16.0 / 4.0 / 4.0, 1e-6));
  REQUIRE_THAT(c->grad(), Catch::Matchers::WithinAbs(-3.0 / 16.0 / 4.0 / 4.0, 1e-6));
  REQUIRE_THAT(d->grad(), Catch::Matchers::WithinAbs(-3.0 / 16.0 / 4.0 / 4.0, 1e-6));
}

TEST_CASE("tanh(a)")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = tanh(a);
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE_THAT(c->data(), Catch::Matchers::WithinAbs(std::tanh(3), 1e-6));
  REQUIRE_THAT(a->grad(), Catch::Matchers::WithinAbs(1 - std::tanh(3) * std::tanh(3), 1e-6));
}

TEST_CASE("exp(a)")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = exp(a);
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE_THAT(c->data(), Catch::Matchers::WithinAbs(std::exp(3), 1e-6));
  REQUIRE_THAT(a->grad(), Catch::Matchers::WithinAbs(std::exp(3), 1e-6));
}

TEST_CASE("relu(a)")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr c = relu(a);
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(c->data() == 3);
  REQUIRE(a->grad() == 1);
}

TEST_CASE("relu(-a)")
{
  ValuePtr a = std::make_shared<Value>(-3);
  ValuePtr c = relu(a);
  c->backward();
  REQUIRE(a->data() == -3);
  REQUIRE(c->data() == 0);
  REQUIRE(a->grad() == 0);
}

TEST_CASE("relu(a) + relu(b)")
{
  ValuePtr a = std::make_shared<Value>(3);
  ValuePtr b = std::make_shared<Value>(-3);
  ValuePtr c = relu(a) + relu(b);
  c->backward();
  REQUIRE(a->data() == 3);
  REQUIRE(b->data() == -3);
  REQUIRE(c->data() == 3);
  REQUIRE(a->grad() == 1);
  REQUIRE(b->grad() == 0);
}