#pragma once
#include "engine.h"
#include <functional>
#include <iostream>
#include <ostream>
#include <random>

using ActFun = std::function<ValuePtr(const ValuePtr &)>;
class Neuron
{
private:
  std::vector<ValuePtr> _weights{};
  ValuePtr _bias;
  ActFun act{ [](const ValuePtr &x) { return tanh(x); } };
  void randomWeightsAndBias();

public:
  explicit Neuron(
    size_t nin,
    ActFun act = [](const ValuePtr &x) { return tanh(x); });
  template<typename T> ValuePtr operator()(const std::vector<T> &x)
  {
    ValuePtr sum = _bias;
    for (int i = 0; i < x.size(); i++) { sum += x[i] * _weights[i]; }
    return act(sum);
  }
  friend std::ostream &operator<<(std::ostream &os, const Neuron &n);
  [[nodiscard]] std::vector<ValuePtr> parameters() const;
};

class Layer
{
private:
  std::vector<Neuron> _neurons{};

public:
  explicit Layer(
    size_t nin,
    size_t nout,
    const ActFun &act = [](const ValuePtr &x) { return tanh(x); });
  template<typename T> std::vector<ValuePtr> operator()(const std::vector<T> &x)
  {
    std::vector<ValuePtr> y(_neurons.size());
    for (int i = 0; i < _neurons.size(); i++) { y[i] = _neurons[i](x); }
    return y;
  }
  friend std::ostream &operator<<(std::ostream &os, const Layer &l);
  [[nodiscard]] std::vector<ValuePtr> parameters() const;
};

class MLP
{
private:
  std::vector<Layer> _layers{};

public:
  explicit MLP(
    const std::vector<size_t> &sizes,
    const ActFun &act = [](const ValuePtr &x) { return tanh(x); });
  template<typename T> std::vector<ValuePtr> operator()(const std::vector<T> &x)
  {
    std::vector<ValuePtr> y;
    if constexpr (std::is_same_v<T, ValuePtr>) {
      y = x;
    } else {
      y.resize(x.size());
      for (int i = 0; i < x.size(); i++) { y[i] = std::make_shared<Value>(x[i]); }
    }
    for (auto &l : _layers) { y = l(y); }
    return y;
  }
  friend std::ostream &operator<<(std::ostream &os, const MLP &m);
  [[nodiscard]] std::vector<ValuePtr> parameters() const;
};

template<typename T> ValuePtr loss(const std::vector<T> &target, const std::vector<std::vector<ValuePtr>> &outputs)
{
  ValuePtr sum = std::make_shared<Value>(0.0);
  for (const auto &y : outputs) {
    for (int i = 0; i < target.size(); i++) { sum += pow(target[i] - y[i], 2); }
  }
  return sum;
}

MLP gradientDescent(const std::vector<size_t> &hiddenLayerSizes,
  const std::vector<std::vector<double>> &inputs,
  const std::vector<double> &target,
  double lr = 0.01,
  double tol = 1e-3,
  int niter = 100);