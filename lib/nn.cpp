#include "nn.h"
#include "engine.h"
#include <algorithm>
#include <ostream>
#include <utility>


void Neuron::randomWeightsAndBias()
{
  std::random_device r;
  std::mt19937 gen(r());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  auto g = [&dis, &gen]() { return std::make_shared<Value>(dis(gen)); };
  std::generate(_weights.begin(), _weights.end(), g);
  _bias = std::make_shared<Value>(dis(gen));
}

Neuron::Neuron(size_t nin, ActFun act) : _weights(nin), act(std::move(act)) { randomWeightsAndBias(); }

std::ostream &operator<<(std::ostream &os, const Neuron &n)
{
  os << "Neuron([";
  for (int i = 0; i < n._weights.size(); i++) {
    os << n._weights[i]->data();
    if (i < n._weights.size() - 1) { os << ", "; }
  }
  os << "], " << n._bias->data() << ")";
  return os;
}

Layer::Layer(size_t nin, size_t nout, const ActFun &act)
{
  for (int i = 0; i < nout; i++) { _neurons.emplace_back(nin, act); }
}

std::ostream &operator<<(std::ostream &os, const Layer &l)
{
  os << "Layer([";
  for (int i = 0; i < l._neurons.size(); i++) {
    os << l._neurons[i];
    if (i < l._neurons.size() - 1) { os << ", "; }
  }
  os << "])";
  return os;
}

MLP::MLP(const std::vector<size_t> &sizes, const ActFun &act)
{
  for (int i = 0; i < sizes.size() - 1; i++) { _layers.emplace_back(sizes[i], sizes[i + 1], act); }
}

std::ostream &operator<<(std::ostream &os, const MLP &m)
{
  os << "MLP([";
  for (int i = 0; i < m._layers.size(); i++) {
    os << m._layers[i];
    if (i < m._layers.size() - 1) { os << ", "; }
  }
  os << "])";
  return os;
}

std::vector<ValuePtr> Neuron::parameters() const
{
  std::vector<ValuePtr> p = _weights;
  p.push_back(_bias);
  return p;
}

std::vector<ValuePtr> Layer::parameters() const
{
  std::vector<ValuePtr> p;
  for (const auto &n : _neurons) {
    auto np = n.parameters();
    p.insert(p.end(), np.begin(), np.end());
  }
  return p;
}

std::vector<ValuePtr> MLP::parameters() const
{
  std::vector<ValuePtr> p;
  for (const auto &l : _layers) {
    auto lp = l.parameters();
    p.insert(p.end(), lp.begin(), lp.end());
  }
  return p;
}

MLP gradientDescent(const std::vector<size_t> &hiddenLayerSizes,
  const std::vector<std::vector<double>> &inputs,
  const std::vector<double> &target,
  double lr,
  double tol,
  int niter)
{
  std::vector<size_t> sizes = hiddenLayerSizes;
  sizes.insert(sizes.begin(), inputs[0].size());
  sizes.push_back(target.size());
  MLP mlp(sizes);
  auto params = mlp.parameters();

  for (int i = 0; i < niter; i++) {
    std::vector<std::vector<ValuePtr>> y;
    y.reserve(inputs.size());
    for (const auto &x : inputs) { y.push_back(mlp(x)); }
    auto l = loss(target, y);
    std::cout << "loss: " << l->data() << std::endl;
    if (l->data() < tol) {
      std::cout << "tolerance reached" << std::endl;
      break;
    }
    l->backward();
    for (auto &p : params) { p->_data -= lr * p->grad(); }
  }


  return mlp;
}