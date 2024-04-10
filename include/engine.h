#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

enum OpType { NONE, ADD, MUL, EXP, POW, RELU, SUB, DIV, TANH };

class Value
{
private:
  double _data{};
  double _grad{};
  std::function<void()> _backward{ []() {} };
  std::string _label{};
  std::vector<std::shared_ptr<Value>> _prev;
  OpType op{};
  [[nodiscard]] std::vector<Value *> topo()
  {
    std::vector<Value *> t{};
    std::set<Value *> s{};
    buildTopo(this, t, s);
    return t;
  }
  static void buildTopo(Value *v, std::vector<Value *> &topo, std::set<Value *> &seen)
  {
    if (seen.find(v) != seen.end()) { return; }
    seen.insert(v);
    for (const auto &p : v->_prev) { buildTopo(p.get(), topo, seen); }
    topo.push_back(v);
  }

public:
  Value() = default;
  explicit Value(double data) : _data(data), _label(std::to_string(data)) {}
  explicit Value(double data, std::string label) : _data(data), _label(std::move(label)) {}
  [[nodiscard]] double data() const { return _data; }

  [[nodiscard]] double grad() const { return _grad; }

  void backward()
  {
    _grad = 1.0;
    auto topo_order = topo();
    for (auto &it : std::ranges::reverse_view(topo_order)) {
      if (it != nullptr) { it->_backward(); }
    }
  }

  using ValuePtr = std::shared_ptr<Value>;

  friend ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs)
  {
    ValuePtr out = std::make_shared<Value>();
    out->_data = lhs->_data + rhs->_data;
    out->_prev.push_back(lhs);
    out->_prev.push_back(rhs);
    out->op = ADD;
    out->_label = lhs->_label + "+" + rhs->_label;
    out->_backward = [lhs, rhs, out]() {
      lhs->_grad += out->_grad;
      rhs->_grad += out->_grad;
    };
    return out;
  }

  friend ValuePtr operator-(const ValuePtr &lhs, const ValuePtr &rhs) { return lhs + (-rhs); }

  friend ValuePtr operator*(const ValuePtr &lhs, const ValuePtr &rhs)
  {
    ValuePtr out = std::make_shared<Value>();
    out->_data = lhs->_data * rhs->_data;
    out->_prev.push_back(lhs);
    out->_prev.push_back(rhs);
    out->_label = lhs->_label + "*" + rhs->_label;
    out->op = MUL;
    out->_backward = [lhs, rhs, out]() {
      lhs->_grad += out->_grad * rhs->_data;
      rhs->_grad += out->_grad * lhs->_data;
    };
    return out;
  }

  friend ValuePtr exp(const ValuePtr &v)
  {
    ValuePtr out = std::make_shared<Value>();
    out->_data = std::exp(v->_data);
    out->_prev.push_back(v);
    out->op = EXP;
    out->_label = "exp(" + v->_label + ")";
    out->_backward = [v, out]() { v->_grad += out->_grad * std::exp(v->_data); };
    return out;
  }

  friend ValuePtr pow(const ValuePtr &x, const ValuePtr &a)
  {
    ValuePtr out = std::make_shared<Value>();
    out->_data = std::pow(x->_data, a->_data);
    out->_prev.push_back(x);
    out->_prev.push_back(a);
    out->op = POW;
    out->_label = "pow(" + x->_label + "," + a->_label + ")";
    out->_backward = [x, a, out]() {
      x->_grad += out->_grad * a->_data * std::pow(x->_data, a->_data - 1);
      a->_grad += out->_grad * std::log(x->_data) * std::pow(x->_data, a->_data);
    };
    return out;
  }

  friend ValuePtr relu(const ValuePtr &v)
  {
    ValuePtr out = std::make_shared<Value>();
    out->_data = std::max(0.0, v->_data);
    out->_prev.push_back(v);
    out->op = RELU;
    out->_label = "relu(" + v->_label + ")";
    out->_backward = [v, out]() { v->_grad += out->_grad * (v->_data > 0 ? 1 : 0); };
    return out;
  }

  friend ValuePtr operator/(const ValuePtr &lhs, const ValuePtr &rhs) { return lhs * pow(rhs, -1); }

  template<typename T> friend ValuePtr operator+(const T &lhs, const ValuePtr &rhs)
  {
    return std::make_shared<Value>(lhs) + rhs;
  }

  template<typename T> friend ValuePtr operator+(const ValuePtr &lhs, const T &rhs)
  {
    return lhs + std::make_shared<Value>(rhs);
  }

  template<typename T> friend ValuePtr operator*(const T &lhs, const ValuePtr &rhs)
  {
    return std::make_shared<Value>(lhs) * rhs;
  }

  template<typename T> friend ValuePtr operator*(const ValuePtr &lhs, const T &rhs)
  {
    return lhs * std::make_shared<Value>(rhs);
  }

  friend ValuePtr operator-(const ValuePtr &rhs) { return -1.0 * rhs; }


  template<typename T> friend ValuePtr operator-(const T &lhs, const ValuePtr &rhs)
  {
    return std::make_shared<Value>(lhs) - rhs;
  }

  template<typename T> friend ValuePtr operator-(const ValuePtr &lhs, const T &rhs)
  {
    return lhs - std::make_shared<Value>(rhs);
  }

  friend std::ostream &operator<<(std::ostream &ostr, const ValuePtr &value)
  {
    ostr << "Value: " << value->_data << " Grad: " << value->_grad;
    return ostr;
  }

  template<typename T> friend ValuePtr pow(const T &x, const ValuePtr &a) { return pow(std::make_shared<Value>(x), a); }

  template<typename T> friend ValuePtr pow(const ValuePtr &x, const T &a) { return pow(x, std::make_shared<Value>(a)); }

  template<typename T> friend ValuePtr operator/(const T &lhs, const ValuePtr &rhs)
  {
    return std::make_shared<Value>(lhs) / rhs;
  }

  template<typename T> friend ValuePtr operator/(const ValuePtr &lhs, const T &rhs)
  {
    return lhs / std::make_shared<Value>(rhs);
  }

  friend ValuePtr tanh(const ValuePtr &v)
  {
    ValuePtr out = std::make_shared<Value>();
    out->_data = std::tanh(v->_data);
    out->_prev.push_back(v);
    out->op = TANH;
    out->_label = "tanh(" + v->_label + ")";
    out->_backward = [v, out]() { v->_grad += out->_grad * (1.0 - std::tanh(v->_data) * std::tanh(v->_data)); };
    return out;
  }
};

using ValuePtr = std::shared_ptr<Value>;