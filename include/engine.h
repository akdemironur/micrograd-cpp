#pragma once
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#include <set>
#include <string>
#include <vector>
enum OpType { NONE, ADD, MUL, EXP, POW, RELU, SUB, DIV, TANH };
std::string opToString(OpType op);
std::pair<std::string, std::string> opDot(OpType *op);

class Value
{
private:
  double _data{};
  double _grad{};
  std::function<void()> _backward{ []() {} };
  std::string _label{};
  std::string _topology_dot_repr{};
  std::vector<std::shared_ptr<Value>> _prev;
  OpType op{};

  [[nodiscard]] std::vector<Value *> topo();
  static void buildTopo(Value *v, std::vector<Value *> &topo, std::set<Value *> &seen);


public:
  using ValuePtr = std::shared_ptr<Value>;
  Value() = default;
  explicit Value(double data) : _data(data) {}
  explicit Value(double data, std::string label) : _data(data), _label(std::move(label)) {}
  [[nodiscard]] double data() const { return _data; }
  [[nodiscard]] double grad() const { return _grad; }
  [[nodiscard]] const std::string &label() const { return _label; }
  void backward();
  static void printDOT(const std::string &filename, Value *value);
  void printDOT(const std::string &filename);

  friend ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs)
  {
    ValuePtr out = std::make_shared<Value>();
    out->_data = lhs->_data + rhs->_data;
    out->_prev.push_back(lhs);
    out->_prev.push_back(rhs);
    out->op = ADD;
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
    ostr << std::format("value: {:.8f}", value->data()) << std::format(" grad: {:.4f}", value->grad()) << std::endl;
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
    out->_backward = [v, out]() { v->_grad += out->_grad * (1.0 - std::tanh(v->_data) * std::tanh(v->_data)); };
    return out;
  }

  friend ValuePtr operator+=(ValuePtr &lhs, const ValuePtr &rhs)
  {
    lhs = lhs + rhs;
    return lhs;
  }

  template<typename T> friend ValuePtr operator+=(ValuePtr &lhs, const T &rhs)
  {
    lhs = lhs + rhs;
    return lhs;
  }

  friend ValuePtr operator-=(ValuePtr &lhs, const ValuePtr &rhs)
  {
    lhs = lhs - rhs;
    return lhs;
  }

  template<typename T> friend ValuePtr operator-=(ValuePtr &lhs, const T &rhs)
  {
    lhs = lhs - rhs;
    return lhs;
  }

  friend ValuePtr operator*=(ValuePtr &lhs, const ValuePtr &rhs)
  {
    lhs = lhs * rhs;
    return lhs;
  }

  template<typename T> friend ValuePtr operator*=(ValuePtr &lhs, const T &rhs)
  {
    lhs = lhs * rhs;
    return lhs;
  }

  friend ValuePtr operator/=(ValuePtr &lhs, const ValuePtr &rhs)
  {
    lhs = lhs / rhs;
    return lhs;
  }

  template<typename T> friend ValuePtr operator/=(ValuePtr &lhs, const T &rhs)
  {
    lhs = lhs / rhs;
    return lhs;
  }
};

using ValuePtr = std::shared_ptr<Value>;