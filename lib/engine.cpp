#include "engine.h"


std::pair<std::string, std::string> opDot(OpType *op)
{
  std::string opName = "OP" + std::to_string(reinterpret_cast<std::uintptr_t>(op));
  return { opName, std::format("{} [label=\"{}\"]", opName, opToString(*op)) };
}

std::pair<std::string, std::string> valueDot(Value *v)
{
  std::string valueName = "VAL" + std::to_string(reinterpret_cast<std::uintptr_t>(v));
  std::string nameField;
  if (!v->label().empty()) { nameField = std::format("|{}", v->label()); }
  return { valueName,
    std::format(
      R"({} [label="{{val: {:.4f}|grad: {:.4f}{}}}" shape="record"]; )", valueName, v->data(), v->grad(), nameField) };
}
std::string opToString(OpType op)
{
  switch (op) {
  case ADD:
    return "+";
  case MUL:
    return "*";
  case EXP:
    return "exp";
  case POW:
    return "pow";
  case RELU:
    return "relu";
  case SUB:
    return "sub";
  case DIV:
    return "div";
  case TANH:
    return "tanh";
  default:
    return "none";
  }
}

std::vector<Value *> Value::topo()
{
  std::vector<Value *> t{};
  std::set<Value *> s{};
  buildTopo(this, t, s);
  return t;
}

void Value::buildTopo(Value *v, std::vector<Value *> &topo, std::set<Value *> &seen)
{
  if (seen.find(v) != seen.end()) { return; }
  seen.insert(v);
  for (const auto &p : v->_prev) { buildTopo(p.get(), topo, seen); }
  topo.push_back(v);
}
void Value::backward()
{
  auto topo_order = topo();
  for (auto &it : topo_order) {
    if (it != nullptr) { it->_grad = 0; }
  }
  _grad = 1.0;
  for (auto &it : std::ranges::reverse_view(topo_order)) {
    if (it != nullptr) { it->_backward(); }
  }
}

void Value::printDOT(const std::string &filename, Value *value)
{
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Error: Unable to open file!" << std::endl;
    return;
  }

  std::function<void(std::ofstream &, Value *)> traverseAndPrintDOT = [&](std::ofstream &out, Value *v) {
    auto [valueName, valueDotStr] = valueDot(v);
    out << valueDotStr << std::endl;
    if (!v->_prev.empty()) {
      auto [opName, opDotStr] = opDot(&v->op);

      out << opDotStr << std::endl;
      out << opName << " -> " << valueName << ";" << std::endl;
      for (const auto &p : v->_prev) {
        auto [prevValueName, prevValueDotStr] = valueDot(p.get());
        out << prevValueDotStr << std::endl;
        out << prevValueName << " -> " << opName << ";" << std::endl;
      }
    }
  };
  auto topo = value->topo();
  outFile << "digraph G {" << std::endl;
  for (const auto &v : topo) { traverseAndPrintDOT(outFile, v); }
  outFile << "}" << std::endl;
  outFile.close();
  std::cout << "DOT representation written to " << filename << std::endl;
}

void Value::printDOT(const std::string &filename) { printDOT(filename, this); }