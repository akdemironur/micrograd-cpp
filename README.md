# micrograd-cpp

This repository contains the C++ implementation of [Andrej Karpathy's `micrograd`](https://github.com/karpathy/micrograd). 

## Project Description

`micrograd` is a tiny Autograd engine. It's a minimalist, educational codebase, to understand how backpropagation works in popular deep learning frameworks.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- A modern (C++20) compiler
- CMake

### Installation

1. Clone the repo
    ```
    git clone https://github.com/akdemironur/micrograd-cpp.git
    ```
2. Build the project
    ```
    cd micrograd-cpp
    mkdir build & cd build
    cmake ..
    make
    ```

## Usage

To utilize the `micrograd-cpp` library, it is required to generate `std::shared_ptr` objects (which are typedef'd as `ValuePtr`). Following this, the `backward()` function should be invoked. A simple example:

```cpp
  auto a = std::make_shared<Value>(-4.0);
  auto b = std::make_shared<Value>(2.0);
  auto c = a + b;
  auto d = a * b + pow(b, 3);
  c = c + c + 1;
  c = c + 1 + c + (-a);
  d = d + d * 2 + relu(b + a);
  d = d + 3 * d + relu(b - a);
  auto e = c - d;
  auto f = pow(e, 2);
  auto g = f / 2.0;
  g = g + 10.0 / f;
  std::cout << std::format("{:.4f}", g->data()) << std::endl; 
  g->backward();
  std::cout << std::format("{:.4f}", a->grad()) << std::endl;
  std::cout << std::format("{:.4f}", b->grad()) << std::endl;
```