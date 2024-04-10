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

To utilize the `micrograd-cpp` library, it is required to generate `std::shared_ptr` objects (which are typedef'd as `ValuePtr`). Following this, the `backward()` function should be invoked. A simple example can be found in `src/micrograd.cpp`.

