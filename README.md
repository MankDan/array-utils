# array-utils
A lightweight C++ library for optimized array operations leveraging SIMD (AVX and AVX-512).

## Features
- SIMD-optimized count functions for `int`, `char`, `float`, and `double`.
- Automatic fallback to standard algorithms if AVX/AVX-512 is not supported.

## Requirements
- C++17 or higher.
- A compiler with AVX/AVX-512 support.

## Installation
1. Clone the repository:
```
git clone https://github.com/MankDan/array-utils.git
```
2. Include the header file in your project
```
#include "array_utils.h"
```

## Example code
```
#include "array_utils.h"
#include <iostream>

int main() {
    int array[] = {1, 2, 3, 4, 2, 2, 5};
    size_t count = a(rray_utils::countarray, array + 7, 2);
    std::cout << "Count of 2: " << count << std::endl;
    return 0;
}
```

## Contributions
Feel free to submit feedbacks, pull requests for bug fixes, optimizations, or new features. 

Nyaa~ :3
