# liboptpy

Python library with implementations of optimization methods

## Installing from source 

- ```git clone https://github.com/amkatrutsa/liboptpy.git```
- ```cd liboptpy```
- ```python setup.py install```

## Examples

1. [Unconstrained smooth and non-smooth optimization](./demo_unconstr_solvers.ipynb)
2. Constrained optimization

## Available optimization methods

### Unconstrained optimization problem

#### Smooth objective functon
1. Gradient descent
2. Nesterov accelerated gradient descent
3. Newton method
4. Conjugate gradient method
    - for convex quadratic function
    - for non-quadratic function (Fletcher-Reeves method)
5. Barzilai-Borwein method

#### Non-smooth objective function

1. Subgradient method
2. Dual averaging method

### Constrained optimization problem

1. Projected gradient method
2. Frank-Wolfe method
3. Primal barrier method 

### Available step size

1. Constant
2. Backtracking
    - Armijo rule
    - Wolfe rule
    - Strong Wolfe rule
    - Goldstein rule
3. Exact line search for quadratic function
