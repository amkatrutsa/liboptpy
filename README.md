# liboptpy

Library with implementations of optimization methods in Python 3

## Installing from source 

- ```git clone https://github.com/amkatrutsa/liboptpy.git```
- ```cd liboptpy```
- ```python setup.py install```

or

```pip install git+https://github.com/amkatrutsa/liboptpy```

## Examples

1. [Unconstrained smooth and non-smooth optimization](./examples/demo_unconstr_solvers.ipynb)
2. [Comparison of projected gradient descent and Frank-Wolfe method](./examples/demo_simple_set.ipynb)

## Available optimization methods

### Unconstrained optimization problem

#### Smooth objective functon
1. Gradient descent
2. Nesterov accelerated gradient descent
3. Newton method and inexact (truncated) Newton method with CG as linear solver
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
2. Inverse number on iteration and scaled by gradient norm version
3. Inverse square root of number of iterationas and scaled by gradient norm version 
4. Backtracking
    - Armijo rule
    - Wolfe rule
    - Strong Wolfe rule
    - Goldstein rule
5. Exact line search for quadratic function

## Contributing

If you find any bugs, please fix them and send pull-request. 
If you want add some enhancement or something new, please open an issue for discussion. 

To send pull-request, you should make the following steps

1. Fork this repository
2. Clone the forked repository
3. Add original repositore as remote one 
4. Create a branch in your local repository with specific name for your changes, e.g. ```bugfix```
5. Switch to this branch
6. Change something that you assume make this repository better
7. Commit your changes in the branch ```bugfix``` with a meaningful comment, e.g. ```Fix typo```
8. Switch to the branch ```master```
9. Pull new commits to the branch ```master``` from this repository, not forked one
10. Switch to branch ```bugfix```
11. Make ```git rebase master``` to take all new commits from original repository to branch ```bugfix```
12. Make push to your forked repository in new remote branch ```bugfix```
13. Send pull-request from your remote branch ```bugfix``` to ```master``` branch of the original repository
