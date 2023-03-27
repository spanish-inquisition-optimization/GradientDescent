from core.gradient_descent import *
from core.visualizer import *

# ax^2 + bxy + cy^2 + dx + ey
def create_quadratic(a, b, c, d, e):
    return lambda x: a * x[0] ** 2 + b * x[0] * x[1] + c * x[1] ** 2 + d * x[0] + e * x[1]

# ax^2 + bxy + cy^2 + dx + ey
def create_quadratic_derivative(a, b, c, d, e):
    return lambda x: np.array([2 * a * x[0] + b * x[1] + d, 2 * c * x[1] + b * x[0] + e])

def analyze_quadratic(roi, x0, fixed_steps, bin_iters, fib_iters, a, b, c, d, e):
    f = create_quadratic(a, b, c, d, e)
    df = create_quadratic_derivative(a, b, c, d, e)

    def visualize_optimizer_with(linear_search):
        visualize_optimizing_process(f, roi, np.array(gradient_descent(f, df, x0, linear_search, lambda f, points: len(points) > 20)))

    print("Function plot:")
    visualize_function_3d(f, roi)

    for step in fixed_steps:
        print(f"Optimizing with fixed step = {step}:")
        visualize_optimizer_with(fixed_step_search(step))

    print("Optimizing with binary search")
    visualize_optimizer_with(bin_search)

    print(f"Optimizing with binary search limited by {bin_iters} iterations:")
    visualize_optimizer_with(bin_search_with_iters(bin_iters))

    print("Optimizing with golden ration")
    visualize_optimizer_with(golden_ratio_search)

    print(f"Optimizing with fibonacci search limited by {fib_iters} iterations:")
    visualize_optimizer_with(fibonacci_search(fib_iters))

