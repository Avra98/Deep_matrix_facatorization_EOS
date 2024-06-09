import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, hessian, lambdify
import argparse

def gradient_descent(alpha, initial_point, iterations, power_N):
    x, y = symbols('x y')
    f = (x * (y ** power_N) - 3) ** 2

    f_grad = [diff(f, var) for var in (x, y)]
    f_hessian = hessian(f, (x, y))

    func = lambdify((x, y), f, 'numpy')
    grad = lambdify((x, y), f_grad, 'numpy')
    hessian_func = lambdify((x, y), f_hessian, 'numpy')

    points = [initial_point]
    losses = [func(*initial_point)]
    hessians = [hessian_func(*initial_point)]

    current_point = np.array(initial_point, dtype=float)
    for _ in range(iterations):
        current_grad = np.array(grad(*current_point), dtype=float)
        current_point -= alpha * current_grad
        points.append(current_point.copy())
        losses.append(func(*current_point))
        hessians.append(hessian_func(*current_point))

    points = np.array(points)
    losses = np.array(losses)
    hessian_max_eigenvalues = [np.max(np.linalg.eigvals(h)) for h in hessians]

    return points, losses, hessian_max_eigenvalues

def main():
    parser = argparse.ArgumentParser(description='Gradient Descent with Argument Parser')
    parser.add_argument('--alpha', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--initial_x', type=float, default=0.0, help='Initial x-coordinate')
    parser.add_argument('--initial_y', type=float, default=0.2, help='Initial y-coordinate')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--power_N', type=int, default=3, help='Power of y in the function')

    args = parser.parse_args()

    alpha = args.alpha
    initial_point = np.array([args.initial_x, args.initial_y], dtype=float)
    iterations = args.iterations
    power_N = args.power_N

    points, losses, hessian_max_eigenvalues = gradient_descent(alpha, initial_point, iterations, power_N)

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(f'Gradient Descent: Function: (x * (y^{power_N}) - 3)^2, lr = {alpha}, (x0, y0) = ({args.initial_x}, {args.initial_y})', fontsize=16, fontweight='bold')

    ax[0].plot(points[:, 0], points[:, 1], 'o-', markersize=8)  # Increase marker size
    ax[0].set_title('Path of (x, y)', fontsize=18)
    ax[0].set_xlabel('x', fontsize=15)
    ax[0].set_ylabel('y', fontsize=15)

    # Calculate and print the specified values
    print(np.abs(power_N*(points[-1,0])**2-(points[-1,1])**2))
    print(points[-1,0], points[-1,1], ((3/(power_N)**(power_N/2)))**(1/(power_N+1)), (3**(1/(power_N+1)))*(power_N**(1/(2*(power_N+1)))))

    # Code to add a dotted line for 2x = y
    x_vals = np.array(ax[0].get_xlim())  # Get the x-axis limits
    y_vals =  x_vals * np.sqrt(power_N)  # Calculate y values as 2x
    ax[0].plot(x_vals, y_vals, 'k:', label=f'({power_N}^0.5)x = y')
    ax[0].legend(fontsize=14)

    ax[1].plot(losses, 'r-', markersize=4)  # Increase marker size
    ax[1].set_title('Loss value over iterations', fontsize=14)
    ax[1].set_xlabel('Iteration', fontsize=15)
    ax[1].set_ylabel('Loss', fontsize=15)
    

    ax[2].plot(hessian_max_eigenvalues, 'g-', markersize=4)  # Increase marker size
    ax[2].set_title('Sharpness', fontsize=14)
    ax[2].set_xlabel('Iteration', fontsize=15)
    ax[2].set_ylabel('Maximum Eigenvalue of Hessian', fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the suptitle
    plt.savefig(f'scalar_gd_{alpha}_{args.initial_y}_{args.power_N}.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
