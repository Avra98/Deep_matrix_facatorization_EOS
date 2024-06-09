import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, hessian, lambdify
import argparse

def gradient_descent(alpha, initial_point,s, iterations):
    x, y1, y2, y3 = symbols('x y1 y2 y3')
    f = 0.5*(x * y1 * y2 * y3 - s) ** 2

    f_grad = [diff(f, var) for var in (x, y1, y2, y3)]
    f_hessian = hessian(f, (x, y1, y2, y3))

    func = lambdify((x, y1, y2, y3), f, 'numpy')
    grad = lambdify((x, y1, y2, y3), f_grad, 'numpy')
    hessian_func = lambdify((x, y1, y2, y3), f_hessian, 'numpy')

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

    distances = {
        'x_y1': np.abs(points[:, 0] - points[:, 1]),
        'x_y2': np.abs(points[:, 0] - points[:, 2]),
        'x_y3': np.abs(points[:, 0] - points[:, 3])
    }

    return points, losses, hessian_max_eigenvalues, distances

def main():
    parser = argparse.ArgumentParser(description='Gradient Descent with Argument Parser')
    parser.add_argument('--alpha', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--initial_y', type=float, default=0.2, help='Initial y-coordinate for y1, y2, and y3')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of iterations')
    parser.add_argument('--s', type=float, default=8, help='target singular value')

    args = parser.parse_args()

    alpha = args.alpha
    initial_point = np.array([0.0, args.initial_y, args.initial_y, args.initial_y], dtype=float)
    iterations = args.iterations
    s = args.s

    points, losses, hessian_max_eigenvalues, distances = gradient_descent(alpha, initial_point, s, iterations)

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(f'Gradient Descent: Function: 0.5*(x * y1 * y2 * y3 - s)^2, lr = {alpha}, s={s}, (x0, y0) = (0, {args.initial_y})', fontsize=20, fontweight='bold')

    ax[0].plot(points[:, 0], points[:, 1], 'o', markersize=6, label='x vs y1')
    ax[0].plot(points[:, 0], points[:, 0], 'r--', label='x = y reference line', linewidth=2)
    ax[0].set_title('x vs y1', fontsize=20)
    ax[0].set_xlabel('x', fontsize=17)
    ax[0].set_ylabel('y1', fontsize=17)
    ax[0].legend(fontsize=16)
    ax[0].tick_params(axis='both', which='major', labelsize=14)

    ax[1].plot(losses, 'r-', markersize=4)
    ax[1].set_title('Loss value over iterations', fontsize=20)
    ax[1].set_xlabel('Iteration', fontsize=17)
    ax[1].set_ylabel('Loss', fontsize=17)
    ax[1].tick_params(axis='both', which='major', labelsize=14)

    ax[2].plot(hessian_max_eigenvalues, 'g-', markersize=4)
    ax[2].set_title('Sharpness', fontsize=20)
    ax[2].set_xlabel('Iteration', fontsize=17)
    ax[2].set_ylabel('Maximum Eigenvalue of Hessian', fontsize=17)
    ax[2].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'scalar_gd_decoupled_{s}.png')
    #plt.savefig(f'scalar_gd_decoupled_{s}.svg')
    plt.show()
    plt.close()

    # Print the distances |x - y1|, |x - y2|, |x - y3| at the final point
    print("Final distances:")
    print(f"|x - y1|: {distances['x_y1'][-1]}")
    print(f"|x - y2|: {distances['x_y2'][-1]}")
    print(f"|x - y3|: {distances['x_y3'][-1]}")

if __name__ == '__main__':
    main()
