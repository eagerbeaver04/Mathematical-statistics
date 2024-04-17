import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt


def lab6():
    a = -1.8
    b = 2
    n = 20
    x = np.linspace(a, b, n)
    e = np.random.normal(0, 1, n)
    y = x * 2 + 2 + e
    plt.scatter(x, y)

    plt.show()
    A = np.vstack([x, np.ones(len(x))]).T
    a_sq, b_sq = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f"a_sq={a_sq}\nb_sq={b_sq}")
    plt.scatter(x, y)
    x_p = np.linspace(a, b, 100)
    y_p = x_p * a_sq + b_sq

    plt.plot(x_p, y_p)
    plt.show()

    def err_func(params, x, y):
        return np.sum(np.abs(params[0] + params[1] * x - y))

    a_abs, b_abs = minimize(err_func, [0, 0], args=(x, y)).x
    print(f"a_abs={a_abs}\nb_abs={b_abs}")

    plt.scatter(x, y)
    x_p = np.linspace(a, b, 100)
    y_p = x_p * a_abs + b_abs
    plt.plot(x_p, y_p)
    plt.show()
    y_mod = y.copy()
    y_mod[0] += 10
    y_mod[-1] -= 10
    A_mod = np.vstack([x, np.ones(len(x))]).T
    b_sq_mod, a_sq_mod = np.linalg.lstsq(A_mod, y_mod, rcond=None)[0]
    print(f"a_sq_mod= {a_sq_mod}\nb_sq_mod ={b_sq_mod}")

    def residuals(params):
        return np.abs(params[0] + params[1] * x - y_mod)

    b_abs_mod, a_abs_mod = minimize(err_func, [0, 0], args=(x, y_mod)).x
    print(f"a_abs_mod={a_abs_mod}\nb_abs_mod={b_abs_mod}")
    print(f"a_delta_abs={abs(a_abs_mod - a_abs) / a_abs}\nb_delta_abs={abs(b_abs_mod - b_abs) / b_abs}")
    print(f"a_delta_sq={abs(a_sq_mod - a_sq / a_sq)}\nb_delta_sq={abs(b_sq_mod - b_sq) / b_sq}")
    plt.show()
    plt.scatter(x, y_mod)
    x_p = np.linspace(a, b, 100)
    y_p = x_p * a_sq_mod + b_sq_mod
    plt.plot(x_p, y_p)

    plt.show()
    plt.scatter(x, y_mod)

    x_p = np.linspace(a, b, 100)
    y_p = x_p * a_abs_mod + b_abs_mod
    plt.plot(x_p, y_p)

    plt.show()
