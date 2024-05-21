import matplotlib.pyplot as plt
import numpy as np

def plot(fx_hist_gd, fx_hist_nm, title):
    plt.plot(range(len(fx_hist_gd)), fx_hist_gd, label="Gradient Descent")
    plt.plot(range(len(fx_hist_nm)), fx_hist_nm, label="Newton Method")
    plt.title(title)
    plt.legend()
    plt.show()

def contour(f, x_hist_gd, x_hist_nm, x_limit, y_limit, title):
    x_p_gd, y_p_gd = zip(*x_hist_gd) if x_hist_gd else ([], [])
    x_p_nm, y_p_nm = zip(*x_hist_nm) if x_hist_nm else ([], [])

    X, Y = np.meshgrid(np.linspace(x_limit[0], x_limit[1], 100), np.linspace(y_limit[0], y_limit[1], 100))
    Z = np.array([[f(np.array([x, y]), False)[0] for x in X[0]] for y in Y[:, 0]])

    plt.figure()
    plt.contour(X, Y, Z, levels=10)
    plt.plot(x_p_gd, y_p_gd, "xr-", label="Gradient Descent")
    plt.plot(x_p_nm, y_p_nm, 'xb-', label="Newton Method")
    plt.legend()
    plt.title(title)
    plt.show()
    
    
def print_iteration_to_console(iteration, current_coord, obj_value):
    print(f"Iteration #{iteration}; x{iteration} = {current_coord}; f(x{iteration}) = {obj_value};")
