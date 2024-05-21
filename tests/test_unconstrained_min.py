import sys
import os
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils import contour, plot
from src.unconstrained_min import LineSearchMinimization
from examples import Qi, Qii, Qiii, rosenbrock, linear, exp

class TestLineSearchMinimization(unittest.TestCase):
    x0 = np.array([1, 1])
    x0_rosenbrock = np.array([-1, 2], dtype=np.float64)
    gd_solver = LineSearchMinimization("Gradient Descent")
    newton_solver = LineSearchMinimization("Newton")

    def run_test(self, f, obj_tol, param_tol, max_iter, x0, xlim, ylim, contour_title, plot_title, use_newton=True):
        self.gd_solver.minimize_function(f, x0, obj_tol, param_tol, max_iter)
        if use_newton:
            self.newton_solver.minimize_function(f, x0, 0.001, 0.001, max_iter)
            contour(f, self.gd_solver.hist_x, self.newton_solver.hist_x, xlim, ylim, contour_title)
            plot(self.gd_solver.hist_obj_value, self.newton_solver.hist_obj_value, plot_title)
        else:
            contour(f, self.gd_solver.hist_x, [], xlim, ylim, contour_title)
            plot(self.gd_solver.hist_obj_value, [], plot_title)

    def test1(self):
        self.run_test(Qi, 1e-8, 1e-12, 100, self.x0,
                      [-2.5, 2.5], [-2.5, 2.5], "Contour (circles) $x_{1}^{2} + x_{2}^{2}$", "Plot (circles) $x_{1}^{2} + x_{2}^{2}$")

    def test2(self):
        self.run_test(Qii, 1e-8, 1e-12, 100, self.x0,
                      [-2.5, 2.5], [-2.5, 2.5], "Contour (axis aligned ellipses) $x_{1}^{2} + 100x_{2}^{2}$", "Plot (axis aligned ellipses) $x_{1}^{2} + 100x_{2}^{2}$")

    def test3(self):
        self.run_test(Qiii, 1e-8, 1e-12, 100, self.x0,
                      [-2.5, 2.5], [-2.5, 2.5], "Contour (rotated ellipses) $100x_{1}^{2} + x_{2}^{2}$", "Plot (rotated ellipses) $100x_{1}^{2} + x_{2}^{2}$")

    def test4(self):
        self.run_test(rosenbrock, 1e-8, 1e-12, 10000, self.x0_rosenbrock,
                      [-2.5, 2.5], [-2.5, 2.5], "Contour Rosenbrock 2D", "Plot Rosenbrock 2D")

    def test5(self):
        self.run_test(linear, 1e-8, 1e-12, 100, self.x0,
                      [-200, 2], [-200, 2], "Contour $f(x)=a^{T}x$", "Plot $f(x)=a^{T}x$", use_newton=False)

    def test6(self):
        self.run_test(exp, 1e-8, 1e-12, 100, self.x0,
                      [-1.3, 2.5], [-3, 3], "Contour $e^{x_{1} + 3x_{2} - 0.1} + e^{x_{1} - 3x_{2} - 0.1} + e^{-x_{1} - 0.1}$", 
                      "Plot $e^{x_{1} + 3x_{2} - 0.1} + e^{x_{1} - 3x_{2} - 0.1} + e^{-x_{1} - 0.1}$")


if __name__ == "__main__":
    unittest.main()
