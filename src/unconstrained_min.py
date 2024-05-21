import numpy as np
from src.utils import *

class LineSearchMinimization:

    def __init__(self, method) -> None:
        self.method = method
        self.hist_x, self.hist_obj_value = [], []
        
    def calc_step_size(self, x, f, f_x, descent_direction, wolfe_constant=0.01, backtrack_constant=0.5, initial_alpha=1):
        """
        Perform backtracking line search to determine the step size.

        Parameters:
        - x: Current point
        - f: Objective function
        - f_x: Function value at current point
        - descent_direction: Descent direction
        - wolfe_constant: Wolfe condition constant (default: 0.01)
        - backtrack_constant: Backtracking constant (default: 0.5)
        - initial_alpha: Initial step size (default: 1)

        Returns:
        - alpha: Computed step size
        """
        alpha = initial_alpha
        while f(x + alpha * descent_direction, False)[0] > f_x + wolfe_constant * alpha * np.dot(-descent_direction, descent_direction):
            alpha *= backtrack_constant
        return alpha
    
    def minimize_function(self, f, x0, obj_tol, param_tol, max_iter):
        """
        Minimizes a given function using either the Gradient Descent or Newton's method.

        Parameters:
        -----------
        f : callable
            The objective function to be minimized. It should return the objective value, 
            gradient, and (if applicable) Hessian at a given point.
        x0 : array-like
            The initial point from which to start the minimization process.
        obj_tol : float
            The tolerance for the change in the objective function value for convergence.
        param_tol : float
            The tolerance for the change in the parameters (coordinates) for convergence.
        max_iter : int
            The maximum number of iterations allowed.

        Returns:
        --------
        current_coord : ndarray
            The coordinates of the minimum found.
        objective_value : float
            The objective function value at the minimum.
        hist_x : list
            The history of coordinates during the optimization process.
        hist_obj_value : list
            The history of objective function values during the optimization process.
        has_converged : bool
            Whether the optimization algorithm has converged. 
        """
        current_coord = np.array(x0)
        has_converged = False
        objective_value, grad, hessian = f(current_coord, self.method == 'Newton')
        iteration = 0
        print_iteration_to_console(iteration, current_coord, objective_value)
        self.hist_x.append(current_coord)
        self.hist_obj_value.append(objective_value)
        
        if self.method == 'Gradient Descent':
            while not has_converged and iteration < max_iter:
                objective_value, grad, _ = f(current_coord, False)
                search_direction = -grad
                step_size = self.calc_step_size(current_coord, f, objective_value, search_direction)
                next_coord = current_coord + search_direction * step_size
                next_objective_value, grad, _ = f(next_coord, False)

                iteration += 1

                if iteration != 0 and (abs(next_objective_value - objective_value) < obj_tol or np.linalg.norm(next_coord - current_coord) < param_tol):
                    has_converged = True
                    print_iteration_to_console(iteration, current_coord, objective_value)
                    return current_coord, objective_value, self.hist_x, self.hist_obj_value, has_converged

                current_coord = next_coord
                objective_value = next_objective_value
                print_iteration_to_console(iteration, current_coord, objective_value)

                self.hist_x.append(current_coord)
                self.hist_obj_value.append(objective_value)

            return current_coord, objective_value, self.hist_x, self.hist_obj_value, has_converged
        
        if self.method == 'Newton':
            while not has_converged and iteration < max_iter:
                objective_value, grad, hessian = f(current_coord, True)
                search_direction = np.linalg.solve(hessian, -grad)
                step_size = self.calc_step_size(current_coord, f, objective_value, search_direction)
                next_coord = current_coord + search_direction * step_size
                next_objective_value, grad, hessian = f(next_coord, True)

                iteration += 1

                lambda_value = np.matmul(search_direction.T, np.matmul(hessian, search_direction)) ** 0.5
                if iteration != 0 and (0.5 * (lambda_value ** 2) < obj_tol or np.linalg.norm(next_coord - current_coord) < param_tol):
                    has_converged = True
                    print_iteration_to_console(iteration, current_coord, objective_value)
                    return current_coord, objective_value, self.hist_x, self.hist_obj_value, has_converged

                current_coord = next_coord
                objective_value = next_objective_value
                print_iteration_to_console(iteration, current_coord, objective_value)

                self.hist_x.append(current_coord)
                self.hist_obj_value.append(objective_value)
            
            return current_coord, objective_value, self.hist_x, self.hist_obj_value, has_converged
            
            
    
    
