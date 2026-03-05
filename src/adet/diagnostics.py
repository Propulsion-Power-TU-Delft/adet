from collections import namedtuple
import logging
from typing import Generic, TypeAlias, TypeVar, Union

import casadi as cs
from jax import Array
import jax
import numpy as np
from numpy.typing import NDArray

from adet.assembly import CasadiSystem, JaxSystem, SystemAssembler


logger = logging.getLogger(__name__)

NumpyArrays: TypeAlias = Union[Array, NDArray]

EquationBound = namedtuple(
    'EquationSensitivities',
    [
        'equation_id',
        'value',
        'arguments',
    ],
)

FirstNewtonIter = namedtuple(
    'FirstNewtonIter',
    [
        'x0',
        'x1',
        'res0',
        'jac0',
        'hes0',
        'max_nlr',
    ],
)

T = TypeVar('T', bound=SystemAssembler)


class SystemDiagnostics(Generic[T]):
    def __init__(self, system: T, constraints_stack):
        self._arguments = system.free_args
        self._num_span = system.num_span
        self.const_stack = constraints_stack

        if isinstance(system, CasadiSystem):
            self._build_casadi_functions(system)
        elif isinstance(system, JaxSystem):
            self._build_jax_functions(system)

        self._arg_mapping = self._remap_arguments()

    def _build_casadi_functions(self, system: CasadiSystem):
        logger.debug('Building casadi functions')
        args_sym = cs.vertcat(*system.free_args_sym)
        const_values = self.const_stack
        num_args = len(system.free_args_sym)

        # Residual
        logger.debug('Building residual functions')
        res_func = system.make_residual_function()
        res_expr = res_func(args_sym, const_values)

        def res_cas(x):
            # Reshape to column
            x = x.reshape(-1, 1)
            return np.array(
                res_func(x, const_values),
            ).flatten()

        # Jacobian
        jac_expr = cs.jacobian(res_expr, args_sym)
        jac_func = cs.Function('J', [args_sym], [jac_expr])

        def jac_cas(x):
            # partialize
            return np.array(jac_func(x))

        # Hessian
        hes_expr = cs.jacobian(jac_expr, args_sym)
        hes_func = cs.Function('H', [args_sym], [hes_expr])

        # CasADi uses a different convention for derivating the jacobian
        # it is in order of arguments and concatenated instead of
        # being a 3D array with an hessian 2D matrix for each equation
        sequence = []
        for i in range(num_args):
            sequence += [i + j * num_args for j in range(num_args)]

        def hes_cas(x):
            return np.array(hes_func(x))[sequence, :].reshape(
                num_args, num_args, num_args
            )

        self._res_func = res_cas
        self._jac_func = jac_cas
        self._hes_func = hes_cas

    def _build_jax_functions(self, system: JaxSystem):
        res_func = system.make_residual_function()

        def res_flat_jax(x):
            # Reshape to column
            x = x.reshape(-1, 1)
            return res_func(
                x,
                self.const_stack,
            ).flatten()

        jac_jax = jax.jacobian(res_flat_jax)
        hes_jax = jax.hessian(res_flat_jax)

        self._res_func = res_flat_jax
        self._jac_func = jac_jax
        self._hes_func = hes_jax

    def _get_nonlin_args_positions(self):
        """
        Split the arguments into linear and nonlinear arguments.
        The linear arguments are detected as the ones whose corresponding column
        of the Hessian matrix is zero, meaning that they have no influence on the
        Jacobian during the N-R iterations. See ref. 1 for more theory.

        References
        ----------
        1. Casella, Francesco, and Bernhard Bachmann.
        "On the choice of initial guesses for the Newton-Raphson algorithm."
        Applied Mathematics and Computation 398 (2021): 125991.
        """

        logger.debug('Sorting arguments by linearity...')

        probe_var = np.ones(len(self._arguments) * self._num_span)

        hessians = self._hes_func(probe_var)

        # Columns in which the Hessians are nonzero
        # => argument is non-linear
        nonzero_col = np.where(hessians != 0)[-1]

        return set(nonzero_col.tolist())

    def _remap_arguments(self):
        """Remap arguments by linearity"""
        args = self._arguments
        nonlin_arg_indices = self._get_nonlin_args_positions()

        nonlin_args = set()
        for idx in nonlin_arg_indices:
            nonlin_args.add(args[idx // self._num_span])

        lin_args = sorted(set(args) - nonlin_args)

        self._linear_arguments = tuple(lin_args)
        self._nonlinear_arguments = tuple(sorted(nonlin_args))

        logger.debug(f'Linear args are: {", ".join(self._linear_arguments)}')
        logger.debug(f'Nonlinear args are: {", ".join(self._nonlinear_arguments)}')

        args = tuple(self._nonlinear_arguments + self._linear_arguments)
        arg_mapping = np.array([self._arguments.index(a) for a in args])

        return arg_mapping

    @staticmethod
    def _get_original_input(arg_mapping: NDArray, x0_sorted: NumpyArrays):
        """Get the sorted by linearity array and returns the original one"""
        return x0_sorted[np.argsort(arg_mapping)]

    def residuals(self, x0_sorted):
        x0 = self._get_original_input(self._arg_mapping, x0_sorted)
        return self._res_func(x0)

    def jacobian(self, x0_sorted):
        x0 = self._get_original_input(self._arg_mapping, x0_sorted)
        return self._jac_func(x0)[:, self._arg_mapping]

    def hessian(self, x0_sorted):
        x0 = self._get_original_input(self._arg_mapping, x0_sorted)
        return self._hes_func(x0)[:, :, self._arg_mapping]

    def _check_jacobian_singularity(self, x0_sorted):
        PERTURBATION = 1e-5
        jac0 = self.jacobian(x0_sorted)

        if np.linalg.det(jac0) == 0.0:
            logger.warning(
                'Jacobian for initial guess is exactly singular, perturbating...'
            )
            x0_sorted += np.random.normal(scale=PERTURBATION * x0_sorted)
            try:
                jac0 = self._jac_func(x0_sorted)
            except np.linalg.LinAlgError:
                raise ValueError('Singular jacobian matrix')

        return x0_sorted, jac0

    def _first_newt_iteration(self, x0: NumpyArrays) -> FirstNewtonIter:
        x0_sorted = x0[self._arg_mapping]

        x0_sorted, jac0 = self._check_jacobian_singularity(x0_sorted)

        res0 = self.residuals(x0_sorted)
        hes0 = self.hessian(x0_sorted)

        # First N-R iteration
        x1 = x0_sorted + np.linalg.solve(jac0, -res0)

        num_linear = len(self._linear_arguments)

        # Linear part of the jacobian (if any)
        if num_linear > 0:
            # Isolate linear variables
            z0 = x0_sorted[-num_linear:]
            z1 = x1[-num_linear:]
            jac_lin = jac0[:, -num_linear:]
            # NLR = Non-linear residuals
            nlr0 = res0 + jac_lin @ (z1 - z0)
        else:
            nlr0 = res0

        max_nlr = np.linalg.norm(nlr0, np.inf)

        return FirstNewtonIter(
            x0=x0_sorted,
            x1=x1,
            res0=res0,
            jac0=jac0,
            hes0=hes0,
            max_nlr=max_nlr,
        )

    def _check_initial_value_type(self, initial_value: FirstNewtonIter | NumpyArrays):
        if not isinstance(initial_value, FirstNewtonIter):
            first_iter = self._first_newt_iteration(initial_value)
        else:
            first_iter = initial_value

        return first_iter

    def compute_bounding_coeffs(self, initial_value: FirstNewtonIter | NumpyArrays):
        """
        :math:`\\alpha_i` bounding coefficients for equation residuals, see ref. 1

        References
        ----------
        1. Casella, Francesco, and Bernhard Bachmann.
        "On the choice of initial guesses for the Newton-Raphson algorithm."
        Applied Mathematics and Computation 398 (2021): 125991.
        """
        first_iter = self._check_initial_value_type(initial_value)

        x0, x1, res0, jac0, hes0, max_nlr = (
            first_iter.x0,
            first_iter.x1,
            first_iter.res0,
            first_iter.jac0,
            first_iter.hes0,
            first_iter.max_nlr,
        )

        res1 = self.residuals(x1)

        relax_fac = 1.0

        # Execute a relaxed first iteration if necessary
        if np.isnan(res1).any():
            for delta_relax in range(50, 100, 5):
                # Relax from 0.5 to 0.05
                relax_fac = 1.0 - delta_relax / 100

                x1 = x0 + relax_fac * np.linalg.solve(jac0, -res0)
                res1 = self._res_func(x1)

                if not np.isnan(res1).any():
                    break

        return np.abs(
            res1 - (1 - relax_fac) * res0 - 0.5 * (x1 - x0).T @ hes0 @ (x1 - x0)
        ) / (relax_fac**3 * max_nlr)

    def compute_curvatures(
        self,
        initial_value: FirstNewtonIter | NumpyArrays,
    ) -> NDArray:
        first_iter = self._check_initial_value_type(initial_value)
        x0, x1, hes0, max_nlr = (
            first_iter.x0,
            first_iter.x1,
            first_iter.hes0,
            first_iter.max_nlr,
        )

        curvatures = 0.5 * np.abs(hes0) * np.outer(x1 - x0, x1 - x0) / max_nlr

        return abs(curvatures)

    def compute_sensitivities(
        self,
        initial_value: FirstNewtonIter | NumpyArrays,
    ):
        """
        Compute the :math:`\\Sigma` matrix from ref. 1,
        representing the first iteration sensitivity to the initial guess.

        .. math::
            \\Sigma = \\frac{\\partial x_1}{\\partial x_0}

        Returns only the diagonal values :math:`\\sigma_{jj}`, as they are
        invariant to units and meaningful in absolute values. They represent
        the relative variation of the value of argument j w.r.t. to itself
        from iteration 0 to iteration 1.
        For example :math:`\\sigma_{jj} = 50` means that a variation of 1 in
        the argument with index `j` in the initial guess, causes a variation of
        magnitude 50 in the same variable after the first iteration.


        References
        ----------
        1. Casella, Francesco, and Bernhard Bachmann.
        "On the choice of initial guesses for the Newton-Raphson algorithm."
        Applied Mathematics and Computation 398 (2021): 125991.
        """
        first_iter = self._check_initial_value_type(initial_value)

        x0, x1, jac0, hes0 = (
            first_iter.x0,
            first_iter.x1,
            first_iter.jac0,
            first_iter.hes0,
        )

        sens_matrix = -np.linalg.inv(jac0) @ ((x1 - x0) @ hes0)

        # The off-diagonal terms are not indicative
        # and are unit-dependent
        diag_sens = np.diag(sens_matrix)

        return diag_sens

    @property
    def arguments(self):
        return tuple(self._arguments[i] for i in self._arg_mapping)

    # TODO:
    # I want to keep it simple for now, this more user-friendly stuff
    # will have to wait, keep it here for the concepts

    # def identify_critical_arguments(
    #     self,
    #     guess: NumpyArrays,
    #     inclusion_threshold: float = 0.1,
    # ):
    #     """
    #     Parameters
    #     ----------
    #     guess: NDArray
    #         Initial guess to analyze
    #     incl_threshold: float
    #         Threshold above which to include the guess indicators as containing
    #         critical variables
    #
    #     Returns
    #     -------
    #     tuple[dict, dict, dict]
    #         Dictionaries of critical variables for equation bounds, first
    #         iteration sensitivities and curvature respectively
    #     """
    #     first_iter = self._compute_first_newton_iter(guess)
    #
    #     eq_bounds, _ = self.compute_bounding_coeffs(first_iter)
    #     iter_sens = self.compute_sensitivities(first_iter)
    #     curvatures = self.compute_curvatures(first_iter)
    #
    #     # 1 === Equation bounds
    #     max_bound = eq_bounds[0].value
    #
    #     eq_crit_vars = {}
    #     for bound in eq_bounds:
    #         if bound.value < inclusion_threshold * max_bound:
    #             break
    #
    #         for arg in bound.arguments:
    #             if arg in self._nonlinear_arguments:
    #                 # This just keeps the highest value
    #                 # if it is already in the dictionary,
    #                 # otherwise overwrites it
    #                 eq_crit_vars[arg] = max(
    #                     bound.value,
    #                     eq_crit_vars.get(arg, 0.0),
    #                 )
    #
    #     # 2 === Curvatures
    #     max_curv = np.linalg.norm(curvatures.flatten(), np.inf)
    #
    #     # Where are the curvatures greater than the threshold?
    #     crit_eqs, crit_rows, crit_cols = np.where(
    #         curvatures > inclusion_threshold * max_curv
    #     )
    #
    #     num_span = curvatures.shape[-1] // len(self._arguments)
    #
    #     curv_crit_vars = {}
    #     for eq, r, c in zip(crit_eqs, crit_rows, crit_cols):
    #         arg_r = self._arguments[r // num_span]
    #         arg_c = self._arguments[c // num_span]
    #
    #         curv_crit_vars[arg_r] = max(
    #             curvatures[eq, r, c],
    #             curv_crit_vars.get(arg_r, 0.0),
    #         )
    #
    #         curv_crit_vars[arg_c] = max(
    #             curvatures[eq, r, c],
    #             curv_crit_vars.get(arg_c, 0.0),
    #         )
    #
    #     # 3 === First iteration sensitivities
    #     # Calculate the maximum absolute sensitivity across all variables
    #     max_iter_sensitivity = np.linalg.norm(iter_sens, np.inf)
    #
    #     iter_crit_vars = {}
    #
    #     # Iterate through the sensitivity array and identify critical variables
    #     for idx, sensitivity in enumerate(iter_sens):
    #         # Check if the absolute sensitivity of the current variable
    #         # is above the threshold
    #         abs_sens = abs(sensitivity)
    #         if abs_sens > inclusion_threshold * max_iter_sensitivity:
    #             argument = self._arguments[idx // num_span]
    #             # Add the variable to the critical variables dictionary
    #             iter_crit_vars[argument] = max(
    #                 abs_sens,
    #                 iter_crit_vars.get(argument, 0.0),
    #             )
    #
    #     return eq_crit_vars, curv_crit_vars, iter_crit_vars
    #
    # def get_sorted_critical_arguments(
    #     self,
    #     guess: NDArray,
    #     inclusion_threshold: float = 0.1,
    # ) -> list[str]:
    #     critical_variable_dicts = self.identify_critical_arguments(
    #         guess, inclusion_threshold
    #     )
    #     max_criticals = {}
    #
    #     for d in critical_variable_dicts:
    #         for key, val in d.items():
    #             if val > max_criticals.get(key, 0.0):
    #                 max_criticals[key] = val
    #
    #     return sorted(max_criticals, key=lambda k: max_criticals[k], reverse=True)

    # def _explore_argument_bounds(
    #     self,
    #     arg: str,
    #     guess: NDArray,
    #     num_samples: int = 100,
    #     custom_bounds: None | tuple[float, float] = None,
    # ):
    #     """
    #     Explore the bounds of an argument looking for a valid solution
    #     """
    #
    #     def wrapped_fun(x, aux):
    #         return self.res_func(x)
    #
    #     var_type = get_arg_type(arg)
    #
    #     if arg not in self.free_arguments:
    #         raise KeyError(
    #             f'Argument {arg} not recognized in the system, available arguments are '
    #             f'{self.free_arguments}'
    #         )
    #
    #     arg_index = self.free_arguments.index(arg)
    #     arg_units = self._system._arguments_units[arg]
    #     scaling_factor = self.argument_scaling[arg_index]
    #
    #     if custom_bounds:
    #         bounds = custom_bounds
    #     elif var_type in VARIABLE_BOUNDS:
    #         bounds = VARIABLE_BOUNDS[var_type]
    #     else:
    #         logger.warning(
    #             f'Missing exploring bounds for critical variable {var_type} '
    #             f'please provide manual bounds'
    #         )
    #         lower_bound = input(f'INPUT >>> {var_type} lower bound [{arg_units}] = ')
    #         upper_bound = input(f'INPUT >>> {var_type} upper bound [{arg_units}] = ')
    #
    #         bounds = (float(lower_bound), float(upper_bound))
    #
    #     bounds = tuple(val / scaling_factor for val in bounds)
    #
    #     guess_space = np.linspace(*bounds, num=num_samples)
    #
    #     solution = None
    #     for value in guess_space:
    #         logger.info(
    #             f'Trying value {value[0] * scaling_factor}[{arg_units}] for {arg}'
    #         )
    #         new_guess = guess.at[arg_index].set(value)
    #         try:
    #             solution = optx.root_find(
    #                 wrapped_fun,
    #                 optx.BestSoFarLeastSquares(
    #                     optx.LevenbergMarquardt(1e-2, 1e-1),
    #                 ),
    #                 new_guess.flatten(),
    #                 max_steps=50,
    #             )
    #             solution = optx.root_find(
    #                 wrapped_fun,
    #                 optx.Newton(1e-5, 1e-7),
    #                 solution.value,
    #                 max_steps=5,
    #             )
    #             break
    #         except EquinoxRuntimeError:
    #             logger.info('Solver did not converge')
    #             continue
    #
    #     return solution
    #
    # def try_solution_bruteforcing(
    #     self,
    #     guess: NDArray,
    #     aggressiveness: float = 10.0,
    #     samples: int = 20,
    #     custom_bounds: dict[str, tuple[float, float]] = {},
    # ):
    #     """
    #     More aggressive => More variables being explored
    #     """
    #
    #     inclusion_threshold = 1 / aggressiveness
    #
    #     critical_arguments = self.get_sorted_critical_arguments(
    #         guess.flatten(), inclusion_threshold
    #     )
    #
    #     logger.info(f'Exploring solution space for {", ".join(critical_arguments)}')
    #
    #     for arg in critical_arguments:
    #         if arg in custom_bounds:
    #             arg_cust_bounds = custom_bounds[arg]
    #         else:
    #             arg_cust_bounds = None
    #
    #         solution = self._explore_argument_bounds(
    #             arg, guess, samples, arg_cust_bounds
    #         )
    #
    #         if solution is not None:
    #             logger.info('Valid solution found')
    #             return solution.value
