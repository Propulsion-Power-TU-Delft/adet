from inspect import getfullargspec
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import re
from typing import get_args, cast, Self, TYPE_CHECKING
import ast
import inspect
import textwrap

import sympy as sp
import numpy as np

from adet.tools.strings import get_index, verify_string_pattern, get_arg_state
from adet.tools.context import override_operators, suppress_output
from adet.constants import NodeStatesNames

if TYPE_CHECKING:
    from adet.fluid.settings import FluidModel

logger = logging.getLogger(__name__)


class EquationBase(ABC):
    """
    Base Class for defining equations, including argument validation and organization,
    node variable creation and simple storage of the last arguments.

    Supports argument aliasing: allowing the residual function signature to use
    different names than the system-level variable names.
    """

    skip_unit_check: bool = False
    manual_units: tuple[str, ...] = ()

    def __init__(
        self,
        scaling_factor: list[float] | None = None,
        argument_aliases: dict[str, str] | None = None,
    ):
        """
        Parameters
        ----------
        scaling_factor : list[float] | None
            Custom scaling factors for equations
        argument_aliases : dict[str, str] | None
            Mapping from residual function argument names to system variable names.
            Format: {residual_arg_name: system_var_name}

        Example
        -------
            For an ideal gas equation that should work on intermediate state:
            >>> argument_aliases = {
            >>> 'stc_p0': 'stc_p_ss0_0', # pressure at suction side position 0
            >>> 'stc_T0': 'stc_T_ss0_0', # temperature at suction side position 0
            >>> 'stc_rhomass0': 'stc_rhomass_ss0_0',
            >>> ...
            >>> }

            The residual function is defined with standard args
            (stc_p0, stc_T0, ...), but when added to the system,
            these are mapped to intermediate variable names.
        """
        self._argument_aliases = argument_aliases or {}

        # Read arguments from residual signature
        residual_args = getfullargspec(self.residual).args[1:]

        # Apply aliasing: use aliased names if provided, otherwise use original names
        self._arguments: tuple[str, ...] = self._read_and_validate_arguments(
            residual_args,
        )

        # TODO: Move scaling factor
        self.scaling_factor = scaling_factor

        self._num_equations: int | None = None

        # If the unit are not checked, make sure the user added units correclty
        if self.skip_unit_check:
            eq_name = self.__class__.__name__
            if not self.manual_units:
                raise AttributeError(
                    f'Missing manual units for unchecked equation {eq_name}'
                )
            if self.num_equations != len(self.manual_units):
                raise ValueError(
                    f'Mismatch in equation `{eq_name}` between manual '
                    f'units length ({len(self.manual_units)}) {self.manual_units} '
                    f'and number of equations ({self.num_equations})'
                )

    def __call__(self, *args):
        return self.residual(*args)

    @abstractmethod
    def residual(self, *args):
        """
        Expected format for argument is <node_state>_<var_type><index>
        where the index corresponds to the FlowNode in the order
        specified during the class definition. The indices are
        expected to be only one digit.
        """
        raise NotImplementedError

    @property
    def arguments(self):
        """Arguments, in the format of <node_state>_<var_type><index>"""
        return self._arguments

    @property
    def num_equations(self):
        # Maybe add a setter for manually imposing num
        # equations?
        if not self._num_equations:
            try:
                # Avoid printing if fails
                # in particular CoolProp stuff
                with suppress_output():
                    self._num_equations = self._count_equations_arg_inj()
            except Exception:
                self._num_equations = self._count_equations_ast()

        return self._num_equations

    @property
    def num_args(self):
        return len(self._arguments)

    def _count_equations_ast(self):
        """
        Count the number of residual equation using
        abstract syntax trees
        """
        method = self.residual.__func__

        try:
            src = inspect.getsource(method)
            # Remove class indentation
            src = textwrap.dedent(src)
        except (OSError, TypeError):
            raise RuntimeError(
                f'Could not get source code for residual function in'
                f' {self.__class__.__name__}'
            )

        tree = ast.parse(src)
        num_ret = 0  # number of returns

        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                v = node.value
                if isinstance(v, ast.Tuple):
                    num_ret += len(v.elts)
                else:
                    num_ret += 1

        return num_ret

    def _count_equations_arg_inj(self):
        """
        Count how many residual equations are contained
        in this residual formulation by argument
        injection
        """
        num_args = len(self.arguments)
        dummy_args = np.full((num_args, 1), np.nan)
        dummy_res = self.residual(*dummy_args)

        if hasattr(dummy_res, '__len__'):
            num_equations = len(dummy_res)
        else:
            num_equations = 1

        return num_equations

    def _read_and_validate_arguments(self, all_arguments: list[str]):
        """
        Retrieve all the arguments of the residual function and
        apply aliasing if present.

        The aliasing system works as follows:
        1. Residual function has standard argument names (e.g., stc_p0, stc_T0)
        2. If aliases are provided, these are mapped to system variable names
           (e.g., stc_p_ss0_0, stc_T_ss0_0)
        3. The validated_arguments list contains the SYSTEM names (aliased)
        4. The _kwarg_map stores: {system_name: residual_arg_name}
        5. The _inverse_alias_map stores: {residual_arg_name: system_name}
        """
        # The 1 is removed because it is the self instance
        # Careful if residual is changed to a static method
        self._kwarg_map = {}
        self._inverse_alias_map = {}  # For looking up system names from residual args
        validated_arguments = []

        for residual_arg in all_arguments:
            # Check if this argument has an alias
            if residual_arg in self._argument_aliases:
                system_var_name = self._argument_aliases[residual_arg]
                logger.debug(
                    f'Aliasing {residual_arg} -> {system_var_name} '
                    f'in {self.__class__.__name__}'
                )
            else:
                # No alias, use the original name
                system_var_name = residual_arg

            # Validate the SYSTEM variable name (the aliased one)
            validated_system_var = self._validate_argument(system_var_name)
            validated_arguments.append(validated_system_var)

            # Map: system_var -> residual_arg (for calling residual with correct names)
            self._kwarg_map[validated_system_var] = residual_arg

            # Map: residual_arg -> system_var (for reverse lookup)
            self._inverse_alias_map[residual_arg] = validated_system_var

        return tuple(validated_arguments)

    def _validate_argument(self, full_argument: str):
        # Updated pattern to allow digits in variable names (for intermediate states)
        # and multiple trailing digits for node indices
        # Example matches: stc_p0, stc_p_ss0_0, stc_rhomass_ps2_1
        TEMPLATE_PATTERN = r'^[a-z]{3}_[a-zA-Z0-9_]*\d+$'

        # Check for trailing digits (node index)
        # Note: We now allow digits anywhere in the name (e.g., 'ss0' in 'stc_p_ss0_0')
        # Only the trailing digits are interpreted as the node index
        arg_index = re.findall(r'\d+$', full_argument)

        if not arg_index:
            logger.info(f'No index found, assigning to state 0 to {full_argument}')
            full_argument += '0'

        if verify_string_pattern(full_argument, TEMPLATE_PATTERN) is False:
            logger.warning(
                f'Argument {full_argument[:-1]} in equation `{self.__class__.__name__}`'
                f' does not declare a state or has an unrecognized format, '
                f'assigning to `oth` state'
            )
            full_argument = 'oth_' + full_argument

        # Validate the node state
        var_state = get_arg_state(full_argument)
        valid_states = get_args(NodeStatesNames)
        if var_state not in valid_states:
            raise ValueError(
                f'Unknown state for `{full_argument}`, valid states are:\n'
                f'{valid_states}'
            )

        return full_argument

    def to_symbolic(self) -> sp.Expr | str:
        """
        Return a symbolic rendering of the equation

        - Temporarily convert numpy functions to sympy
        - Return class attributes as symbols
        """

        # Add a shape to the symbol class for
        # symbolic representation
        class ShapedSymbol(sp.Symbol):
            shape = (1,)

        class SymbolMaker:
            """
            This overwrites the self class to return the symbols
            self.ratio -> sp.Symbol('ratio')
            """

            def __getattr__(self, name):
                return ShapedSymbol(name)

        # This is so that any attribute that appears
        # in the equation is translated to a symbol
        # in theory this is not used anymore (for jax compatibility)
        dummy_self = SymbolMaker()
        # Recast it as an instance of Self
        dummy_self = cast(Self, dummy_self)

        res_func = self.residual

        # Build the residual function arguments as symbols
        symbolic_args = []
        for arg in self.arguments:
            symbolic_args.append(ShapedSymbol(arg))

        # Substitute numpy with sympy
        symbolic_res = override_operators(res_func, 'numpy', sp)

        try:
            return symbolic_res(*symbolic_args)
        except Exception:
            raise

    def __str__(self):
        return str(self.to_symbolic()) + ' = 0'


# ============================================================================
# COMPOSITE EQUATION PATTERN - For equations with intermediate states
# ============================================================================


@dataclass
class IntermediateState:
    """
    Represents an intermediate thermodynamic state that needs to be computed
    within a CompositeEquation.

    Example: In Denton profile loss, we need p, rho, T at 4 streamwise positions
    along the blade surface. Each position is an IntermediateState.

    Attributes
    ----------
    variables : tuple[str, ...]
        The thermodynamic variables to compute (e.g., ['p', 'rho', 'T'])
    position_id : str
        Unique identifier for this intermediate position (e.g., 'ss_0', 'ps_1')
    parent_state : str
        Which node state this is derived from (e.g., 'stc', 'tot', 'rlt')
    input_variables : tuple[str, ...]
        The input variables needed to compute this state (e.g., ['hmass', 'smass'])
        These are computed by the parent equation before being passed to sub-equations
    """

    variables: tuple[str, ...] = field(default_factory=tuple)
    position_id: str = ''
    parent_state: str = 'stc'  # Default to static state
    input_variables: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Generate argument names for this intermediate state"""
        # These will be the arguments that sub-equations need
        # Format: <parent_state>_<var>_<position_id><node_idx>
        # Example: stc_p_ss0_0 (static pressure at suction side position 0, node 0)
        self._output_args = tuple(
            f'{self.parent_state}_{var}_{self.position_id}' for var in self.variables
        )
        self._input_args = tuple(
            f'{self.parent_state}_{var}_{self.position_id}'
            for var in self.input_variables
        )

    @property
    def output_arguments(self) -> tuple[str, ...]:
        """The arguments that this intermediate state will produce"""
        return self._output_args

    @property
    def input_arguments(self) -> tuple[str, ...]:
        """The arguments needed to compute this intermediate state"""
        return self._input_args


class CompositeEquation(EquationBase):
    """
    Base class for equations that require intermediate thermodynamic state updates
    within the equation itself (not between nodes).

    This enables two modes of operation:
    1. **Analytical mode**: Sub-equations are exposed to SystemAssembler and solved
       as part of the global system
    2. **External mode**: Sub-computations are handled via callbacks (e.g., CasadiEoS)
       within the residual function

    Usage Example
    -------------
    ```python
    class DentonProfileLoss(CompositeEquation):
        def __init__(self, fluid_model: FluidModel, ...):
            # Define intermediate states along blade surface
            self.intermediate_states = [
                IntermediateState(
                    variables=('p', 'rhomass', 'T'),
                    position_id=f'ss{i}',  # suction side position i
                    parent_state='stc',
                    input_variables=('hmass', 'smass'),
                )
                for i in range(4)  # 4 streamwise positions
            ]

            super().__init__(fluid_model=fluid_model, ...)

        def residual(self, ...):
            # Compute velocity distribution along blade
            W_distr_ss = self._build_velocity_profile(...)

            # Compute intermediate enthalpies
            h_distr = rlt_hmass0 - W_distr_ss**2 / 2

            if self._use_analytical_sub_equations:
                # In analytical mode: intermediate p, rho, T are provided
                # as arguments by the system assembler
                p_ss = self._get_intermediate_values('p', 'ss')
                rho_ss = self._get_intermediate_values('rhomass', 'ss')
                T_ss = self._get_intermediate_values('T', 'ss')
            else:
                # In external mode: compute via callbacks
                p_ss, rho_ss, T_ss = self._compute_via_eos_callback(
                    h_distr, stc_smass0
                )

            # Continue with loss computation...
            return r1, r2
    ```

    Integration with SystemAssembler
    ---------------------------------
    When SystemAssembler encounters a CompositeEquation:

    1. Check if fluid model is AnalyticalFluidModel:
       - YES: Call `.get_sub_equations()` to extract all sub-equations
       - Add these sub-equations to the system at the same nodal position
       - The intermediate variables become part of the system's free arguments
       - The main equation residual receives these as normal arguments

    2. Check if fluid model is ExternalFluidModel:
       - NO: The equation handles intermediate states internally via callbacks
       - Only the main equation's residual is added to the system
       - Intermediate computations happen inside the residual function
    """

    def __init__(
        self,
        fluid_model: 'FluidModel',
        scaling_factor: list[float] | None = None,
    ):
        """
        Parameters
        ----------
        fluid_model : FluidModel
            The fluid model to use for intermediate state computations.
            Determines whether analytical sub-equations or external callbacks are used.
        """
        # Import here to avoid circular dependency
        from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel

        self._fluid_model = fluid_model
        self.intermediate_states: list[IntermediateState] = []

        # Determine operation mode
        self._use_analytical_sub_equations = isinstance(
            fluid_model, AnalyticalFluidModel
        )
        self._use_external_callbacks = isinstance(fluid_model, ExternalFluidModel)

        # Storage for sub-equations (only populated in analytical mode)
        self._sub_equations: list[EquationBase] = []

        # Call parent init (this will validate arguments from residual signature)
        super().__init__(scaling_factor)

    def _build_sub_equations(self):
        """
        Build sub-equations for all intermediate states.
        Only called in analytical mode.

        This method should be called after intermediate_states are defined,
        typically in the child class __init__ after super().__init__().

        INTEGRATION NOTE for SystemAssembler:
        =====================================
        In SystemAssembler._add_exact_eqs_of_state(), after the current logic:

        ```python
        def _add_exact_eqs_of_state(self):
            fl_model = self.fluid_settings.model

            if isinstance(fl_model, AnalyticalFluidModel):
                for node_idx, _ in enumerate(self.nodes):
                    # Existing: add base EOS equations
                    [self.add_equation(eq, node_idx) for eq in fl_model.get_equations()]

            # NEW: Check for composite equations and add their sub-equations
            for eq_instance, eq_position in self.equations.items():
                if isinstance(eq_instance, CompositeEquation):
                    if eq_instance._use_analytical_sub_equations:
                        # Get the sub-equations for intermediate states
                        sub_eqs = eq_instance.get_sub_equations()
                        # Add them at the same nodal position as the parent
                        for sub_eq in sub_eqs:
                            self.add_equation(sub_eq, eq_position)
        ```
        """
        if not self._use_analytical_sub_equations:
            return

        from adet.fluid.settings import AnalyticalFluidModel

        assert isinstance(self._fluid_model, AnalyticalFluidModel)

        # Get the base EOS equations from the fluid model
        base_eos_equations = self._fluid_model.get_equations()

        # For each intermediate state, create equation instances
        # that operate on that state's variables
        for intermed_state in self.intermediate_states:
            for base_eq_class in base_eos_equations:
                # Create a new instance of the equation for this intermediate state
                # This will need custom argument remapping to point to intermediate vars
                sub_eq = self._adapt_equation_for_intermediate_state(
                    base_eq_class, intermed_state
                )
                self._sub_equations.append(sub_eq)

    def _adapt_equation_for_intermediate_state(
        self, base_equation: EquationBase, intermed_state: IntermediateState
    ) -> EquationBase:
        """
        Create a modified version of a base EOS equation that operates on
        intermediate state variables instead of node variables.

        Uses argument aliasing: the equation's residual function keeps its original
        signature (e.g., stc_p0, stc_T0, ...), but the system sees aliased names
        (e.g., stc_p_ss0_0, stc_T_ss0_0, ...).

        This approach is clean because:
        1. No need to modify residual functions
        2. No wrapper classes needed
        3. Base equation classes can be reused directly
        4. SystemAssembler automatically handles the mapping

        Example:
        --------
        For IdealStcEos with arguments (stc_p0, stc_T0, stc_rhomass0, ...),
        create an instance with aliases:
            {
                'stc_p0': 'stc_p_ss0_0',
                'stc_T0': 'stc_T_ss0_0',
                'stc_rhomass0': 'stc_rhomass_ss0_0',
                ...
            }

        The residual function is unchanged, but when SystemAssembler:
        - Reads .arguments -> gets ['stc_p_ss0_0', 'stc_T_ss0_0', ...]
        - Builds kwarg_map -> maps system vars to residual args
        - Calls residual -> uses original names from signature
        """
        # Build the alias mapping for this intermediate state
        # We need to map each standard argument to its intermediate equivalent

        # Get the original equation's arguments (from its residual signature)
        original_args = getfullargspec(base_equation.residual).args[1:]

        # Build the alias dictionary
        aliases = {}
        for orig_arg in original_args:
            # Parse the original argument: e.g.,
            # 'stc_p0' -> state='stc', var='p', idx='0'
            # We need to replace 'var' with 'var_position_id'

            # Extract parts using the existing parsing logic
            arg_state = get_arg_state(orig_arg)  # 'stc', 'tot', 'rlt', or 'oth'
            arg_idx = get_index(orig_arg)  # The node index (0, 1, etc.)

            # Remove state prefix and index to get just the variable name
            # e.g., 'stc_p0' -> 'p'
            var_part = orig_arg.replace(f'{arg_state}_', '').rstrip('0123456789')

            # Check if this variable should be remapped for this intermediate state
            if var_part in intermed_state.variables:
                # This is an output variable of the intermediate state
                # Map to: <state>_<var>_<position_id><idx>
                aliased_name = (
                    f'{arg_state}_{var_part}_{intermed_state.position_id}{arg_idx}'
                )
                aliases[orig_arg] = aliased_name

            elif var_part in intermed_state.input_variables:
                # This is an input variable needed to compute the intermediate state
                # Also needs to be aliased
                aliased_name = (
                    f'{arg_state}_{var_part}_{intermed_state.position_id}{arg_idx}'
                )
                aliases[orig_arg] = aliased_name

            # else: keep original name (e.g., constants like cp, cv)

        # Create a new instance of the base equation with aliases
        # Note: We create a NEW instance, not modify the existing one
        adapted_eq = base_equation.__class__(
            scaling_factor=base_equation.scaling_factor,
            argument_aliases=aliases,
        )

        logger.debug(
            f'Adapted {base_equation.__class__.__name__} for intermediate state '
            f'{intermed_state.position_id} with aliases: {aliases}'
        )

        return adapted_eq

    def get_sub_equations(self) -> list[EquationBase]:
        """
        Return all sub-equations for intermediate states.
        Only relevant in analytical mode.

        Returns
        -------
        list[EquationBase]
            Sub-equations that define intermediate thermodynamic states.
            Empty list if using external callbacks.
        """
        if not self._use_analytical_sub_equations:
            return []

        if not self._sub_equations:
            self._build_sub_equations()

        return self._sub_equations

    def _get_intermediate_values(self, variable: str, position_prefix: str) -> tuple:
        """
        Helper to extract intermediate state values from arguments.
        Only used in analytical mode when intermediate values are provided as args.

        This would be called from the residual function to get intermediate
        pressure, density, temperature, etc.

        Parameters
        ----------
        variable : str
            Variable name (e.g., 'p', 'rhomass', 'T')
        position_prefix : str
            Position identifier (e.g., 'ss' for suction side)

        Returns
        -------
        tuple
            Values for this variable at all positions matching the prefix

        IMPLEMENTATION NOTE:
        ===================
        This requires the residual function to have these intermediate variables
        in its signature. The SystemAssembler will need to recognize these
        special arguments and provide them from the solution vector.

        Example: If intermediate_states define positions ss0, ss1, ss2, ss3,
        and we ask for ('p', 'ss'), this should return:
        (stc_p_ss0_0, stc_p_ss1_0, stc_p_ss2_0, stc_p_ss3_0)

        These would need to be in the residual signature as:
        def residual(self, ..., stc_p_ss0_0, stc_p_ss1_0, stc_p_ss2_0, ...):
        """
        # This would extract from self._last_args or similar
        # Placeholder implementation
        raise NotImplementedError(
            'Intermediate value extraction requires tracking of argument values'
        )

    @property
    def num_equations(self):
        """
        Total number of equations including sub-equations in analytical mode.

        INTEGRATION NOTE for SystemAssembler:
        =====================================
        When counting equations, SystemAssembler should:
        - In analytical mode: Count parent + all sub-equations separately
          (they're added as separate equations via get_sub_equations())
        - In external mode: Only count the parent equation
          (sub-computations are internal)

        The current implementation just counts the parent equation's residuals,
        which is correct since sub-equations are added separately.
        """
        return super().num_equations

    @property
    def arguments(self):
        """
        All arguments including intermediate state variables in analytical mode.

        INTEGRATION NOTE for SystemAssembler:
        =====================================
        When building the system in analytical mode, the intermediate variables
        will be automatically picked up because:
        1. They appear in the residual function signature
        2. EquationBase._read_and_validate_arguments() processes them
        3. They follow the naming convention: <state>_<var>_<position><idx>

        The SystemAssembler's argument detection will find them and create
        corresponding variables in the nodes (via node.create_vars()).

        The intermediate state variables will become part of the system's
        free_args (assuming they're not constrained).
        """
        return super().arguments


# ============================================================================
# INTEGRATION SUMMARY FOR SYSTEMASSEMBLER
# ============================================================================
"""
To fully integrate CompositeEquation into the system, modifications are needed
in SystemAssembler (assembly.py). Here's a summary of the key integration points:

1. **In SystemAssembler._add_exact_eqs_of_state() [Line ~360]**
   ---------------------------------------------------------------
   After adding base EOS equations for AnalyticalFluidModel, check for
   CompositeEquation instances and extract their sub-equations:

   ```python
   def _add_exact_eqs_of_state(self):
       fl_model = self.fluid_settings.model

       # Existing: Add base EOS equations to all nodes
       if isinstance(fl_model, AnalyticalFluidModel):
           for node_idx, _ in enumerate(self.nodes):
               [self.add_equation(eq, node_idx) for eq in fl_model.get_equations()]

       # NEW: Add sub-equations from CompositeEquations
       composite_equations = [
           (eq, pos) for eq, pos in self.equations.items()
           if isinstance(eq, CompositeEquation)
       ]

       for comp_eq, eq_position in composite_equations:
           if comp_eq._use_analytical_sub_equations:
               # Extract and add sub-equations at same nodal position
               sub_equations = comp_eq.get_sub_equations()
               for sub_eq in sub_equations:
                   self.add_equation(sub_eq, eq_position)
   ```

2. **In SystemAssembler._build_equations_of_state() [Line ~840]**
   ----------------------------------------------------------------
   The current logic handles ExternalFluidModel by creating CasadiEoS callbacks
   for discarded thermodynamic arguments. This should continue to work as-is
   because:
   - CompositeEquations in external mode handle their own callbacks internally
   - They don't expose intermediate variables as system arguments
   - No special handling needed

   However, you may want to add a check to skip intermediate variables:
   ```python
   def _get_discarded_thermo_args(self):
       all_discarded = (
           set(self._declared_arguments) - set(self.constraints) - set(self.free_args)
       )

       # NEW: Filter out intermediate state variables from composite equations
       # (they're handled internally, not through the main EOS callback system)
       for eq in self.equations:
           if isinstance(eq, CompositeEquation):
               if eq._use_analytical_sub_equations:
                   # In analytical mode, intermediate vars are system variables,
                   # so they'll be in free_args (not discarded)
                   pass
               else:
                   # In external mode, we might want to explicitly exclude
                   # any intermediate variable patterns from discarded args
                   # (though they shouldn't appear in declared_arguments anyway)
                   pass

       # ... continue with existing logic
   ```

3. **Argument Naming Convention for Intermediate States**
   -------------------------------------------------------
   Intermediate variables follow the pattern:
       <state>_<var>_<position_id><node_idx>

   Examples:
       - stc_p_ss0_0  (static pressure at suction side position 0, node 0)
       - stc_rhomass_ps2_1  (density at pressure side position 2, node 1)

   This is compatible with EquationBase._validate_argument() because:
   - Starts with valid state prefix ('stc', 'tot', 'rlt')
   - Contains valid variable name
   - Ends with single digit node index
   - The position_id is just part of the variable name

   The regex pattern in _validate_argument might need adjustment:
   ```python
   TEMPLATE_PATTERN = r'^[a-z]{3}_[a-zA-Z_]*\\d{1}$'
   ```
   This already allows underscores in the variable name, so 'p_ss0' is valid.

4. **Testing Strategy**
   --------------------
   To test this implementation:

   a) Create a simple CompositeEquation with known analytical solution:
      ```python
      class TestCompositeEq(CompositeEquation):
          def __init__(self, fluid_model):
              self.intermediate_states = [
                  IntermediateState(
                      variables=('p', 'T'),
                      position_id='int0',
                      parent_state='stc',
                      input_variables=('hmass', 'smass'),
                  )
              ]
              super().__init__(fluid_model)

          def residual(self, stc_p0, stc_p_int0_0, ...):
              # Use intermediate pressure in computation
              return stc_p0 - stc_p_int0_0  # Simple equality test
      ```

   b) Test with IdealGasModel (analytical mode)
   c) Test with ExternalFluidModel (callback mode)
   d) Verify that both produce same results for same conditions

5. **Key Challenges to Address**
   ------------------------------
   a) **Argument remapping for sub-equations**: The
      _adapt_equation_for_intermediate_state() method needs careful implementation.
      Wrapper pattern seems most flexible.

   b) **Spanwise dimension handling**: Intermediate states exist at each spanwise
      station, so the sub-equations need to handle the same spanwise dimension
      as the parent equation.

   c) **Initial guess generation**: SystemAssembler.get_initial_guess() needs to
      provide reasonable guesses for intermediate variables. These can often be
      interpolated from inlet/outlet values.

   d) **Symbolic representation**: The to_symbolic() method for CompositeEquation
      should probably show the main equation only, with a note about sub-equations.

6. **Alternative: Simpler Approach for Denton Loss**
   -------------------------------------------------
   If the full CompositeEquation pattern proves too complex, a simpler alternative
   specific to DentonProfileLoss:

   ```python
   class DentonProfileLoss(EquationBase):
       def __init__(self, fluid_model, ...):
           self._fluid_model = fluid_model

           # In analytical mode: create explicit intermediate variables
           # as regular equation arguments
           if isinstance(fluid_model, AnalyticalFluidModel):
               self._use_explicit_intermediates = True
               # The residual signature includes all intermediate vars
           else:
               self._use_explicit_intermediates = False
               # Create CasadiEoS callbacks as before

           super().__init__(...)

       def residual(self, ..., stc_p_ss0_0=None, stc_p_ss1_0=None, ...):
           # Optional args for intermediate states (None in external mode)
           if self._use_explicit_intermediates:
               p_ss = [stc_p_ss0_0, stc_p_ss1_0, ...]
           else:
               p_ss = self._eos_callback(h_distr, s_distr)[0]
   ```

   Then manually add the IdealGasModel equations for each intermediate position
   in the component setup, not in the equation itself.

   This avoids the complexity of CompositeEquation but requires more manual setup.
"""


# ============================================================================
# ARGUMENT ALIASING - DETAILED USAGE EXAMPLES
# ============================================================================
"""
The argument aliasing feature enables equations to be reused with different
variable names without modifying their residual functions. This is the key
mechanism that makes CompositeEquation work elegantly.

Example: Using Aliasing to Adapt IdealStcEos for Intermediate State
--------------------------------------------------------------------
```python
from adet.equations.ideal_gas import IdealStcEos

# Standard usage
standard_eq = IdealStcEos()
# arguments: ('stc_p0', 'stc_T0', 'stc_rhomass0', ...)

# Adapted for intermediate state at suction side position 0
aliases = {
    'stc_p0': 'stc_p_ss0_0',
    'stc_T0': 'stc_T_ss0_0',
    'stc_rhomass0': 'stc_rhomass_ss0_0',
    # ... all thermodynamic variables
}
intermediate_eq = IdealStcEos(argument_aliases=aliases)
# arguments: ('stc_p_ss0_0', 'stc_T_ss0_0', 'stc_rhomass_ss0_0', ...)

# The residual function code is UNCHANGED
# SystemAssembler handles the mapping automatically
```

How It Works:
-------------
1. EquationBase reads residual signature: (stc_p0, stc_T0, ...)
2. Applies aliases to get system names: (stc_p_ss0_0, stc_T_ss0_0, ...)
3. SystemAssembler sees the aliased names in eq.arguments
4. Builds kwarg_map: {'stc_p_ss0_0': 'stc_p0', ...}
5. Calls residual with positional args (names don't matter!)

Key Benefits:
-------------
- No code duplication (reuse existing equation classes)
- Type-safe (original signatures preserved)
- Automatic (SystemAssembler handles everything)
- Flexible (same class for nodes AND intermediate states)
"""
