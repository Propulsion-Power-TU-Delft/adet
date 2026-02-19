import logging
from pint import Quantity
import matplotlib.pyplot as plt
import numpy as np

from adet.solution import solve_root_problem
from adet.assembly import CasadiSystem
from adet.components import BladeRow
from adet.components.blade_row import VanelessDiffuser, plot_from_nodes
from adet.components.connections import Inlet, Shaft
from adet.components.network import ComponentNetwork

from adet.equations.definitions import IsentropicProperties, EffectiveBladeNumber
from adet.equations.geometrical import (
    MeridionalUniform,
    MinimalCamberLine,
    MeridionalVariable,
)
from adet.equations.nondimensional import (
    WorkCoefficient,
    TotalTotalPressureRatio,
    TotalTotalCompressionEfficiency,
)
from adet.fluid.settings import AnalyticalFluidModel, ExternalFluidModel, FluidSettings
from adet.fluid.symbolic_eos import IdealGasState
from adet.losses.basic import (
    PercTotalPressureLoss,
    PercentageEntropyLoss,
    ZeroDeviation,
)

from adet.losses.compressors import (
    BackstromSlip,
    BladeLoadingCoppage,
    ClearanceJansen,
    LossAdder,
    SkinFrictionJansen,
)
from adet.registries import GuessRegistry, VariableBoundsRegistry
from adet.tools.coolprop_utils import DebugAbstractState
from adet.tools.interpolation import resample_linear
from adet.tools.loggers import setup_logger

logger = logging.Logger(__name__)
setup_logger(logger)

# This makes the missing guesses default to 1
_bounds_reg = VariableBoundsRegistry()
_bounds_reg.reset()
_bounds_reg.from_dict(
    {
        'Vm': (10.0, 500.0),
        'U': (0, 500),
        'beta': (-1.5, 0.0),
        'relmach': (0.0, 1.04),
        # 'delta_hmass_.*': (10.0, 1e5),
        # 'delta_hmass_loading': (10.0, 1e4),  # This tends to diverge, bound it
    }
)

_greg = GuessRegistry()
_greg.reset()
_greg.from_dict(
    {
        'beta': -0.5,
        'gamma_pv': 1.4,
    }
)
_greg.set_fallback_value(0.5)  # Missing values defaults to 0.5

NUM_SPAN = 11
PLOTS = True
ENABLE_LOSSES = False
RUN_MULTI = True
# +++ Shaftskin_omega0 (node 0) is unknown,
shaft = Shaft(
    omega=Quantity(21000, 'rpm'),
    is_constrained=True,
)

casing = Shaft(
    omega=Quantity(0, 'rpm'),
    is_constrained=True,
)

# +++ Fluid settings
fluid_model_real = ExternalFluidModel(
    DebugAbstractState('HEOS', 'Air'),  # This just counts the number of updates
)
fluid_model_ideal = AnalyticalFluidModel(
    IdealGasState(1.4, 287, 1.8e-5),
)

fluid_settings = FluidSettings(
    model=fluid_model_ideal,
    update_variables=('p', 'T'),  # Thermodynamic iteration variables
    update_length=2,  # Single phase => Two update vars
)

# +++ Boundary conditions
inlet = Inlet(
    {
        'tot': {
            'p': 101352.9,
            'T': 288.16,
        },
        'kin': {
            'alpha': 0.0,
        },
        'oth': {
            'cum_massflow': 4.989512,
        },
    },
)

EQS_ISENTROPIC = {
    ZeroDeviation(): 1,  # No slip
    PercentageEntropyLoss(0.0): (0, 1),
}

EQS_WITH_LOSSES = {
    BackstromSlip(): (0, 1),
    ClearanceJansen(): (0, 1),
    SkinFrictionJansen(): (0, 1),
    BladeLoadingCoppage(): (0, 1),
    LossAdder(): 1,  # Use losses
}


losses = EQS_WITH_LOSSES if ENABLE_LOSSES else EQS_ISENTROPIC

# - # - # - # - #
# Metal angle distribution
METAL_ANGLE = [-30, -44, -53]
angle_values = resample_linear(np.array(METAL_ANGLE), NUM_SPAN)
angle_distribution = Quantity(angle_values, 'deg')
# - # - # - # - #

# +++ Components
rotor = BladeRow(
    name='rotor',
    shaft=shaft,
    row_type='rotor',
    in_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(0, 'deg'),
            'rr_midspan': Quantity(0.07416165, 'm'),
            'height': Quantity(0.0670433, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-44, 'deg'),
            'thick_by_pitch': 0.02,
            'tip_clearance': Quantity(0.3048, 'mm'),
        },
    },
    out_constraints={
        'geo': {
            # *** Meridional geometry
            'meridional_angle': Quantity(90, 'deg'),
            'rr_midspan': Quantity(0.2159, 'm'),
            'height': Quantity(0.01524, 'm'),
            # *** Blades specs
            'metal_angle': Quantity(-30, 'deg'),
            'thick_by_pitch': 0.02,  # Thickness by pitch ratio
            'chord_ax': Quantity(0.133879895, 'm'),
            'num_blades': 15,
            'num_splitters': 15,
        },
        'oth': {
            # 'eta_tt': 0.87,  # Total total efficiency
            # For losses
            'slip_factCoeff': 5.0,
            'abs_roughness': Quantity(1.524, 'micron'),
            'bl_loadingCoeff': 0.75,
        },
    },
    extra_equations={
        # ZeroDeviation(): 1,
        MinimalCamberLine(): (0, 1),
        EffectiveBladeNumber(): 1,
        # *** Enthalpy based Losses
        IsentropicProperties(): (0, 1),
        TotalTotalCompressionEfficiency(): (0, 1),
        # *** Blockage (optional)
        # Definitions
        WorkCoefficient(): (0, 1),
        TotalTotalPressureRatio(): (0, 1),
        **losses,
    },
)

vaneless_diff = VanelessDiffuser(
    'diffuser',
    in_constraints={},
    out_constraints={
        'geo': {
            'rr_midspan': Quantity(0.3055659, 'm'),
            'heightRatio': 1.0,
        },
    },
    extra_equations={
        PercTotalPressureLoss(0.0): (0, 1),  # Isentropic
    },
)

ntw_hecc = ComponentNetwork(
    fluid_settings=fluid_settings,
    inlet=inlet,
    backend=CasadiSystem(num_span=1, scale_suffix='<|'),
    components=[rotor],
)

ntw_hecc.system.add_spanwise_constants('kin_Vm0')
ntw_hecc.system.add_spanwise_constants('stc_p1')


if ntw_hecc.system.num_span > 1:
    ntw_hecc.system.remove_equation(MeridionalUniform, 1)
    ntw_hecc.system.add_equation(MeridionalVariable(), 1)

ntw_hecc.build()

x0 = ntw_hecc.system.get_scaled_guess()
kn_hecc_is = ntw_hecc.system.get_scaled_constraints()
bnd_hecc_is = ntw_hecc.system.get_arguments_bounds()

# IPOPT is more robust, takes variable limits into account -> For 'bi-stable' solutions
# KINSOL is faster, sometimes converges on problems where ipopt struggles
rootfinder_hecc_is = ntw_hecc.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': True,
        # 'ipopt.max_iter': 2000,
        # 'ipopt.max_wall_time': 10,
    },
)
solution_hecc_is = solve_root_problem(
    rootfinder_hecc_is,
    x0,
    kn_hecc_is,
    bnd_hecc_is,
    suppress_output=False,
)
sol_is_dict = ntw_hecc.system.write_solution_to_nodes(solution_hecc_is)

if RUN_MULTI:
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    ntw_hecc.system.num_span = NUM_SPAN
    # Meridional uniform inlet
    ntw_hecc.system.add_spanwise_constants('geo_hh0')
    ntw_hecc.system.boundary_conditions[0]['geo']['metal_angle'] = angle_distribution

    if ntw_hecc.system.num_span > 1:
        ntw_hecc.system.remove_equation(MeridionalVariable, 1)
        ntw_hecc.system.remove_equation(MeridionalUniform, 1)
        ntw_hecc.system.add_equation(MeridionalVariable(), 1)

    ntw_hecc.build()
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    rootfinder_hecc_multi = ntw_hecc.system.make_rootfinder(
        'ipopt',
        opts={
            'error_on_fail': False,
            'ipopt.max_iter': 1000,
            'ipopt.max_wall_time': 25,
        },
    )
    x0_multi = ntw_hecc.system.get_scaled_guess(sol_is_dict)
    kn_hecc_multi = ntw_hecc.system.get_scaled_constraints()
    bnd_hecc_multi = ntw_hecc.system.get_arguments_bounds()
    solution_hecc_multi = solve_root_problem(
        rootfinder_hecc_multi,
        x0_multi,
        kn_hecc_multi,
        bnd_hecc_multi,
        suppress_output=True,
    )
    sol_multi_dict = ntw_hecc.system.write_solution_to_nodes(solution_hecc_multi)

if __name__ == '__main__':
    if RUN_MULTI:
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # Remove isentropic and add losses
        for eq, pos in EQS_ISENTROPIC.items():
            ntw_hecc.system.remove_equation(eq.__class__, pos)
        for eq, pos in EQS_WITH_LOSSES.items():
            ntw_hecc.system.add_equation(eq, pos)
        ntw_hecc.build()

        rootfinder_hecc_loss = ntw_hecc.system.make_rootfinder(
            'ipopt',
            opts={'error_on_fail': True},
        )
        x0_loss = ntw_hecc.system.get_scaled_guess(sol_multi_dict)
        kn_loss = ntw_hecc.system.get_scaled_constraints()
        bnd_loss = ntw_hecc.system.get_arguments_bounds()
        solution_loss = solve_root_problem(
            rootfinder_hecc_loss,
            x0_loss,
            kn_loss,
            bnd_loss,
            suppress_output=True,
        )
        sol_loss_dict = ntw_hecc.system.write_solution_to_nodes(solution_loss)
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

        # ---------------- PLOT ---------------------
    n0 = ntw_hecc.system.nodes[0]
    n1 = ntw_hecc.system.nodes[1]

    if PLOTS:
        fig, axs = plt.subplots(2, 2, figsize=(8, 20))
        for cmp_idx, cmp in enumerate(ntw_hecc.components):
            if cmp.inlet_node is None or cmp.outlet_node is None:
                raise ValueError('Missing nodes')

            node_idx = 0
            for n in (cmp.inlet_node, cmp.outlet_node):
                ax = axs[cmp_idx][node_idx]

                ax.set_title(f'Node number {2 * cmp_idx + node_idx}')
                ax.set_aspect('equal')
                n.kin.plot(n.geo, 8, ax)

                node_idx += 1

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        offset = 0.0
        for c in ntw_hecc.components:
            if not c.inlet_node or not c.outlet_node:
                raise ValueError('missing nodes')

            lines = plot_from_nodes(
                c.inlet_node,
                c.outlet_node,
                False,
                offset,
            )

            offset += c.outlet_node.geo.chord_ax[0]

        print(ntw_hecc.system.nodes[1].oth)
        plt.show(block=True)
        # plt.close('all')

        plt.plot(n1.oth.delta_hmass_loading)
        plt.plot(n1.oth.delta_hmass_clearance)
        plt.plot(n1.oth.delta_hmass_skin)
        plt.ylabel('Enthalpy loss [J / kg / K]')
        plt.xlabel('Spanwise station []')
        plt.legend(['loading', 'clearance', 'skin'])
        plt.grid()
        plt.show()
