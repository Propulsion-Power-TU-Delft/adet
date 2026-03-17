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
from adet.equations.geometrical import MinimalCamberLine
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
    SkinFrictionJansen,
    CompressorLosses,
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
        'Vm': (10.0, 480.0),
        'U': (0, 600),
        'beta': (-1.48, 1.48),
        'relmach': (0.0, 1.04),
        'eta_tt': (0.8, 1.0),
        # 'pRatio_tt': (0.0, 7.0),
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

NUM_SPAN = 1
PLOTS = True
ENABLE_LOSSES = False
RUN_MULTI = True
RPM_DES = 21789
# +++ Shaftskin_omega0 (node 0) is unknown,
shaft = Shaft(
    omega=Quantity(RPM_DES, 'rpm'),
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
            # 'cum_massflow': 4.989512,
            'cum_massflow': 4.5,
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
    CompressorLosses(): 1,  # Use losses
}


# - # - # - # - #
# Metal angle distribution
METAL_ANGLE = np.array([-30, -44, -53])
angle_values = resample_linear(METAL_ANGLE, NUM_SPAN)
if NUM_SPAN == 1:
    angle_values = np.array([-44])
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
            # 'eta_tt': 0.821,  # Total total efficiency
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
        **EQS_ISENTROPIC,
    },
)

vaneless_diff = VanelessDiffuser(
    'diffuser',
    outlet_bc={
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
    components=[rotor, vaneless_diff],
)

rotor.set_spanwise_constant('kin_Vm0', 'geo_hh0', 'stc_p1')
vaneless_diff.set_spanwise_constant('stc_p1')

ntw_hecc.build()

x0 = ntw_hecc.system.get_scaled_guess()
kn_hecc_is = ntw_hecc.system.get_scaled_constraints()
bnd_hecc_is = ntw_hecc.system.get_arguments_bounds()

# IPOPT is more robust, takes variable limits into account -> For 'bi-stable' solutions
# KINSOL is faster, sometimes converges on problems where ipopt struggles
rootfinder_hecc_is = ntw_hecc.system.make_rootfinder(
    'ipopt',
    opts={
        'error_on_fail': False,
        'ipopt.max_wall_time': 10,
    },
)

rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')
print('*** SOLVING SINGLE SPAN ISENTROPIC***')
solution_hecc_is = solve_root_problem(
    rootfinder_hecc_is,
    x0,
    kn_hecc_is,
    # bnd_hecc_is,
    suppress_output=False,
)
solution_hecc_is = solve_root_problem(rtfn_kin, solution_hecc_is, kn_hecc_is)
sol_is_dict = ntw_hecc.system.write_solution_to_nodes(solution_hecc_is)

if RUN_MULTI:
    print('*** SOLVING MULTISPAN ISENTROPIC***')
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    ntw_hecc.system.num_span = NUM_SPAN
    rotor.set_boundary_cond('geo_metal_angle0', angle_distribution)

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
        # bnd_hecc_multi,
        suppress_output=True,
    )
    sol_multi_dict = ntw_hecc.system.write_solution_to_nodes(solution_hecc_multi)


if __name__ == '__main__':
    if RUN_MULTI:
        print('*** SOLVING MULTISPAN WITH LOSSES***')
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # Remove isentropic and add losses
        for eq, pos in EQS_ISENTROPIC.items():
            rotor.remove_equation(eq.__class__, pos)
        for eq, pos in EQS_WITH_LOSSES.items():
            rotor.add_equation(eq, pos)
        ntw_hecc.build()

        rootfinder_hecc_loss = ntw_hecc.system.make_rootfinder(
            'ipopt',
            opts={'error_on_fail': False},
        )
        rtfn_kin = ntw_hecc.system.make_rootfinder('kinsol')
        x0_loss = ntw_hecc.system.get_scaled_guess(sol_multi_dict)
        kn_loss = ntw_hecc.system.get_scaled_constraints()
        bnd_loss = ntw_hecc.system.get_arguments_bounds()
        solution_loss = solve_root_problem(
            rootfinder_hecc_loss,
            x0_loss,
            kn_loss,
            # bnd_loss,
            suppress_output=True,
        )
        solution_loss = solve_root_problem(rtfn_kin, solution_loss, kn_loss)
        sol_loss_dict = ntw_hecc.system.write_solution_to_nodes(solution_loss)
        #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    SPEED_LINES = {
        RPM_DES: (4.5, 5.68),
        0.95 * RPM_DES: (3.86, 5.4),
        0.9 * RPM_DES: (3.6, 5.0),
        0.85 * RPM_DES: (3.18, 4.6),
        0.75 * RPM_DES: (2.31, 3.86),
    }
    N_PTS = 4

    speed_lines = {
        rpm * 2 * np.pi / 60: np.linspace(k[0], k[1], N_PTS)
        for rpm, k in SPEED_LINES.items()
    }

    ntw_hecc.build()
    kn = ntw_hecc.get_scaled_constraints()
    x0 = ntw_hecc.get_scaled_guess(sol_loss_dict)
    bnd = ntw_hecc.get_arguments_bounds()

    omega_idx = ntw_hecc.system.constraints.index('kin_omega1')
    omega_scl = ntw_hecc.system.constraints_scaling[omega_idx]

    mf_idx = ntw_hecc.system.constraints.index('oth_cum_massflow0')
    mf_scl = ntw_hecc.system.constraints_scaling[mf_idx]

    pr_idx = ntw_hecc.system.free_args.index('oth_pRatio_tt1')
    pr_scl = ntw_hecc.system.free_args_scaling[pr_idx]
    RUN_SPEEDLINE = True
    if RUN_SPEEDLINE:
        print('*** RUNNING SPEEDLINES ***')
        sol = None
        rtfn = ntw_hecc.system.make_rootfinder(
            'ipopt',
            opts={
                'error_on_fail': False,
                'ipopt.max_iter': 500,
            },
        )

        # Find indices for additional parameters
        eta_idx = ntw_hecc.system.free_args.index('oth_eta_tt1')
        eta_scl = ntw_hecc.system.free_args_scaling[eta_idx]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        for omega, massflows in speed_lines.items():
            pratios = []
            etas = []
            converged_count = 0
            kn[omega_idx] = np.array([omega / omega_scl])
            for mf in massflows:
                kn[mf_idx] = np.array([mf / mf_scl])

                try:
                    if sol is None:
                        sol = solve_root_problem(
                            rtfn, x0, kn, bnd, suppress_output=True
                        )
                    else:
                        sol = solve_root_problem(
                            rtfn, sol, kn, bnd, suppress_output=True
                        )

                    pr = sol[pr_idx][0] * pr_scl
                    eta = sol[eta_idx][0] * eta_scl
                    pratios.append(pr)
                    etas.append(eta)
                    converged_count += 1
                except Exception as e:
                    print(
                        f'  Warning: convergence failed at mf={mf:.3f} kg/s, omega={omega:.0f} RPM'
                    )
                    pratios.append(np.nan)
                    etas.append(np.nan)

            print(
                f'Speed {omega:.0f} RPM: {converged_count}/{len(massflows)} points converged'
            )
            print(
                f'  Pressure ratios: {[f"{pr:.4f}" if not np.isnan(pr) else "nan" for pr in pratios]}'
            )

            axs[0, 0].plot(
                massflows,
                pratios,
                label=f'{omega:.0f} RPM',
                marker='o',
                markersize=5,
                linewidth=2,
            )
            axs[0, 1].plot(
                massflows,
                etas,
                label=f'{omega:.0f} RPM',
                marker='s',
                markersize=5,
                linewidth=2,
            )
            axs[1, 0].plot(
                pratios,
                etas,
                label=f'{omega:.0f} RPM',
                marker='^',
                markersize=5,
                linewidth=2,
            )

        # Pressure ratio vs mass flow
        axs[0, 0].set_xlabel('Mass flow [kg/s]', fontsize=11)
        axs[0, 0].set_ylabel('Pressure ratio [−]', fontsize=11)
        axs[0, 0].set_title(
            'Compressor Map: Pressure Ratio', fontsize=12, fontweight='bold'
        )
        axs[0, 0].legend(loc='best')
        axs[0, 0].grid(True, alpha=0.3)

        # Efficiency vs mass flow
        axs[0, 1].set_xlabel('Mass flow [kg/s]', fontsize=11)
        axs[0, 1].set_ylabel('Total-to-total efficiency [−]', fontsize=11)
        axs[0, 1].set_title(
            'Compressor Map: Isentropic Efficiency', fontsize=12, fontweight='bold'
        )
        axs[0, 1].legend(loc='best')
        axs[0, 1].grid(True, alpha=0.3)

        # Efficiency vs pressure ratio
        axs[1, 0].set_xlabel('Pressure ratio [−]', fontsize=11)
        axs[1, 0].set_ylabel('Total-to-total efficiency [−]', fontsize=11)
        axs[1, 0].set_title(
            'Compressor Performance: η vs PR', fontsize=12, fontweight='bold'
        )
        axs[1, 0].legend(loc='best')
        axs[1, 0].grid(True, alpha=0.3)

        # Hide fourth subplot
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        # ---------------- PLOT ---------------------
    n0 = ntw_hecc.system.nodes[0]
    n1 = ntw_hecc.system.nodes[1]
    n2 = ntw_hecc.system.nodes[2]
    n3 = ntw_hecc.system.nodes[3]

    fig, axs = plt.subplots(2, 2, figsize=(8, 20))
    for cmp_idx, comp in enumerate(ntw_hecc.components):
        inlet_node = comp.get_inlet_node(ntw_hecc)
        outlet_node = comp.get_outlet_node(ntw_hecc)

        if inlet_node is None or outlet_node is None:
            raise ValueError('Missing nodes')

        node_idx = 0
        for n in (inlet_node, outlet_node):
            ax = axs[cmp_idx][node_idx]

            ax.set_title(f'Node number {2 * cmp_idx + node_idx}')
            ax.set_aspect('equal')
            n.kin.plot(n.geo, 8, ax)

            node_idx += 1

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    offset = 0.0
    for comp in ntw_hecc.components:
        inlet_node = comp.get_inlet_node(ntw_hecc)
        outlet_node = comp.get_outlet_node(ntw_hecc)
        if not inlet_node or not outlet_node:
            raise ValueError('missing nodes')

        lines = plot_from_nodes(
            inlet_node,
            outlet_node,
            False,
            offset,
        )

        offset += outlet_node.geo.chord_ax[0]

    print(n1.oth)
    show_plots = input('Show plots? [y/N] ').strip().lower() == 'y'
    if show_plots:
        plt.show()
    else:
        plt.close('all')

    plt.plot(n1.oth.delta_hmass_loading)
    plt.plot(n1.oth.delta_hmass_clearance)
    plt.plot(n1.oth.delta_hmass_skin)
    plt.ylabel('Enthalpy loss [J / kg / K]')
    plt.xlabel('Spanwise station []')
    plt.legend(['loading', 'clearance', 'skin'])
    plt.grid()
    if show_plots:
        plt.show()
    else:
        plt.close('all')
