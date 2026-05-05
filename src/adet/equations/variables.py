from adet.equations.varspec import NodeStates, VarSpec


class BaseEnum:
    def __init__(self, node: int, state: NodeStates | None):
        self._state = state
        self._node = node

    @classmethod
    def __contains__(cls, name: str):
        return name in cls.__dict__

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        if isinstance(attr, VarSpec):
            return attr._with_state(self._state)._at_node(self._node)
        else:
            return attr


class VariableEnum(BaseEnum):
    def __init__(self, node: int):
        super().__init__(node, None)


class ThermoVariables(BaseEnum):
    def __init__(self, node: int, state: NodeStates | None = None):
        super().__init__(node, state)

    Entropy = VarSpec('smass', 'Specific entropy', 'J / kg / K')
    Density = VarSpec('rhomass', 'Density', 'kg / m**3')
    Pressure = VarSpec('p', 'Pressure', 'Pa')
    Enthalpy = VarSpec('hmass', 'Specific enthalpy', 'J / kg')
    Temperature = VarSpec('T', 'Temperature', 'K')
    InternalEnergy = VarSpec('umass', 'Internal Energy', 'J / kg')
    Cp = VarSpec('cpmass', 'Spefic heat (pressure)', 'J / kg / K')
    Cv = VarSpec('cvmass', 'Spefic heat (volume)', 'J / kg / K')
    SpeedSound = VarSpec('speed_sound', 'Speed of sound', 'm / s')
    CriticalTemp = VarSpec('T_critical', 'Critical temperature', 'K')
    Viscosity = VarSpec('viscosity', 'Dynamic viscosity', 'Pa * s')
    GammaPV = VarSpec('gamma_pv', 'Isentropic exponent', 'dimensionless')


class KinematicVariables(VariableEnum):
    # Absolute velocity components
    V_mag = VarSpec('V', 'Absolute velocity', 'm / s')
    V_tan = VarSpec('Vt', 'Absolute velocity (tangential)', 'm / s')
    V_mer = VarSpec('Vm', 'Absolute velocity (meridional)', 'm / s')
    # Relative velocity components
    W_mag = VarSpec('W', 'Relative Velocity', 'm / s')
    W_tan = VarSpec('Wt', 'Relative Velocity (tangential)', 'm / s')
    W_mer = VarSpec('Wm', 'Relative Velocity (meridional)', 'm / s')
    # Blade speed and rotation
    BladeSpeed = VarSpec('U', 'Blade speed', 'm / s')
    Omega = VarSpec('omega', 'Rotational speed', 'rad / s', scalar=True)
    # Flow angles
    FlowAngleRel = VarSpec('beta', 'Relative flow angle', 'rad')
    FlowAngleAbs = VarSpec('alpha', 'Absolute flow angle', 'rad')
    Deflection = VarSpec('deflection', 'Flow deflection', 'rad')
    IncAngle = VarSpec('inc_angle', 'Incidence angle', 'rad')
    DevAngle = VarSpec('dev_angle', 'Deviation angle', 'rad')
    BetaOpt = VarSpec('beta_opt', 'Optimal relative flow angle', 'rad')
    # Mach numbers
    Mach = VarSpec('mach', 'Absolute Mach number', 'dimensionless')
    RelMach = VarSpec('rel_mach', 'Relative Mach number', 'dimensionless')
    MerMach = VarSpec('mermach', 'Meridional Mach number', 'dimensionless')
    # Ratios
    VmRatio = VarSpec('VmRatio', 'Meridional velocity ratio', 'dimensionless')
    # Endwall scalars
    W_choke = VarSpec('W_choke', 'Relative velocity at choke', 'm / s', scalar=True)
    W_hub = VarSpec('W_hub', 'Relative velocity at hub', 'm / s', scalar=True)
    W_tip = VarSpec('W_tip', 'Relative velocity at tip', 'm / s', scalar=True)
    Beta_hub = VarSpec('beta_hub', 'Relative flow angle at hub', 'rad', scalar=True)
    Beta_tip = VarSpec('beta_tip', 'Relative flow angle at tip', 'rad', scalar=True)
    RelMach_hub = VarSpec(
        'relmach_hub', 'Relative Mach at hub', 'dimensionless', scalar=True
    )
    RelMach_tip = VarSpec(
        'relmach_tip', 'Relative Mach at tip', 'dimensionless', scalar=True
    )


class GeometricVariables(VariableEnum):
    # Annulus spanwise distributions
    HDistr = VarSpec('hh', 'Height distribution', 'm')
    RDistr = VarSpec('rr', 'Radius distribution', 'm')
    Area = VarSpec('area', 'Annulus area (geometric)', 'm**2')
    EffArea = VarSpec('area_eff', 'Annulus area (w/ blockage)', 'm**2')
    CumArea = VarSpec('cum_area', 'Cumulative area', 'm**2', scalar=True)
    # Meridional channel specs (scalars)
    Rmid = VarSpec('Rmid', 'Mid radius', 'm', scalar=True)
    Rhub = VarSpec('Rhub', 'Hub radius', 'm', scalar=True)
    Rtip = VarSpec('Rtip', 'Tip radius', 'm', scalar=True)
    Height = VarSpec('height', 'Channel height', 'm', scalar=True)
    MeridionalAngle = VarSpec('mer_angle', 'Meridional angle', 'rad', scalar=True)
    # Geometrical ratios (scalars)
    FlareAngle = VarSpec('fl_angle', 'Endwall flare angle', 'rad', scalar=True)
    RadiusRatio = VarSpec('radRatio', 'Radius ratio', 'dimensionless', scalar=True)
    HubTipRatio = VarSpec('ht_ratio', 'Hub-to-tip ratio', 'dimensionless', scalar=True)
    HeightRatio = VarSpec('heightRatio', 'Height ratio', 'dimensionless', scalar=True)
    AspectRatio = VarSpec('aspRatio', 'Blade a. ratio', 'dimensionless', scalar=True)
    # Blade specs
    NumBlades = VarSpec('n_blades', 'Number of blades', 'dimensionless', scalar=True)
    NumBladesEff = VarSpec(
        'n_bl_eff', 'Effective number of blades', 'dimensionless', scalar=True
    )
    NumSplitters = VarSpec(
        'num_splitters', 'Number of splitters', 'dimensionless', scalar=True
    )
    Chord = VarSpec('chord', 'Chord', 'm')
    ChordAx = VarSpec('chord_ax', 'Axial chord', 'm')
    CamberLength = VarSpec('camb_len', 'Camber length', 'm')
    Pitch = VarSpec('pitch', 'Blade pitch', 'm')
    Throat = VarSpec('throat', 'Throat width', 'm')
    Solidity = VarSpec('solidity', 'Blade solidity', 'dimensionless')
    SolidityMidspan = VarSpec(
        'solidity_mid', 'Midspan solidity', 'dimensionless', scalar=True
    )
    TipClearance = VarSpec('tip_cl', 'Tip clearance', 'm')
    BackClearance = VarSpec('back_clearance', 'Back clearance', 'm')
    ClearanceByHeight = VarSpec(
        'clearance_by_height', 'Clearance to height ratio', 'dimensionless'
    )
    AbsRoughness = VarSpec('abs_roughness', 'Absolute roughness', 'm')
    BladeAngle = VarSpec('beta_bl', 'Blade angle', 'rad')
    MetalAngle = VarSpec('metal_angle', 'Metal (blade) angle', 'rad')
    BldThick = VarSpec('bld_thick', 'Blade thickness', 'm')
    ThickByPitch = VarSpec(
        'thick_by_pitch', 'Thickness to pitch ratio', 'dimensionless'
    )
    Stagger = VarSpec('stag_angle', 'Stagger angle', 'rad')
    ZweifelCoeff = VarSpec(
        'zweifelCoeff', 'Zweifel loading coefficient', 'dimensionless'
    )


class Nondimensional(VariableEnum):
    WorkCoeff = VarSpec('work_coeff', 'Work coefficient', 'dimensionless')
    FlowCoeff = VarSpec('flow_coeff', 'Flow coefficient', 'dimensionless')
    # Degree of reaction
    DegreeOfReaction = VarSpec(
        'reactDegree', 'Degree of reaction', 'dimensionless', scalar=True
    )
    DegreeOfReactionTS = VarSpec(
        'reactDegree_ts',
        'Total-to-static degree of reaction',
        'dimensionless',
        scalar=True,
    )
    # Efficiencies
    EtaTT = VarSpec('eta_tt', 'Total-to-total efficiency', 'dimensionless')
    # Pressure and density ratios
    PRatioTT = VarSpec('pRatio_tt', 'Total-to-total pressure ratio', 'dimensionless')
    PRatioTS = VarSpec('pRatio_ts', 'Total-to-static pressure ratio', 'dimensionless')
    PRatio = VarSpec('pRatio', 'Pressure ratio', 'dimensionless')
    RhoRatio = VarSpec('rhoRatio', 'Density ratio', 'dimensionless')
    VolflowRatio = VarSpec(
        'volflowRatio', 'Volume flow ratio', 'dimensionless', scalar=True
    )
    # Loading and performance
    TSLoadCoeff = VarSpec(
        'ts_loadCoeff',
        'Total-to-static loading coefficient',
        'dimensionless',
        scalar=True,
    )
    HdropCoeff = VarSpec('hdropCoeff', 'Head drop coefficient', 'dimensionless')
    SwallowingCap = VarSpec('swllCap', 'Swallowing capacity', 'dimensionless')
    SpecificSpeed = VarSpec('specificSpeed', 'Specific speed', 'dimensionless')
    SizeParameter = VarSpec('sizeParameter', 'Size parameter', 'dimensionless')
    # Blade camber/geometry ratios
    CamberCoeff = VarSpec('camberCoeff', 'Camber coefficient', 'dimensionless')
    ChAxOutRadRatio = VarSpec(
        'chAx_outRad_Ratio', 'Axial chord to outer radius ratio', 'dimensionless'
    )
    VmRatio = VarSpec('VmRatio', 'Meridional velocity ratio', 'dimensionless')


class OtherVariables(VariableEnum):
    # Mass flows
    MassFlow = VarSpec('mf', 'Annulus massflow', 'kg / s')
    CumMassFlow = VarSpec('cum_mf', 'Annulus massflow', 'kg / s', scalar=True)
    ChMassflow = VarSpec('ch_massflow', 'Channel mass flow', 'kg / s')
    ChokeMassflow = VarSpec('massflow_choke', 'Choke mass flow', 'kg / s', scalar=True)
    # Thermodynamic derived
    PBase = VarSpec('p_base', 'Base pressure', 'Pa')
    Enthalpy_totIs = VarSpec('tot_hmass_is', 'Isentropic total enthalpy', 'J / kg')
    Tis_tot = VarSpec('tot_T_is', 'Isentropic total temperature', 'K')
    Enthalpy_Is = VarSpec('stc_hmass_is', 'Isentropic static enthalpy', 'J / kg')
    Tis_stc = VarSpec('stc_T_is', 'Isentropic static temperature', 'K')
    TotPRed = VarSpec('tot_p_red', 'Reduced total pressure', 'dimensionless')
    TotTRed = VarSpec('tot_T_red', 'Reduced total temperature', 'dimensionless')
    # Boundary layer quantities
    MomThick = VarSpec('mom_thick', 'Momentum thickness', 'm')
    DispThick = VarSpec('disp_thick', 'Displacement thickness', 'm')
    DispThickEW = VarSpec(
        'disp_thick_ew', 'Endwall displacement thickness', 'm', scalar=True
    )
    DispByMom = VarSpec(
        'disp_by_mom', 'Displacement to momentum thickness ratio', 'dimensionless'
    )
    MomByBld = VarSpec(
        'mom_by_bld', 'Momentum to blade thickness ratio', 'dimensionless'
    )
    DispByHgt = VarSpec('disp_by_hgt', 'Displacement to height ratio', 'dimensionless')


class NodeVariables:
    def __init__(self, node: int):
        self._node = node

    @property
    def kin(self) -> KinematicVariables:
        return KinematicVariables(self._node)

    @property
    def geo(self) -> GeometricVariables:
        return GeometricVariables(self._node)

    @property
    def ndim(self) -> Nondimensional:
        return Nondimensional(self._node)

    @property
    def oth(self) -> OtherVariables:
        return OtherVariables(self._node)

    @property
    def tot(self) -> ThermoVariables:
        return ThermoVariables(self._node)

    @property
    def stc(self) -> ThermoVariables:
        return ThermoVariables(self._node)

    @property
    def rlt(self) -> ThermoVariables:
        return ThermoVariables(self._node)
