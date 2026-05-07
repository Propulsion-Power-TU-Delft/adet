from adet.constants import CoolProperties
from adet.equations.varspec import NodeStates, VarSpec, DEF_NODE


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
    def __init__(
        self,
        node: int = DEF_NODE,
        state: NodeStates | None = None,
    ):
        super().__init__(node, state)

    Entropy = VarSpec(CoolProperties.Smass.value, 'J / kg / K', 1e4)
    Density = VarSpec(CoolProperties.Dmass.value, 'kg / m**3', 2.0, (1e-3, 800.0))
    Pressure = VarSpec(CoolProperties.Press.value, 'Pa', 5e5, (1e4, 150e5))
    Enthalpy = VarSpec(CoolProperties.Hmass.value, 'J / kg', 6e5)
    Temperature = VarSpec(CoolProperties.Temp.value, 'K', 800.0, (80.0, 1800.0))
    IntEnergy = VarSpec(CoolProperties.Umass.value, 'J / kg', 5e5)
    Cp = VarSpec(CoolProperties.Cpmass.value, 'J / kg / K')
    Cv = VarSpec(CoolProperties.Cvmass.value, 'J / kg / K')
    SpeedSound = VarSpec(CoolProperties.SpeedSound.value, 'm / s')
    CriticalTemp = VarSpec(CoolProperties.Pcrit.value, 'K')
    Viscosity = VarSpec(CoolProperties.Viscosity.value, 'Pa * s')


class KinematicVariables(VariableEnum):
    V_mag = VarSpec('V', 'm / s', 1e-2, (0.1, 600.0))
    V_tan = VarSpec('Vt', 'm / s', 1e-2, (-600.0, 600.0))
    V_mer = VarSpec('Vm', 'm / s', 1e-2, (0.1, 600.0))
    W_mag = VarSpec('W', 'm / s', 1e-2, (0.1, 600.0))
    W_tan = VarSpec('Wt', 'm / s', 1e-2, (-600.0, 600.0))
    W_mer = VarSpec('Wm', 'm / s', 1e-2, (0.1, 600.0))
    BladeSpeed = VarSpec('U', 'm / s', 1e-2, (-600.0, 600.0))
    Omega = VarSpec('omega', 'rad / s', 1e-2, (-15000.0, 15000.0), 0, True)
    FlowAngleRel = VarSpec('beta', 'rad', 1e-2, (-1.45, 1.45))
    FlowAngleAbs = VarSpec('alpha', 'rad', 1e-2, (-1.45, 1.45))
    Deflection = VarSpec('deflection', 'rad', 1.0)
    IncAngle = VarSpec('inc_angle', 'rad')
    DevAngle = VarSpec('dev_angle', 'rad', 0.01)
    BetaOpt = VarSpec('beta_opt', 'rad')
    Mach = VarSpec('mach', '', 0.3)
    RelMach = VarSpec('rel_mach', '', 0.3)
    MerMach = VarSpec('mermach', '', 0.3)
    VmRatio = VarSpec('VmRatio', '', 1.0)
    W_choke = VarSpec('W_choke', 'm / s', None, None, 0, True)
    W_hub = VarSpec('W_hub', 'm / s', None, None, 0, True)
    W_tip = VarSpec('W_tip', 'm / s', None, None, 0, True)
    Beta_hub = VarSpec('beta_hub', 'rad', None, None, 0, True)
    Beta_tip = VarSpec('beta_tip', 'rad', None, None, 0, True)
    RelMach_hub = VarSpec('relmach_hub', '', 0.3, None, 0, True)
    RelMach_tip = VarSpec('relmach_tip', '', 0.3, None, 0, True)


class GeometricVariables(VariableEnum):
    HDistr = VarSpec('hh', 'm', 0.1, (1e-8, 1.0))
    RDistr = VarSpec('rr', 'm', 0.1, (1e-4, 3.0))
    Area = VarSpec('area', 'm**2', 0.1, (0.0, 2.0))
    EffArea = VarSpec('area_eff', 'm**2', 0.1, (0.0, 2.0))
    CumArea = VarSpec('cum_area', 'm**2', 0.3, None, 0, True)
    Rmid = VarSpec('Rmid', 'm', None, None, 0, True)
    Rhub = VarSpec('Rhub', 'm', None, None, 0, True)
    Rtip = VarSpec('Rtip', 'm', None, None, 0, True)
    Height = VarSpec('height', 'm', 0.1, (1e-5, 3.0), 0, True)
    MeridionalAngle = VarSpec('mer_angle', 'rad', 0.1, None, 0, True)
    FlareAngle = VarSpec('fl_angle', 'rad', 0.1, (-1.5, 1.5), 0, True)
    RadiusRatio = VarSpec('radRatio', '', 1.0, None, 0, True)
    HubTipRatio = VarSpec('ht_ratio', '', None, None, 0, True)
    HeightRatio = VarSpec('heightRatio', '', 1.0, None, 0, True)
    AspectRatio = VarSpec('aspRatio', '', 2.0, None, 0, True)
    NumBlades = VarSpec('n_blades', '', 20.0, None, 0, True)
    NumBladesEff = VarSpec('n_bl_eff', '', 20.0, None, 0, True)
    NumSplitters = VarSpec('num_splitters', '', 20.0, None, 0, True)
    Chord = VarSpec('chord', 'm', 0.1)
    ChordAx = VarSpec('chord_ax', 'm', 0.1)
    CamberLength = VarSpec('camb_len', 'm', 0.1)
    Pitch = VarSpec('pitch', 'm', 0.1)
    Throat = VarSpec('throat', 'm', 0.1)
    Solidity = VarSpec('solidity', '', 1.0, (0.05, 10.0))
    SolidityMidspan = VarSpec('solidity_mid', '', 1.0, (0.05, 10.0), 0, True)
    TipClearance = VarSpec('tip_cl', 'm', 0.001)
    BackClearance = VarSpec('back_clearance', 'm')
    ClearanceByHeight = VarSpec('clearance_by_height', '')
    AbsRoughness = VarSpec('abs_roughness', 'm')
    BladeAngle = VarSpec('beta_bl', 'rad')
    MetalAngle = VarSpec('metal_angle', 'rad', -0.3, (-1.45, 1.45))
    BldThick = VarSpec('bld_thick', 'm', 0.005)
    ThickByPitch = VarSpec('thick_by_pitch', '', 0.01)
    Stagger = VarSpec('stag_angle', 'rad', 0.1)
    ZweifelCoeff = VarSpec('zweifelCoeff', '')


class Nondimensional(VariableEnum):
    WorkCoeff = VarSpec('work_coeff', '')
    FlowCoeff = VarSpec('flow_coeff', '', 0.8)
    DegreeOfReaction = VarSpec('reactDegree', '', None, None, 0, True)
    DegreeOfReactionTS = VarSpec('reactDegree_ts', '', None, None, 0, True)
    EtaTT = VarSpec('eta_tt', '', 0.9)
    PRatioTT = VarSpec('pRatio_tt', '')
    PRatioTS = VarSpec('pRatio_ts', '')
    PRatio = VarSpec('pRatio', '')
    RhoRatio = VarSpec('rhoRatio', '')
    VolflowRatio = VarSpec('volflowRatio', '', None, None, 0, True)
    TSLoadCoeff = VarSpec('ts_loadCoeff', '', None, None, 0, True)
    HdropCoeff = VarSpec('hdropCoeff', '')
    SwallowingCap = VarSpec('swllCap', '')
    SpecificSpeed = VarSpec('specificSpeed', '')
    SizeParameter = VarSpec('sizeParameter', '')
    CamberCoeff = VarSpec('camberCoeff', '')
    ChAxOutRadRatio = VarSpec('chAx_outRad_Ratio', '')
    VmRatio = VarSpec('VmRatio', '', 1.0)


class OtherVariables(VariableEnum):
    MassFlow = VarSpec('mf', 'kg / s', 20.0, (1e-4, 5e4))
    CumMassFlow = VarSpec('cum_mf', 'kg / s', 20.0, (1e-4, 5e4), 0, True)
    ChMassflow = VarSpec('ch_massflow', 'kg / s', 1.0, (1e-4, 5e4))
    ChokeMassflow = VarSpec('massflow_choke', 'kg / s', None, (1e-4, 5e4), 0, True)
    PBase = VarSpec('p_base', 'Pa', 3e5)
    GammaPV = VarSpec('gamma_pv', '', 1.4)
    Enthalpy_totIs = VarSpec('tot_hmass_is', 'J / kg', 6e5)
    Tis_tot = VarSpec('tot_T_is', 'K', 300.0)
    Enthalpy_Is = VarSpec('stc_hmass_is', 'J / kg', 6e5)
    Tis_stc = VarSpec('stc_T_is', 'K', 300.0)
    TotPRed = VarSpec('tot_p_red', '', 1.5)
    TotTRed = VarSpec('tot_T_red', '', 1.5)
    MomThick = VarSpec('mom_thick', 'm', 1e-5)
    DispThick = VarSpec('disp_thick', 'm', 2e-5)
    DispThickEW = VarSpec('disp_thick_ew', 'm', 2e-5, None, 0, True)
    DispByMom = VarSpec('disp_by_mom', '')
    MomByBld = VarSpec('mom_by_bld', '')
    DispByHgt = VarSpec('disp_by_hgt', '')


class NodeVariables:
    def __init__(self, node: int = DEF_NODE):
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
        return ThermoVariables(self._node, NodeStates.TOTAL)

    @property
    def stc(self) -> ThermoVariables:
        return ThermoVariables(self._node, NodeStates.STATIC)

    @property
    def rlt(self) -> ThermoVariables:
        return ThermoVariables(self._node, NodeStates.RELTOT)
