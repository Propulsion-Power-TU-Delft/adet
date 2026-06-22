from adet.constants import CoolProperties
from adet.varspec import DEF_NODE, DEF_STATE, NodeStates, VarSpec


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
        super().__init__(node, DEF_STATE)


class ThermoVariables(BaseEnum):
    def __init__(
        self,
        node: int = DEF_NODE,
        state: NodeStates | None = DEF_STATE,
    ):
        super().__init__(node, state)

    Cp = VarSpec(CoolProperties.Cpmass.value, 'J / kg / K')
    Cv = VarSpec(CoolProperties.Cvmass.value, 'J / kg / K')
    Entropy = VarSpec(CoolProperties.Smass.value, 'J / kg / K', 1e4)
    Density = VarSpec(CoolProperties.Dmass.value, 'kg / m**3', 2.0, (0.0, 2e3))
    Pressure = VarSpec(CoolProperties.Press.value, 'Pa', 5e5, (0.1, 2e7))
    Enthalpy = VarSpec(CoolProperties.Hmass.value, 'J / kg', 6e5)
    MolarMass = VarSpec(CoolProperties.MolarMass.value, 'kg / mol')
    IntEnergy = VarSpec(CoolProperties.Umass.value, 'J / kg', 5e5)
    Viscosity = VarSpec(CoolProperties.Viscosity.value, 'Pa * s')
    SpeedSound = VarSpec(CoolProperties.SpeedSound.value, 'm / s')
    Temperature = VarSpec(CoolProperties.Temp.value, 'K', 500.0, (30.0, 1800.0))
    GasConstant = VarSpec(CoolProperties.GasConstant.value, 'J / (mol * K)')
    CriticalTemp = VarSpec(CoolProperties.Tcrit.value, 'K')
    CriticalPressure = VarSpec(CoolProperties.Pcrit.value, 'Pa')


class KinematicVariables(VariableEnum):
    Mach = VarSpec('mach', '', 0.3)
    Omega = VarSpec('omega', 'rad / s', 0.5, scalar=True)
    V_mag = VarSpec('V', 'm / s', 0.5, (0.0, 2e3))
    V_mer = VarSpec('Vm', 'm / s', 0.5, (0.0, 2e3))
    V_tan = VarSpec('Vt', 'm / s', 0.5)
    W_hub = VarSpec('W_hub', 'm / s', scalar=True)
    W_mag = VarSpec('W', 'm / s', 0.5, (0.0, 2e3))
    W_mer = VarSpec('Wm', 'm / s', 0.5, (0.0, 2e3))
    W_tan = VarSpec('Wt', 'm / s', 0.5)
    W_tip = VarSpec('W_tip', 'm / s', scalar=True)
    MachPresh = VarSpec('mach_prsh', '', 2.0, (1.0, 10.0))
    MachThroat = VarSpec('mach_thr', '', 0.5)
    VmRatio = VarSpec('VmRatio', '', 1.0)
    W_choke = VarSpec('W_choke', 'm / s', scalar=True)
    BetaOpt = VarSpec('beta_opt', 'rad')
    DevAngle = VarSpec('dev_angle', 'rad', 0.01)
    IncAngle = VarSpec('inc_angle', 'rad')
    MerMach = VarSpec('mermach', '', 0.1)
    RelMach = VarSpec('relmach', '', 0.1)
    Beta_mid = VarSpec('beta_mid', 'rad', scalar=True)
    Beta_hub = VarSpec('beta_hub', 'rad', scalar=True)
    Beta_tip = VarSpec('beta_tip', 'rad', scalar=True)
    BladeSpeed = VarSpec('U', 'm / s', 1e-2)
    RelMach_hub = VarSpec('relmach_hub', '', 0.3, scalar=True)
    RelMach_tip = VarSpec('relmach_tip', '', 0.3, scalar=True)
    FlowAngleAbs = VarSpec('alpha', 'rad', 1e-2, (-1.5, 1.5))
    FlowAngleRel = VarSpec('beta', 'rad', 1e-2, (-1.5, 1.5))
    RowDeflection = VarSpec('row_defl', 'rad', 1.0)


class GeometricVariables(VariableEnum):
    Area = VarSpec('area', 'm**2', 0.1)
    Chord = VarSpec('chord', 'm', 0.1)
    Pitch = VarSpec('pitch', 'm', 0.1)
    Rhub = VarSpec('Rhub', 'm', scalar=True)
    Rmid = VarSpec('Rmid', 'm', scalar=True)
    Rtip = VarSpec('Rtip', 'm', scalar=True)
    HDistr = VarSpec('hh', 'm', 0.1, bounds=(0.0, 1e10))
    RDistr = VarSpec('rr', 'm', 0.1)
    Height = VarSpec('height', 'm', 0.1, (0.0, 1e2), scalar=True)
    CumArea = VarSpec('cum_area', 'm**2', 0.3, scalar=True)
    EffArea = VarSpec('area_eff', 'm**2', 0.1)
    ChordAx = VarSpec('chord_ax', 'm', 0.1)
    Stagger = VarSpec('stag_angle', 'rad', 0.1)
    BldThick = VarSpec('bld_thick', 'm', 0.005)
    Solidity = VarSpec('solidity', '', 1.0)
    NumBlades = VarSpec('n_blades', '', 50.0, scalar=True)
    NumBladesOpt = VarSpec('n_blades_opt', '', 50.0, scalar=True)
    ThroatArea = VarSpec('A_throat', 'm**2', 0.1)
    ThroatRadius = VarSpec('rr_throat', 'm', 0.01)
    EffSolidity = VarSpec('eff_solidity', '', 1.0)
    FlareAngle = VarSpec('fl_angle', 'rad', 0.1, (-1.5, 1.5), scalar=True)
    MetalAngle = VarSpec('beta_geom', 'rad', -0.3, (-1.5, 1.5))
    RadiusRatio = VarSpec('radRatio', '', 1.0, scalar=True)
    HubTipRatio = VarSpec('hubtip_ratio', '', scalar=True)
    HeightRatio = VarSpec('heightRatio', '', 1.0, scalar=True)
    AspectRatio = VarSpec('aspRatio', '', 2.0, scalar=True)
    TipClearance = VarSpec('tip_cl', 'm', 0.001)
    AbsRoughness = VarSpec('abs_roughness', 'm')
    NumBladesEff = VarSpec('n_bl_eff', '', 20.0, scalar=True)
    NumSplitters = VarSpec('num_splitters', '', 20.0, scalar=True)
    CamberLength = VarSpec('camb_len', 'm', 0.1)
    ThickByPitch = VarSpec('thick_by_pitch', '', 0.01)
    ZweifelCoeff = VarSpec('zweifelCoeff', '')
    BackClearance = VarSpec('back_clearance', 'm')
    MetalAngleHub = VarSpec('metal_angle_hub', 'rad', -0.3, (-1.5, 1.5), scalar=True)
    MetalAngleTip = VarSpec('metal_angle_tip', 'rad', -0.3, (-1.5, 1.5), scalar=True)
    SolidityMidspan = VarSpec('solidity_mid', '', 1.0, scalar=True)
    MeridionalAngle = VarSpec('mer_angle', 'rad', 0.1, scalar=True)
    ClearanceByHeight = VarSpec('clearance_by_height', '')
    ShapeCoeff = VarSpec('shape_k', '')
    HydLen = VarSpec('hyd_len', 'm')
    HydDiam = VarSpec('hyd_diam', 'm')


class Nondimensional(VariableEnum):
    EtaTT = VarSpec('eta_tt', '', 0.9)
    PRatio = VarSpec('pRatio', '')
    PRatio_choke = VarSpec('pRatio_chk', '')
    VmRatio = VarSpec('VmRatio', '', 1.0)
    PRatioTT = VarSpec('pRatio_tt', '')
    PRatioTS = VarSpec('pRatio_ts', '', guess=1.0)
    RhoRatio = VarSpec('rhoRatio', '')
    WorkCoeff = VarSpec('work_coeff', '')
    FlowCoeff = VarSpec('flow_coeff', '', 0.8)
    HdropCoeff = VarSpec('hdropCoeff', '')
    TSLoadCoeff = VarSpec('ts_loadCoeff', '', scalar=True)
    CamberCoeff = VarSpec('camberCoeff', '')
    VolflowRatio = VarSpec('volflowRatio', '', scalar=True)
    SwallowingCap = VarSpec('swllCap', '')
    SpecificSpeed = VarSpec('specificSpeed', '')
    SizeParameter = VarSpec('sizeParameter', '')
    ChAxOutRadRatio = VarSpec('chAx_outRad_Ratio', '')
    DegreeOfReaction = VarSpec('reactDegree', '', scalar=True)
    DegreeOfReactionTS = VarSpec('reactDegree_ts', '', scalar=True)


class Losses(VariableEnum):
    # Enthalpy based
    Dht_loading = VarSpec('dht_loading', 'J / kg')
    Dht_clearance = VarSpec('dht_clearance', 'J / kg')
    Dht_skin = VarSpec('dht_skin', 'J / kg')
    Dht_incidence = VarSpec('dht_incidence', 'J / kg')
    Dht_mixing = VarSpec('dht_mixing', 'J / kg')
    Dht_disk = VarSpec('dht_disk', 'J / kg')
    Dht_recirculation = VarSpec('dht_recirculation', 'J / kg')
    Dht_leakage = VarSpec('dht_leakage', 'J / kg')
    Dht_lost = VarSpec('dht_lost', 'J / kg')
    Dht_total = VarSpec('dtot_hmass', 'J / kg')
    # Entropy based
    Ds_shock = VarSpec('ds_shock', 'J / kg / K')
    Ds_leakage = VarSpec('ds_leakage', 'J / kg / K')
    Ds_mixing = VarSpec('ds_mixing', 'J / kg / K')
    Ds_profile = VarSpec('ds_profile', 'J / kg / K')
    Ds_secondary = VarSpec('ds_secondary', 'J / kg / K')
    Ds_total = VarSpec('ds_total', 'J / kg / K')
    Ds_main = VarSpec('ds_main', 'J / kg / K')


class OtherVariables(VariableEnum):
    PBase = VarSpec('p_base', 'Pa', 3e5)
    GammaPV = VarSpec('gamma_pv', '', 1.4)
    Tis_tot = VarSpec('tot_T_is', 'K', 300.0)
    Tis_stc = VarSpec('stc_T_is', 'K', 300.0)
    TotPRed = VarSpec('tot_p_red', '', 1.5)
    TotTRed = VarSpec('tot_T_red', '', 1.5)
    MomThick = VarSpec('mom_thick', 'm', 1e-5)
    DispThick = VarSpec('disp_thick', 'm', 2e-5)
    DispByMom = VarSpec('disp_by_mom', '')
    DispByHgt = VarSpec('disp_by_hgt', '')
    MomByBld = VarSpec('mom_by_bld', '')
    MassFlow = VarSpec('mf', 'kg / s', 20.0, (0.0, 5e4))
    CumMassFlow = VarSpec('cum_mf', 'kg / s', 20.0, (0.0, 5e4), scalar=True)
    ChanMassflow = VarSpec('ch_massflow', 'kg / s', 1.0, (0.0, 5e4))
    ThrMassFlow = VarSpec('thr_mass', 'kg / s', 1.0)
    TgtMassFlow = VarSpec('tgt_mass', 'kg / s', 1.0)
    ChokeMassflow = VarSpec('massflow_choke', 'kg / s', guess=5.0)
    RltEnthalpyChoke = VarSpec('hrlt_choke', 'J / kg', guess=5e5)
    ThrTemperature = VarSpec('thr_temp', 'K', 500)
    ThrPressure = VarSpec('thr_prss', 'Pa', 5e5)
    DispThickEW = VarSpec('disp_thick_ew', 'm', 2e-5, scalar=True)
    Enthalpy_Is = VarSpec('stc_hmass_is', 'J / kg', 6e5)
    ShockAngle = VarSpec('sh_angle', 'rad', 0.9, (0.0, 1.6))
    ShockDeflection = VarSpec('sh_deflec', 'rad', -0.8, (-1.5, 1.5))
    EntropyPresh = VarSpec('s_presh', 'J / kg / K', 1e3)
    EnthalpyPresh = VarSpec('h_presh', 'J / kg', 1e5)
    Enthalpy_totIs = VarSpec('tot_hmass_is', 'J / kg', 6e5)
    BlLoadingCoeff = VarSpec('bl_loadingCoeff', '')
    SlipFactor = VarSpec('slip_factor', '')
    SlipFactCoeff = VarSpec('slip_factCoeff', '')
    Cf_smooth = VarSpec('Cf_smooth', '')
    Cf_rough = VarSpec('Cf_rough', '')
    WakeFrac = VarSpec('wake_frac', '')
    MinWakeFrac = VarSpec('minWake_frac', '')
    MaxWakeFrac = VarSpec('maxWake_frac', '')
    IncCoeff = VarSpec('incCoeff', '')
    WorkLossCoeff = VarSpec('worklossCoeff', '')
    XiCambLenA = VarSpec('xi_camb_len_A', '')
    XiCambLenB = VarSpec('xi_camb_len_B', '')
    ProfileLoading = VarSpec('k_prof', '', 0.3)
    CdProfile = VarSpec('cd_profile', '')
    DischCoeff = VarSpec('disch_coeff', '')


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

    @property
    def loss(self) -> Losses:
        return Losses(self._node)
