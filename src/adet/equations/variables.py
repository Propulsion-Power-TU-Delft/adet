from adet.equations.varspec import NodeStates, VarSpec


class VariableEnum:
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


class ThermoVariables(VariableEnum):
    def __init__(self, node: int, state: NodeStates):
        super().__init__(node, state)

    Entropy = VarSpec('smass', 'Specific entropy', 'J / kg / K')
    Density = VarSpec('rhomass', 'Density', 'kg / m**3')
    Pressure = VarSpec('p', 'Pressure', 'Pa')
    Enthalpy = VarSpec('hmass', 'Specific enthalpy', 'J / kg')
    Temperature = VarSpec('T', 'Temperature', 'K')
    InternalEnergy = VarSpec('umass', 'Internal Energy', 'J / kg')
    Cp = VarSpec('cpmass', 'Spefic heat (pressure)', 'J / kg')
    Cv = VarSpec('cvmass', 'Spefic heat (volume)', 'J / kg')


class GenericVariables(VariableEnum):
    def __init__(self, node: int):
        super().__init__(node, None)

    V_mag = VarSpec('V', 'Absolute velocity', 'm / s')
    V_tan = VarSpec('Vt', 'Absolute velocity (tangential)', 'm / s')
    V_mer = VarSpec('Vm', 'Absolute velocity (meridional)', 'm / s')
    W_mag = VarSpec('W', 'Relative Velocity', 'm / s')
    W_tan = VarSpec('Wt', 'Relative Velocity (tangential)', 'm / s')
    W_mer = VarSpec('Wm', 'Relative Velocity (meridional)', 'm / s')
    FlowAngleRel = VarSpec('beta', 'Relative flow angle', 'rad')
    FlowAngleAbs = VarSpec('alpha', 'Absolute flow angle', 'rad')
    HDistr = VarSpec('hh', 'Height distribution', 'm')
    RDistr = VarSpec('rr', 'Radius distribution', 'm')
    # Scalars
    Rmid = VarSpec('Rmid', 'Mid radius', 'm', scalar=True)
    Rhub = VarSpec('Rhub', 'Hub radius', 'm', scalar=True)
    Rtip = VarSpec('Rtip', 'Tip radius', 'm', scalar=True)
    Omega = VarSpec('omega', 'Rotational speed', 'rad / s', scalar=True)
    Height = VarSpec('height', 'Channel height', 'm', scalar=True)
    MerAngle = VarSpec('mer_angle', 'Meridional angle', 'rad', scalar=True)
    BladeAngle = VarSpec('beta_bl', 'Blade angle', 'rad', scalar=True)


# NOTE: The inheritance enforces correct lsp type recognition
class NodeVariables(GenericVariables):
    def __init__(self, node: int):
        self._node = node

        self._generics = GenericVariables(node)

    def __getattribute__(self, name: str):
        if name in GenericVariables(-1):
            return getattr(self._generics, name).Hint
        else:
            return super().__getattribute__(name)

    @property
    def tot(self) -> ThermoVariables:
        return ThermoVariables(self._node, NodeStates.TOTAL)

    @property
    def stc(self) -> ThermoVariables:
        return ThermoVariables(self._node, NodeStates.STATIC)

    @property
    def rlt(self) -> ThermoVariables:
        return ThermoVariables(self._node, NodeStates.RELTOT)
