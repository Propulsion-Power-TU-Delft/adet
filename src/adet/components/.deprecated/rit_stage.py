import volute as VoluteModel

class RadialInflowTurbineStage:
    """ This class is used to define a radial-inflow turbine stage. \n
    :param components: ordered array of the components to compute \n
    :param EoS: equation of state object \n
    :param fluid: fluid name \n
    :param lib: thermodynamic library \n TODO: add options for thermodynamic library
    :param Pt: total inlet pressure in [Pa] \n
    :param Tt: total inlet temperature in [K] \n
    :param beta: total to static pressure ratio across the stage \n
    :param psi_is: isentropic work coefficient computed as Dhtt,is / Uin^2 \n
    :param phi_is: isentropic flow coefficient at the inlet of the impeller \n
    :param omega: rotational speed in [rad/s] \n
    :param mf: massflow in [kg/s] \n
    TODO: finish adding design variables
    """
    def __init__(self):
        self.ciao = 0

        self.user_input_check()

    def user_input_check(self):
        if 'imp' not in self.components.casefold():
            raise Exception ('The rit stage must have an impeller! Check your inputs.')

    def set(self, flag='design'):

        if flag.casefold() == 'design':
            # compute and assign boundary conditions
            for ii, component in enumerate(self.components.casefold()):
                if ii == 0:
                    if component == 'volute':
                        self.vol = VoluteModel(EoS, lib, fluid, shape, loss_model, verbose)
                        self.vol.inlet.Pt = self.Pt
                        self.vol.inlet.Tt = self.Tt
                    elif component == 'vaneless stator':
                        self.vls = VanelessModel()
                        self.vls.inlet.Pt = self.Pt
                        self.vls.inlet.Tt = self.Tt
                    elif component == 'vaned stator':
                        self.vst = VanedModel()
                        self.vst.inlet.Pt = self.Pt
                        self.vst.inlet.Tt = self.Tt
                    elif component == 'impeller':
                        self.imp = ImpellerModel()
                        self.imp.inlet.Pt = self.Pt
                        self.imp.inlet.Tt = self.Tt
                    else:
                        raise Exception ('The first component of a rit stage must '
                                         'be either one of the following: volute, '
                                         'vaneless stator, vaned stator, impeller.')
                if ii == len(self.components):
                    if component == 'impeller':
                        self.imp = ImpellerModel()
                        self.imp.outlet.P = self.P
                    elif component == 'diffuser':
                        self.dif = DiffuserModel()
                        self.dif.outlet.P = self.P
                    else:
                        raise Exception ('The last component of a rit stage must '
                                         'be either one of the following: diffuser, impeller.')



        elif flag.casefold() == 'offdesign':
            raise NotImplementedError