# Openmdao model for the Acutator Disc

from openmdao.api import Component


class ActuatorDisc(Component):
    """Simple wind turbine model based on actuator disc theory"""

    def __init__(self):
        super(ActuatorDisc, self).__init__()

        # Inputs
        self.add_param('a', 0.5, desc="Induced Velocity Factor")
        self.add_param('Area', 10.0, units="m**2", lower=0.0,
                       desc="Rotor disc area")
        self.add_param('rho', 1.225, units="kg/m**3",
                       desc="air density")
        self.add_param('Vu', 10.0, units="m/s",
                       desc="Freestream air velocity, upstream of rotor")

        # Outputs
        self.add_output('Vr', 0.0, units="m/s",
                        desc="Air velocity at rotor exit plane")
        self.add_output('Vd', 0.0, units="m/s",
                        desc="Slipstream air velocity, downstream of rotor")
        self.add_output('Ct', 0.0, desc="Thrust Coefficient")
        self.add_output('thrust', 0.0, units="N",
                        desc="Thrust produced by the rotor")
        self.add_output('Cp', 0.0, desc="Power Coefficient")
        self.add_output('power', 0.0, units="W",
                        desc="Power produced by the rotor")

    def solve_nonlinear(self, params, unknowns, resids):
        """ Considering the entire rotor as a single disc that extracts
        velocity uniformly from the incoming flow and converts it to
        power."""

        a = params['a']
        Vu = params['Vu']

        qA = .5*params['rho']*params['Area']*Vu**2

        unknowns['Vd'] = Vd = Vu*(1-2 * a)
        unknowns['Vr'] = .5*(Vu + Vd)

        unknowns['Ct'] = Ct = 4*a*(1-a)
        unknowns['thrust'] = Ct*qA

        unknowns['Cp'] = Cp = Ct*(1-a)
        unknowns['power'] = Cp*qA*Vu


if __name__ == "__main__":

    from openmdao.api import Problem, Group

    prob = Problem()
    prob.root = Group()
    prob.root.add('disc', ActuatorDisc(), promotes=['power', 'thrust', 'Cp'])

    prob.setup()
    prob.run()

    print('Power', prob['power'])
    print('Thrust', prob['thrust'])
    print('Cp', prob['Cp'])
