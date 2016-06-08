# Openmdao model for the Acutator Disc, this version has derivatives defined.

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

    def linearize(self, params, unknowns, resids):
        """ Jacobian of partial derivatives."""

        a = params['a']
        Vu = params['Vu']
        Area = params['Area']
        rho = params['rho']
        J = {}

        # pre-compute commonly needed quantities
        a_times_area = a*Area
        one_minus_a = 1.0 - a
        a_area_rho_vu = a_times_area*rho*Vu

        J['Vr', 'a'] = -Vu
        J['Vr', 'Vu'] = one_minus_a

        J['Vd', 'a'] = -2.0*Vu
        J['Vd', 'Vo'] = 1.0 - 2.0*a

        J['Ct', 'a'] = 4.0 - 8.0*a

        J['thrust', 'a'] = Area*Vu**2 * rho*(-4.0*a + 2.0)
        J['thrust', 'Area'] = 2.0*Vu**2 * a*rho*one_minus_a
        J['thrust', 'rho']  = 2.0*a_times_area*Vu**2*(one_minus_a)
        J['thrust', 'Vu'] = 4.0*a_area_rho_vu*(one_minus_a)

        J['Cp', 'a'] = 4.0*a*(2.0*a - 2.0) + 4.0*(one_minus_a)**2

        J['power', 'a'] = 2.0*Area*Vu**3 * a*rho*(2.0*a - 2.0) + 2.0*Area*Vu**3 * rho*one_minus_a**2
        J['power', 'Area'] = 2.0*Vu**3*a*rho*one_minus_a**2
        J['power', 'rho'] = 2.0*a_times_area*Vu**3 * (one_minus_a)**2
        J['power', 'Vu'] = 6.0*Area*Vu**2 * a*rho*one_minus_a**2

        return J


if __name__ == "__main__":

    from openmdao.api import Problem, Group, IndepVarComp

    prob = Problem()
    prob.root = Group()
    prob.root.add('disc', ActuatorDisc(), promotes=['a', 'Area', 'rho', 'Vu'])

    prob.root.add('p_a', IndepVarComp('a', 0.5), promotes=['*'])
    prob.root.add('p_Area', IndepVarComp('Area', 10.0), promotes=['*'])
    prob.root.add('p_rho', IndepVarComp('rho', 1.225), promotes=['*'])
    prob.root.add('p_Vu', IndepVarComp('Vu', 10.0), promotes=['*'])

    prob.setup()
    prob.run()

    prob.check_partial_derivatives()


