# Simple OpenMDAO problem that calculates the Betz limit on a wind turbine


import time

from openmdao.api import Problem, Group, ScipyOptimizer, IndepVarComp

#Import components from the plugin
from actuator_disc import ActuatorDisc


class Betz_Limit(Group):
    """Simple wind turbine assembly to calculate the Betz Limit"""

    def __init__(self):
        super(Betz_Limit, self).__init__()

        self.add('aDisc', ActuatorDisc(), promotes=['a', 'Area', 'rho', 'Vu'])

        self.add('p_a', IndepVarComp('a', 0.5), promotes=['*'])
        self.add('p_Area', IndepVarComp('Area', 10.0), promotes=['*'])
        self.add('p_rho', IndepVarComp('rho', 1.225), promotes=['*'])
        self.add('p_Vu', IndepVarComp('Vu', 10.0), promotes=['*'])

        # The optimizer needs derivatives. We will finite difference the
        # whole model with central difference to provide these.
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'


if __name__ == "__main__":

    prob = Problem()
    prob.root = Betz_Limit()

    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1.0e-8

    prob.driver.add_desvar('a', lower=0.0, upper=1.0)
    prob.driver.add_desvar('Area', lower=0.0, upper=1.0)
    prob.driver.add_desvar('rho', lower=0.0, upper=1.0)
    prob.driver.add_desvar('Vu', lower=0.0, upper=1.0)

    # Scaler -1.0 so that we maximize.
    prob.driver.add_objective('aDisc.Cp', scaler=-1.0)

    prob.setup()

    t = time.time()
    prob.run()
    print "time:", time.time() - t
    print
    print "Cp:", prob['aDisc.Cp']