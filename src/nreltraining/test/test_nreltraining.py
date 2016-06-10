
import unittest

from openmdao.api import Problem, Group, ScipyOptimizer, IndepVarComp
from openmdao.test.util import assert_rel_error

from nreltraining.bem import AutoBEM
from nreltraining.betz_limit import Betz_Limit


class ActuatorDiskTestCase(unittest.TestCase):

    def test_ActuatorDisk(self):

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

        prob.run()

        assert_rel_error(self, prob['a'], 0.333, 0.005)
        assert_rel_error(self, prob['aDisc.Cp'], 0.593, 0.005)  # Betz Limit



class AutoBEMTestCase(unittest.TestCase):

    def setUp(self):
        self.top = Problem()
        self.top.root = AutoBEM(6)

    def tearDown(self):
        self.top = None

    #def test_AutoBEM_DOE(self):
        ## perform a DOE
        #self.top.replace('driver', DOEdriver())
        #self.top.driver.DOEgenerator = FullFactorial(3)

        #self.top.driver.add_parameter('b.chord_hub', low=.1, high=2)
        #self.top.driver.add_parameter('b.chord_tip', low=.1, high=2)
        #self.top.driver.add_parameter('b.rpm',       low=20, high=300)
        #self.top.driver.add_parameter('b.twist_hub', low=-5, high=50)
        #self.top.driver.add_parameter('b.twist_tip', low=-5, high=50)

        #self.top.run()

        #self.assertEqual(self.top.b.exec_count, 243)

    def test_AutoBEM_Opt(self):
        # perform optimization

        top = self.top
        driver = top.driver = ScipyOptimizer()
        driver.options['optimizer'] = 'SLSQP'

        # set up optimization
        driver.add_desvar('chord_hub', lower=.1, upper=2)
        driver.add_desvar('chord_tip', lower=.1, upper=2)
        driver.add_desvar('twist_hub', lower=-5.0, upper=50.0)
        driver.add_desvar('twist_tip', lower=-5.0, upper=50.0)
        driver.add_desvar('rpm', lower=20.0, upper=300.0)
        driver.add_desvar('r_tip', lower=1.0, upper=10.0)

        driver.add_objective('data:Cp', scaler=-1.0)

        # For now, FD the whole model at once
        top.root.deriv_options['type'] = 'fd'

        top.setup()
        top.run()

        assert_rel_error(self, top['data:Cp'], 0.57, 0.01)


if __name__ == '__main__':
    unittest.main()

