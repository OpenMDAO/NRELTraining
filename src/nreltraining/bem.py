# Blade Rotor with a specified number of blade elements.
#
# INCOMPLETE : may not have been completed for the tutorial course.
#


from math import pi, cos, sin, tan

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

from openmdao.api import Group, Component


class BladeElement(Component):
    """Calculations for a single radial slice of a rotor blade"""

    def __init__(self):
        super(BladeElement, self).__init__()

        # Inputs
        self.add_param('a_init', 0.2,
                       desc="initial guess for axial inflow factor")
        self.add_param('b_init', 0.01,
                       desc="initial guess for angular inflow factor")
        self.add_param('rpm', 106.952, lower=0, units="min**-1",
                       desc="rotations per minute")
        self.add_param('r', 5.0, units="m",
                       desc="mean radius of the blade element")
        self.add_param('dr', 1.0, units="m",
                       desc="width of the blade element")
        self.add_param('twist', 1.616, units="rad",
                       desc="local twist angle")
        self.add_param('chord', .1872796, lower=0, units="m",
                       desc="local chord length")
        self.add_param('B', 3,
                       desc="Number of blade elements")

        self.add_param('rho', 1.225, units="kg/m**3",
                       desc="air density")
        self.add_param('V_inf', 7.0, units="m/s",
                       desc="free stream air velocity")

        # Outputs
        self.add_output(V_0, 0.0, units="m/s",
                        desc="axial flow at propeller disk")
        self.add_output(V_1, 0.0, units="m/s",
                        desc="axial local flow velocity")
        self.add_output(V_2, 0.0, units="m/s",
                        desc="angular flow at propeller disk")
        self.add_output(omega, 0.0, units="rad/s",
                        desc="average angular velocity for element")
        self.add_output(sigma, 0.0,
                        desc="Local solidity")
        self.add_output(alpha, 0.0, units="rad",
                        desc="local angle of attack")
        self.add_output(delta_Ct, 0.0, units="N",
                        desc="section thrust coefficient")
        self.add_output(delta_Cp, 0.0,
                        desc="section power coefficent")
        self.add_output(a, 0.0,
                        desc="converged value for axial inflow factor")
        self.add_output(b, 0.0,
                        desc="converged value for radial inflow factor")
        self.add_output(lambda_r, 8.0,
                        desc="local tip speed ratio")
        self.add_output(phi, 1.487, units="rad",
                        desc="relative flow angle onto blades")

        # rough linear interpolation from naca 0012 airfoil data
        rad = np.array([0., 13., 15, 20, 30])*pi/180
        self.cl_interp = interp1d(rad, [0., 1.3, .8, .7, 1.1], fill_value=0.001,
                                  bounds_error=False)

        rad = np.array([0., 10, 20, 30, 40])*pi/180
        self.cd_interp = interp1d(rad, [0., 0., 0.3, 0.6, 1.], fill_value=0.001,
                                  bounds_error=False)

    def _coeff_lookup(self, i):
        C_L = self.cl_interp(i)
        C_D = self.cd_interp(i)
        return C_D, C_L

    def solve_nonlinear(self, params, unknowns, resids):

        B = params['B']
        chord = params['chord']
        dr = params['dr']
        r = params['r']
        rho = params['rho']
        rpm = params['rpm']
        V_inf = params['V_inf']

        sigma = B*chord / (2.0 * np.pi * r)
        omega = rpm*2*pi/60.0
        omega_r = omega*r
        unknowns['lambda_r'] = omega*r/V_inf  # need lambda_r for iterates

        a, b = fsolve(self._iteration, [params['a_init'], params['b_init']],
                      args = [params, unknowns])

        V_0 = V_inf - a*V_inf
        V_2 = omega_r-b*omega_r
        V_1 = (V_0**2 + V_2**2)**.5

        phi = unknowns['phi']
        alpha = unknowns['alpha']

        q_c = B*.5*(rho*V_1**2)*chord*dr
        cos_phi = cos(phi)
        sin_phi = sin(phi)
        C_D, C_L = self._coeff_lookup(alpha)
        unknowns['delta_Ct'] = q_c*(C_L*cos_phi-C_D*sin_phi)/(.5*rho*(V_inf**2)*(pi*r**2))
        unknowns['delta_Cp'] = b*(1.0-a)*unknowns['lambda_r']**3*(1.0-C_D/C_L*tan(phi))

        unknowns['sigma'] = sigma
        unknowns['omega'] = omega
        unknowns['a'] = a
        unknowns['b'] = b
        unknowns['V_0'] = V_0
        unknowns['V_1'] = V_1
        unknowns['V_2'] = V_2

    def _iteration(self, X, params, unknowns):

        twist = params['twist']
        sigma = params['sigma']
        lambda_r = unknowns['lambda_r']

        phi = np.arctan(lambda_r*(1 + X[1])/(1 - X[0]))
        alpha = pi/2 - twist - phi
        C_D, C_L = self._coeff_lookup(alpha)
        a = 1./(1 + 4.*(np.cos(phi)**2)/(sigma*C_L*np.sin(phi)))
        b = (sigma*C_L) / (4.0 * lambda_r * np.cos(phi)) * (1 - a)

        unknowns['phi'] = phi
        unknowns['alpha'] = alpha

        return (X[0] - a), (X[1] - b)


class BEMPerf(Component):
    """collects data from set of BladeElements and calculates aggregate values"""

    def __init__(self, n=10):
        """Collects data from set of BladeElements and calculates aggregate values.

        Args
        ----
        n : int
            Number of elements.
        """
        super(BEMPerf, self).__init__()

        self.add_param('r', 0.8, units='m',
                       desc="tip radius of the rotor")
        self.add_param('rpm', 2100, lower=0.0, units="min**-1",
                       desc="rotations per minute")

        # "Variable Tree" FlowConditions
        self.add_param('free_stream:rho', 1.225, units="kg/m**3",
                       desc="air density")
        self.add_param('free_stream:V', 7.0, units="m/s",
                       desc="free stream air velocity")

        # "Variable Tree" BEMPerfData
        self.add_output('data:net_thrust', 0.0, units="N",
                        desc="net axial thrust")
        self.add_output('data:net_power', 0.0, units="W",
                        desc="net power produced")
        self.add_output('data:Ct', 0.0,
                        desc="thrust coefficient")
        self.add_output('data:Cp', 0.0,
                        desc="power coefficient")
        self.add_output('data:J', 0.0,
                        desc="advance ratio")
        self.add_output('data:tip_speed_ratio', 0.0,
                        desc="tip speed ratio")
        self.add_output('data:eta', 0.0,
                        desc="turbine efficiency")

        # array size based on number of elements
        self.add_param('delta_Ct', np.ones((n, )), units="N",
                       desc='thrusts from %d different blade elements' % n)
        self.add_param('delta_Cp', np.ones((n, )),
                       desc='Cp integrant points from %d different blade elements' % n)
        self.add_param('lambda_r', np.ones((n, )),
                       desc='lambda_r from %d different blade elements' % n)

    def solve_nonlinear(self, params, unknowns, resids):

        V_inf = params['free_stream:V']
        rho = params['free_stream:rho']
        r = params['r']
        rpm = params['rpm']
        delta_Ct = params['delta_Ct']
        lambda_r = params['lambda_r']

        norm = (.5*rho*(V_inf**2)*(pi*r**2))
        Ct = np.trapz(delta_Ct, x=lambda_r)
        unknowns['data:net_thrust'] = Ct*norm

        Cp = np.trapz(params['delta_Cp'], x=lambda_r) * 8. / lambda_r.max()**2
        unknowns['data:net_power'] = Cp*norm*V_inf

        unknowns['data:J'] = V_inf/(rpm/60.0*2*r)

        omega = rpm*2*pi/60
        unknowns['data:tip_speed_ratio'] = omega*r/V_inf

        unknowns['data:Ct'] = Ct
        unknowns['data:Cp'] = Cp


class AutoBEM(Group):
    """Blade Rotor with user specified number BladeElements"""

    def __init__(self, n_elements=6):
        """Blade Rotor with user specified number BladeElements.

        Args
        ----
        n_elements : int
            Number of elements.
        """
        super(AutoBEM, self).__init__()

        ## physical properties inputs
        #r_hub = Float(0.2, iotype="in", desc="blade hub radius", units="m", low=0)
        #twist_hub = Float(29, iotype="in", desc="twist angle at the hub radius", units="deg")
        #chord_hub = Float(.7, iotype="in", desc="chord length at the rotor hub", units="m", low=.05)
        #r_tip = Float(5, iotype="in", desc="blade tip radius", units="m")
        #twist_tip = Float(-3.58, iotype="in", desc="twist angle at the tip radius", units="deg")
        #chord_tip = Float(.187, iotype="in", desc="chord length at the rotor hub", units="m", low=.05)
        #pitch = Float(0, iotype="in", desc="overall blade pitch", units="deg")
        #rpm = Float(107, iotype="in", desc="rotations per minute", low=0, units="min**-1")
        #B = Int(3, iotype="in", desc="number of blades", low=1)

        ## wind condition inputs
        #free_stream = VarTree(FlowConditions(), iotype="in")
        #self.add('free_stream', VarTree(FlowConditions(), iotype="in"))  # initialize

        self.add('driver', SLSQPdriver())

        self.add('radius_dist', LinearDistribution(n=n_elements, units="m"))
        self.connect('r_hub', 'radius_dist.start')
        self.connect('r_tip', 'radius_dist.end')

        self.add('chord_dist', LinearDistribution(n=n_elements, units="m"))
        self.connect('chord_hub', 'chord_dist.start')
        self.connect('chord_tip', 'chord_dist.end')

        self.add('twist_dist', LinearDistribution(n=n_elements, units="deg"))
        self.connect('twist_hub', 'twist_dist.start')
        self.connect('twist_tip', 'twist_dist.end')
        self.connect('pitch', 'twist_dist.offset')

        self.driver.workflow.add('chord_dist')
        self.driver.workflow.add('radius_dist')
        self.driver.workflow.add('twist_dist')

        self.add('perf', BEMPerf(n=n_elements))
        self.create_passthrough('perf.data')
        self.connect('r_tip', 'perf.r')
        self.connect('rpm', 'perf.rpm')
        self.connect('free_stream', 'perf.free_stream')

        for i in range(n_elements):

            name = 'BE%d' % i
            self.add(name, BladeElement())
            self.driver.workflow.add(name)

            self.connect('radius_dist.output[%d]' % i, name+'.r')
            self.connect('radius_dist.delta', name+'.dr')
            self.connect('twist_dist.output[%d]' % i, name+'.twist')
            self.connect('chord_dist.output[%d]' % i, name+".chord")

            self.connect('B', name+'.B')
            self.connect('rpm', name+'.rpm')

            self.connect('free_stream.rho', name+'.rho')
            self.connect('free_stream.V', name+'.V_inf')
            self.connect(name+'.delta_Ct', 'perf.delta_Ct[%d]' % i)

            self.connect(name+'.delta_Cp', 'perf.delta_Cp[%d]' % i)
            self.connect(name+'.lambda_r', 'perf.lambda_r[%d]' % i)

        self.driver.workflow.add('perf')

        # set up optimization
        self.driver.add_parameter('chord_hub', low=.1, high=2)
        self.driver.add_parameter('chord_tip', low=.1, high=2)
        self.driver.add_parameter('twist_hub', low=-5, high=50)
        self.driver.add_parameter('twist_tip', low=-5, high=50)
        self.driver.add_parameter('rpm', low=20, high=300)
        self.driver.add_parameter('r_tip', low=1, high=10)

        self.driver.add_objective('-data.Cp')


if __name__ == "__main__":

    top = AutoBEM(6)

    # set up case recording
    JSON_recorder = JSONCaseRecorder('bem.json')
    top.recorders = [JSON_recorder]

    top.run()

    print
    print
    print "rpm:", top.rpm
    print "cp:", top.data.Cp
    print 'top.b.chord_hub: ', top.chord_hub
    print 'top.b.chord_tip: ', top.chord_tip
    print 'lambda: ', top.perf.data.tip_speed_ratio

    cds = CaseDataset("bem.json", 'json')
    caseset_query_to_html(cds.data)
