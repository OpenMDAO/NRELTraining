=============================================================
Using the OpenMDAO derivatives system
=============================================================

Many optimization algorithms make use of gradients. By default, OpenMDAO will
finite difference it during the calculation of the full model gradient.
However, an optimization process can often be sped up by having a component
supply its own derivatives.

In OpenMDAO, derivatives can be specified in the component API by following
this one step:

    1. Define a ``linearize`` function method that calculates and returns the Jacobian.

Let's return to the actuator disc component, and show how derivatives can be specified:

.. testcode:: simple_component_actuatordisk_review

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


Let's create a copy of this file called ``actuator_disc_derivatives.py``, that
we will use to implement derivatives. Having it as a separate file from the
derivates-free actuator disc file will allow for easy comparison later.

Now, we just need a function called ``linearize`` that calculates and returns
a matrix (the Jacobian) of the derivatives, evaluated at the current state of
the model. OpenMDAO expects the Jacobian to be a dictionary whose keys are a
tuple of (output, input) pairs. This structure is "sparse" in that if an
output-input pair is not present, it is assumed to be zero, so we only need
to specify the ones that are there.

.. testcode:: simple_component_actuatordisk_provideJ

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


Running the Optimization, with derivatives
-------------------------------------------

To summarize, ``actuator_disc_derivatives.py`` is displayed in its entirety below:

.. testcode:: simple_assembly_betzlimit

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


Modify ``betz_limit.py`` to import ``ActuatorDisc`` from
``actuator_disc_derivatives.py`` instead of ``actuator_disc.py``. Dont'
forget to change the deriv_options 'type' setting to 'user' so that it solves
the unified derivatives equations for the gradient.

::

  self.deriv_options['type'] = 'user'

Running ``python betz_limit.py`` we see that the optimization takes less time
when derivatives are provided.
