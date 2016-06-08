.. _`BuildingAComponent`:

=============================================================
Building a Component - Actuator Disc
=============================================================

In this tutorial, we're going to define a component that uses
actuator disc theory to provide a very simple model of a wind turbine.
We will reproduce an engineering design limitation known as the Betz limit.

.. image:: actuator_disk.png
    :width: 750 px
    :align: center


A component takes a set of inputs and operates on them to produce a set of
outputs. In the OpenMDAO framework, a class called *Component*
provides this behavior. Component definition files are usually comprised of
three parts that are explained in depth below.

- Importing Libraries
- Class Definition
- Stand-Alone Testing

Importing Libraries
=========================================

In Python, a class or function must be imported before it can be used. So
firstly, we will import some helpful libraries from OpenMDAO.

Following standard python, specific classes can be imported with the following syntax:
``from <library> import <Class>``

Optionally, imports can be assinged a nickname or alias:
``from <library> import <Class> as <alias>``

::

    from openmdao.api import Component


**Avoid** importing entire libraries,
this slows down your code and can cause namespace collisions.

::

    import openmdao.api #This is bad.
    from openmdao.api import * #Don't do this.

Most of what you need in OpenMDAO can be imported from the
``openmdao.api`` module.
For this example we will only import the base OpenMDAO ``Component`` class.
This provides us with basic boilerplate
functionality useful for constructing engineering models.

.. _`ComponentDefinition`:

Component Class Definition
=========================================
A class definition containes a minimum of three things:

- Class name declaration
- Input (param) and output variable definition
- Execute method

There are more advanced features that can be added to components,
but this is a good starting point.

**Class name declaration**

Just as you would define a class in standard python, you follow the syntax

``class <Name>(<ParentClass>):``
        ``"""<Description>"""``

As an example::

    class ActuatorDisc(Component):
        """Simple wind turbine model based on actuator disc theory"""

Our class is named ``ActuatorDisc`` and it inherents from the Component
base class that we imported above. The second line containing triple quotes
provides a brief description of the class.

**Input/Output Variable Definition**

Next, the inputs and outputs of the component class must be defined in the ``__init__`` method.

::

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

The I/O variables shown in this snippet are declared using ``add_param`` for
inputs and ``add_output`` for outputs. All variables are given a name and an
initial value. OpenMDAO does not enforce datatype, but it undersands the
distinction between floating point numbers (which can be differentiated) and
any other type (which can't.) Note that a default value (or alternatively a
'shape' attribute) are required in most cases.

Other optional arguments can also be added to a variable, including 'units',
which is supported by automatic unit conversion and checking, and 'lower' or
'upper', which set bounds that a solver will observe if it supports them.
Also, the 'desc' attribute allows you to specify a docstring.

Specifying the physical units (if applicable) allows openMDAO to validate
and automatically convert between compatible units when connecting variables across other components.
A list of valid unit types can be found `here <http://openmdao.org/docs/units.html>`_.

**solve_nonlinear Method**

Next, the ``solve_nonlinear`` method contains the main calculations and engineering
operations of the component.

::

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

As standard Python convention, this internal method inherits from ``self``
(Actuator Disc), and all I/O variables can be found in the dictionary-like
objects 'params' (for params) and 'unknowns' (for unknowns.) Local variables
can be defined for convenience and readability, as done with ``a``, ``Vu``,
and ``qA``. These variables can only be called locally from within the method
and any references to local variables outside of this method will throw
errors. Only I/O variables from the class definition will be accessible
elsewhere.

In this particular execution method, we treat the entire rotor as a single disc
that extracts velocity uniformly from the incoming flow and converts it to
power. If you define the upstream, rotor, and downstream velocities as
:math:`V_u`, :math:`V_r`, :math:`V_d` respectively, then you can describe the
axial induction factor, :math:`a`, as the amount of velocity extracted from the
flow. :math:`a = \frac{V_u-V_r}{V_r}`

.. _`ifNameEqualsMain`:

Stand-Alone Testing
=========================================

The final (optional) section of a component generally includes a script that
allows you to run your component by itself.
This is often helpful for debugging as you build up your model.

::

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

In this snippet, the ``if __name__ == "__main__":`` is a common Python pattern
that in plain english means "if this file is called directly, run the following commands"
This section is ignored if the component class is instantiated elsewhere.

An OpenMDAO model is built using a single ``Problem``, which is a Python
object that contains methods that let you setup, run, or check derivatives on
your model. The ``Problem`` contains a ``Group`` called root. A ``Group`` is
just a ``System`` that can contain Components or Groups. We would like to
test the ActuatorDisc running by itself, so we create an empty Group and add
our component to it. Groups will be discussed further in the next section.

In this particular run script, an *instance* of the ActuatorDisc class is
created called ``disc`` and added to the root group of a problem. Then,
``setup`` is called on the problem; this method prepares the model for
execution, allocating the vectors needed for data passing and calculating
derivatives if needed. Finally, the problem is ``run``, and three outputs are
printed to the console.

To summarize, ``actuator_disc.py`` is displayed in its entirety below:

.. testcode:: simple_component_actuatordisc

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

Running your Component
=========================================
To run the component and ensure your getting the expected output,
open an activated terminal window and navigate the parent folder of
this file. Simply run:

::

    python actuator_disc.py


Thats it! You've built and ran your first OpenMDAO component.

With legacy code written in other language, such as Fortran or C/C++,
components can also contain wrappers.
The `Plugin-Developer-Guide <http://openmdao.org/docs/plugin-guide/index.html>`_
gives some examples of how to incorporate these kinds of components into OpenMDAO.
