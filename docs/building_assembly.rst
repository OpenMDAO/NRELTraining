

Building a Group - Unconstrained Optimization for the Betz Limit
==================================================================
We're going to set up an optimization to look for Betz's limit. This is a well-known result that
states that for a wind turbine, as you try  to extract more and more velocity from the incoming
wind, the best you can do is to extract about 60% of the power from the wind. This result comes from
an analysis of the equations used to build  our ActuatorDisk component. We'll try to use an
optimizer to confirm that our component returns the correct value for Betz's limit.

Components are connected to form a larger model using a construct in OpenMDAO
called a ``Group``. Unlike components, groups can be nested to become part of
even larger groups. Although nesting groups often makes sense from an
organizational persepctive, there are sometimes some small computational and aesthetic
trade-offs to be considered. The main usage of groups is to encapsulate a
portion of a model in which you wish to use a specific nonlinear or linear
solver. However, this is beyond the scope of this tutorial.

Group files are comprised of the same three parts as a component files.

- Importing Libraries
- Class Definition
- Stand-Alone Testing

Generally, importing and testing work exactly the same as :ref:`components <BuildingAComponent>`.

Group Class Definition
-----------------------
A group class definition containes a minimum of two things:

- Class name declaration
- __init__ method

**Class name declaration**

As an example::

    from openmdao.api import Group

    class Betz_Limit(Group):
        """Simple wind turbine assembly to calculate the Betz Limit"""

Our class is named ``Betz_Limit`` and it inherents from the Group
base class that we imported above.

To define a group, the only method you normally need to specify is __init__.
The following are the things you typically do here:

- Adding component/group instances
- Connecting component variables
- Specifying nonlinear and linear solvers and modifying their settings

Let's walk through the rest of the Betz_Limit group.

**Adding component/group instances**

Although we've defined a component class we need to create an *instance* of the
object, just as we did in the :ref:`component testing script <ifNameEqualsMain>`.
If you're unfamiliar with object oriented programming, class definitions are
analogous to blueprints or templates for a design.
Instances represent actualized objects created from the blueprint or template.

As an example,
    ``jeff = teacher()``

    ``tristan = teacher()``

creates two instances, named ``jeff`` and ``tristan``, derived from the same
class: ``teacher``.

Likewise, this group class (or blueprint/template) definition contains
at least one of instance of a sub-group or sub-component.

::

    def __init__(self):
        super(Betz_Limit, self).__init__()

        self.add('aDisc', ActuatorDisc(), promotes=['a', 'Area', 'rho', 'Vu'])

Note that we need to use super to call the parent's __init__ function first.
After that, we use the `add` method to add to the group an instance of
ActuatorDisc, which we call 'aDisc'. The name we give a component needs to be
a unique valid Python name, so only upper case, lower case, numbers (though
it can't start with a number) and underscores are permitted. A local variable
is also created, so that 'self.aDisc' returns the instance.

The pomotes keyword is used for automatically connecting variables. It will
be discussed in more detail in a little bit.

For this problem, we would like to use all four params in the ActuatorDisc as
design variables. To make sure that these variables appear in the global
vector, and are thus available to be used as design variables, we need to
connect them to another source. OpenMDAO has a special kind of component --
the ``IndepVarComp`` for this purpose. As the name implies, it is a component
that provides an output that is controlled independently by an external
driver.

We need an IndepVarComp for each of our 4 params on the actuator dics component.

::

    self.add('p_a', IndepVarComp('a', 0.5), promotes=['*'])
    self.add('p_Area', IndepVarComp('Area', 10.0), promotes=['*'])
    self.add('p_rho', IndepVarComp('rho', 1.225), promotes=['*'])
    self.add('p_Vu', IndepVarComp('Vu', 10.0), promotes=['*'])

The basic syntax for an IndepVarComp is two arguments: the first is a name
for the output, and the second is the initial value. During optimization the
optimizer will start at the initial value before moving on. We also need to
give each IndepVarComp instance a unique name.

**Connecting variables**

For this model, we need for the outputs of the indepvarcomps to be connected
to the corresponding inputs on the ActuatorDisc. There are two ways to
connect in OpenMDAO: explicit and implicit.

To make an explicit connection, we would use the ``connect`` method, which
takes a source and a target variable pathname string as its arguments.

``self.connect('comp1.a', 'comp2.a')``

In this example, the value of output variable ``a`` from ``comp1`` would be
passed to ``comp2``'s input variable ``a``. With explicit connections, it is
not necessary for connected variables to share the same name. OpenMDAO will
automatically check for compatible units and raise an error if they are not
compatible.

However, in the code above, we have already connected them implicitly. When
adding our ActuatorDisc component to the Betz_Limit group, we specified the
optional 'promotes' argument and gave it a list of variable names.

::

    self.add('aDisc', ActuatorDisc(), promotes=['a', 'Area', 'rho', 'Vu'])

When you promote a variable, you make it available at the group level, so
that instead of it being accessible as "aDisc.rho", it becomes accessible as
"rho". You can promote any param or unknown on any component. If you promote
an unknown and a param with the same name, then they are connected
automatically. In our example code, we also promote the outputs of the
IndepVarComps, so OpenMDAO automatically connects those sources to the params
on the ActuatorDisc.

**Solvers and settings**

Since we don't have any cyclic data dependencies or implicit relationships,
we can use the default nonlinear and linear solvers. You can learn more about
the solvers available in OpenMDAO `here
<http://openmdao.readthedocs.io/en/1.6.4/srcdocs/packages/openmdao.solvers.html>`_.

We will be using SLSQP to solve our optimization problem, and this optimizer
requires a gradient. We haven't specified derivatives for our ActuatorDisc
component yet, so we need to prepare to estimate them with a full-model
finite difference. We can activate this via the `deriv_options` dictionary in
the group. The option 'type' is set to 'user' by default, which means the
derivatives are calculated by solving the unified derivatives equations. We
need to set this to 'fd' for finite difference.

::

    # The optimizer needs derivatives. We will finite difference the
    # whole model with central difference to provide these.
    self.deriv_options['type'] = 'fd'
    self.deriv_options['form'] = 'central'

In addition, we set the 'form' to central for increased accuracy at the cost
of an extra evaluation per derivative. We could alternatively set these same
options in the ActuatorDisc instance instead, which means that OpenMDAO would
just finite difference that component instead of the whole model. It makes
little difference though when our model only has one engineering component.


**Specifying the driver**

As we did in the previous example, we can set up an OpenMDAO Problem at the end of our file.

if __name__ == "__main__":

    prob = Problem()
    prob.root = Betz_Limit()

This time, we use the Betz_Limit group for our root. Next, we need to set up
the driver for optimization.

OpenMDAO provides a selection of optimization algorithms to drive a model
towards a specified objective. These drop-in algorithms are know as
*drivers*. The Problem has a default driver that runs everything once and
stops. You can specify a different driver by:

::

    prob.driver = ScipyOptimizer()

Many optimizers also contain tunable settings which can be found by searching
the `docs
<http://openmdao.readthedocs.io/en/1.6.4/srcdocs/packages/openmdao.drivers.html>`_.
The ScipyOptimizer is actually a wrapper around several optimizers. We want
to use the gradient optimizer SLSQP, so we specify it in the options:

    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1.0e-8

We also set a tighter tolerance (the default is 1.0e-6).

Both drivers and solvers must be given control of a variable that it can modify
to achieve the user specified goal.

::

    prob.driver.add_desvar('a', lower=0.0, upper=1.0)
    prob.driver.add_desvar('Area', lower=0.0, upper=1.0)
    prob.driver.add_desvar('rho', lower=0.0, upper=1.0)
    prob.driver.add_desvar('Vu', lower=0.0, upper=1.0)

    # Scaler -1.0 so that we maximize.
    prob.driver.add_objective('aDisc.Cp', scaler=-1.0)

In this example, the driver is allowed to vary all four design variables
between 0 and 1 until ``Cp`` is maximized. Optimizers by default will try to
minimize the objective, so we specify a 'scaler' of -1.0 to get a
maximization. Notice that promoted variables (such as 'Area') are always
refered to by their promoted name, but unpromoted variables (such as
'aDisc.Cp') are referenced with their full pathname.

We can also add *contraints* to the problem, though we don't necessarily need
one here. An example of a constraint that keeps aDisc.p above 0 is:

``prob.driver.add_constraint('aDisc.Cp', lower=0.0)``


Run the Optimization
---------------------------

To summarize, ``betz_limit.py`` is displayed in its entirety below:

.. testcode:: simple_assembly_betzlimit

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

Running ``python betz_limit.py`` from an activated terminal will show that the
optimizer found a value of approximately 1/3 for axial induction factor,
yielding a power coefficient just under .6. Congratulations! You have
just found Betz's limit. You can close down the project for now.


Finite Difference Options
--------------------------

Most optimization drivers will require derivative values to be estimated in order
to run. By default, OpenMDAO will automatically compute these via finite-difference
approximation. There are a few options for finite difference calculations which
are available.

These options can all be set to an optimization driver's `gradient_options` object.

::

    prob.root.deriv_options['form'] = 'central'
    prob.root.deriv_options['step_size'] = 1.0e-3
    prob.root.deriv_options['step_calc'] = 'relative'

You can also choose to calculate the derivatives with the Complex Step method
instead of finite difference. Its advantage over finite difference is that
its accuracty is not sensitive to the choice of stepsize. However, to use
this method, your model needs to be able to operate on complex inputs and
produce complex outputs. This will already be true of most python modules,
but your external codes may need special modification to use complex step.

::

    prob.root.deriv_options['type']= 'cs'








