Recording Data from your Runs
=============================================================

Running an optimization in OpenMDAO is great, but it's not really useful
unless you can record the results. In OpenMDAO, data recording is done
through a special type of object called a `recorder.` OpenMDAO comes with a
few different kinds of recorders but you could also design your own if you
have special needs.

The recommended general purpose recorder is the ``SqliteRecorder``, which
records your data in an SQLite dictionary.


Setting Up Case Recording
-------------------------------------------------------------

We'll use the Betz limit optimization in order to demonstrate case recording,
outputing all of the case data in SQL format.

To do this, import the `Betz_limit` assembly, as well as OpenMDAO's SQL recorder:

::

    from openmdao.api import Problem, ScipyOptimizer, SqliteRecorder

    from betz_limit import Betz_Limit


Now, create a Problem that optimizes our Betz_Limit group as we did before.

::

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


Next, create instances of the SQL case recorder. As an argument,
both take the filename of where you would like the outputted data to be recorded:

::

    recorder = SqliteRecorder('betz_limit.sql')


Set both recorders to the optimizer by placing them both within the driver's
recorder list via the ``add_recorder`` method:

::

    prob.driver.add_recorder(recorder)


Finally, setup and run the optimization just as before:

::

    prob.setup()
    prob.run()


This will populate the SQL file set above with the case data from the
optimization.


Reading Data From The Recorder Output
------------------------------------------

In the example above, case data was written to SQL file. This is not a
human-readable format, but there is a tool called ``sqlitedict`` that can be
used to read in the data and access it as if it were a Python dictionary.

::

    import sqlitedict

    # Load the database
    db = sqlitedict.SqliteDict('betz_limit.sql', 'openmdao')

Now, db has access to everything in the SQL file. To find out what is in it:

::

    print(db.keys())

which prints out:

::

    ['metadata', 'rank0:SLSQP/1', 'rank0:SLSQP/1/derivs', 'rank0:SLSQP/2', 'rank0:SLSQP/3', 'rank0:SLSQP/3/derivs', 'rank0:SLSQP/4', 'rank0:SLSQP/5', 'rank0:SLSQP/5/derivs', 'rank0:SLSQP/6', 'rank0:SLSQP/7', 'rank0:SLSQP/7/derivs', 'rank0:SLSQP/8', 'rank0:SLSQP/9', 'rank0:SLSQP/9/derivs']

The first key called 'metadata' can be used to access some problem metadata
that was saved at the start of the run. The remaining keys are for cases that
were recorded. The name 'rank0:SLSQP/1' includes the processor rank for
multiprocessing (always 'rank0' for serial runs), the driver name ('SLSQP')
and the iteration number (this is the first iteration, so 1.)

If we take a look at one of these cases,

::

    case = db['rank0:SLSQP/9']
    print(case.keys())

we find that it is also a dictionary with the following keys:

::

    ['timestamp', 'success', 'msg', 'Unknowns']

Here, 'timestamp' is the time this iteration completed, 'success' is a flag
that indicates if that case executed succesfully, 'msg' contains any error
message that was raised, and 'unknowns' contains the data.

If we drill down a little bit further

::

    unknowns = case['Unknowns']
    print(unknowns.keys())

We get a list of variables that were saved for this case.

::

    ['a', 'Area', 'aDisc.Ct', 'aDisc.Vr', 'aDisc.Cp', 'aDisc.power', 'Vu', 'rho', 'aDisc.Vd', 'aDisc.thrust']

One final dictionary access gives us the value of any variable for this iteration.

Finally, let us write some code so that we can grab all of the data and pull out some points to plot:

::

    import sqlitedict
    import matplotlib.pyplot as plt

    # Now, let's make a function that makes it easier to pull data from all the
    # case points.
    def extract_all_vars_sql(name):
        """ Reads in the file given in name and extracts all variables."""

        db = sqlitedict.SqliteDict( name, 'openmdao' )

        data = {}
        for iteration in range(len(db)-1):
            iteration_coordinate = 'rank0:SLSQP/{}'.format(iteration + 1 )

            try:
                record = db[iteration_coordinate]
            except KeyError:
                break

            for key, value in record['Unknowns'].items():
                if key not in data:
                    data[key] = []
                data[key].append(value)

        return data

    # Pick some that we want
    data = extract_all_vars_sql('betz_limit.sql')
    a = data['a']
    Cp = data['aDisc.Cp']


    # Finally make some plots
    for area, cp in zip(a, Cp):
        plt.plot(area, cp, "ko")
    plt.xlabel("a")
    plt.ylabel("Cp")
    plt.show()











