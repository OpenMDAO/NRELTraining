""" Find the Betz limit and record the optimization with an sql recorder."""


from openmdao.api import Problem, ScipyOptimizer, SqliteRecorder

from betz_limit import Betz_Limit


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

    recorder = SqliteRecorder('betz_limit.sql')
    #recorder.options['record_params'] = True
    recorder.options['record_resids'] = False

    prob.driver.add_recorder(recorder)

    prob.setup()
    prob.run()


