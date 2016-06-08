""" Read in data from the file `betz_limit.sql` and plot it"""

import sqlitedict

import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load the database
    db = sqlitedict.SqliteDict('betz_limit.sql', 'openmdao')

    # show all the recorded case name
    print db.keys()

    # show all the case variable names for the last case
    case = db['rank0:SLSQP/9']
    print case.keys()

    # show all the variable names for the last case
    unknowns = case['Unknowns']
    print unknowns.keys()

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