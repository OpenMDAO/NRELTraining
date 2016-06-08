#
# This file is autogenerated during plugin quickstart and overwritten during
# plugin makedist. DO NOT CHANGE IT if you plan to use plugin makedist to update
# the distribution.
#

from setuptools import setup, find_packages

kwargs = {'author': 'Kenneth T. Moore',
 'author_email': 'kenneth.t.moore-1@nasa.gov',
 'classifiers': ['Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering'],
 'description': 'OpenMDAO component wrapper for FLOPS',
 'download_url': '',
 'include_package_data': True,
 'install_requires': ['openmdao'],
 'keywords': ['openmdao'],
 'license': 'Apache License, Version 2.0',
 'maintainer': 'Kenneth T. Moore',
 'maintainer_email': 'kenneth.t.moore-1@nasa.gov',
 'name': 'nreltraining',
 'package_data': {'nreltraining': ['sphinx_build/html/*.html',]},
 'package_dir': {'': 'src'},
 'packages': ['nreltraining', 'nreltraining.test'],
 'url': '',
 'version': '1.0',
 'zip_safe': False}


setup(**kwargs)

