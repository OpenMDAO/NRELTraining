#
# This file is autogenerated during plugin quickstart and overwritten during
# plugin makedist. DO NOT CHANGE IT if you plan to use plugin makedist to update 
# the distribution.
#

from setuptools import setup, find_packages

kwargs = {'author': '',
 'author_email': '',
 'classifiers': ['Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering'],
 'description': '',
 'download_url': '',
 'entry_points': '[openmdao.component]\nnreltraining.actuator_disc_derivatives.ActuatorDisc=nreltraining.actuator_disc_derivatives:ActuatorDisc\nnreltraining.nreltraining.BladeElement=nreltraining.nreltraining:BladeElement\nnreltraining.nreltraining.ActuatorDisk=nreltraining.nreltraining:ActuatorDisk\nnreltraining.betz_limit.Betz_Limit=nreltraining.betz_limit:Betz_Limit\nnreltraining.nreltraining.BEMPerf=nreltraining.nreltraining:BEMPerf\nnreltraining.nreltraining.AutoBEM=nreltraining.nreltraining:AutoBEM\nnreltraining.actuator_disc.ActuatorDisc=nreltraining.actuator_disc:ActuatorDisc\nnreltraining.nreltraining.BEM=nreltraining.nreltraining:BEM\n\n[openmdao.container]\nnreltraining.actuator_disc_derivatives.ActuatorDisc=nreltraining.actuator_disc_derivatives:ActuatorDisc\nnreltraining.nreltraining.BEMPerfData=nreltraining.nreltraining:BEMPerfData\nnreltraining.nreltraining.BladeElement=nreltraining.nreltraining:BladeElement\nnreltraining.nreltraining.ActuatorDisk=nreltraining.nreltraining:ActuatorDisk\nnreltraining.nreltraining.FlowConditions=nreltraining.nreltraining:FlowConditions\nnreltraining.betz_limit.Betz_Limit=nreltraining.betz_limit:Betz_Limit\nnreltraining.nreltraining.BEMPerf=nreltraining.nreltraining:BEMPerf\nnreltraining.nreltraining.AutoBEM=nreltraining.nreltraining:AutoBEM\nnreltraining.actuator_disc.ActuatorDisc=nreltraining.actuator_disc:ActuatorDisc\nnreltraining.nreltraining.BEM=nreltraining.nreltraining:BEM',
 'include_package_data': True,
 'install_requires': ['openmdao.main'],
 'keywords': ['openmdao'],
 'license': '',
 'maintainer': '',
 'maintainer_email': '',
 'name': 'nreltraining',
 'package_data': {'nreltraining': ['sphinx_build/html/.buildinfo',
                                   'sphinx_build/html/bem.html',
                                   'sphinx_build/html/bem_design.html',
                                   'sphinx_build/html/building_assembly.html',
                                   'sphinx_build/html/building_component.html',
                                   'sphinx_build/html/derivatives.html',
                                   'sphinx_build/html/genindex.html',
                                   'sphinx_build/html/index.html',
                                   'sphinx_build/html/introduction.html',
                                   'sphinx_build/html/objects.inv',
                                   'sphinx_build/html/pkgdocs.html',
                                   'sphinx_build/html/py-modindex.html',
                                   'sphinx_build/html/recording_data.html',
                                   'sphinx_build/html/search.html',
                                   'sphinx_build/html/searchindex.js',
                                   'sphinx_build/html/srcdocs.html',
                                   'sphinx_build/html/_images/bem.png',
                                   'sphinx_build/html/_modules/index.html',
                                   'sphinx_build/html/_modules/nreltraining/actuator_disc.html',
                                   'sphinx_build/html/_modules/nreltraining/actuator_disc_derivatives.html',
                                   'sphinx_build/html/_modules/nreltraining/betz_limit.html',
                                   'sphinx_build/html/_modules/nreltraining/nreltraining.html',
                                   'sphinx_build/html/_modules/nreltraining/test/test_nreltraining.html',
                                   'sphinx_build/html/_sources/bem.txt',
                                   'sphinx_build/html/_sources/bem_design.txt',
                                   'sphinx_build/html/_sources/building_assembly.txt',
                                   'sphinx_build/html/_sources/building_component.txt',
                                   'sphinx_build/html/_sources/derivatives.txt',
                                   'sphinx_build/html/_sources/index.txt',
                                   'sphinx_build/html/_sources/introduction.txt',
                                   'sphinx_build/html/_sources/pkgdocs.txt',
                                   'sphinx_build/html/_sources/recording_data.txt',
                                   'sphinx_build/html/_sources/srcdocs.txt',
                                   'sphinx_build/html/_static/ajax-loader.gif',
                                   'sphinx_build/html/_static/basic.css',
                                   'sphinx_build/html/_static/comment-bright.png',
                                   'sphinx_build/html/_static/comment-close.png',
                                   'sphinx_build/html/_static/comment.png',
                                   'sphinx_build/html/_static/default.css',
                                   'sphinx_build/html/_static/doctools.js',
                                   'sphinx_build/html/_static/down-pressed.png',
                                   'sphinx_build/html/_static/down.png',
                                   'sphinx_build/html/_static/file.png',
                                   'sphinx_build/html/_static/jquery.js',
                                   'sphinx_build/html/_static/minus.png',
                                   'sphinx_build/html/_static/plus.png',
                                   'sphinx_build/html/_static/pygments.css',
                                   'sphinx_build/html/_static/searchtools.js',
                                   'sphinx_build/html/_static/sidebar.js',
                                   'sphinx_build/html/_static/underscore.js',
                                   'sphinx_build/html/_static/up-pressed.png',
                                   'sphinx_build/html/_static/up.png',
                                   'sphinx_build/html/_static/websupport.js',
                                   'test/.gitignore',
                                   'test/__init__.py',
                                   'test/test_nreltraining.py']},
 'package_dir': {'': 'src'},
 'packages': ['nreltraining', 'nreltraining.test'],
 'url': '',
 'version': '2.0',
 'zip_safe': False}


setup(**kwargs)

