# -*- coding: utf-8 -*-
from setuptools import setup,find_packages
import os,sys
import logging

developMode = False
if len(sys.argv) >= 2:
    if sys.argv[1] == "develop": developMode = True
if developMode:
    logging.warning("You have sleected a developer model ( local install)")


VERSION ="0.0.1"

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = None#parse_requirements("requirements.txt")


#------------------------- INSTALL--------------------------------------------
setup(name = 'pyCGM2-IntellEvent',
    version = VERSION,
    author = 'Fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    description = "pyCGM2-IntellEvent ",
    long_description= "",
    url = '',
    keywords = 'Gait events',
    packages=find_packages(),
	include_package_data=True,
    license='CC-BY-SA',
	install_requires = reqs,
    classifiers=['Programming Language :: Python',
                 'Operating System :: Microsoft :: Windows',
                 'Natural Language :: English'],
    entry_points={
          'console_scripts': [
                'intellevent_Server  =  intellevent.Apps.server.server_intellevent:main',
                'intellevent_Request  = intellevent.Apps.intelleventDetector:main',
          ]
      },
    )
