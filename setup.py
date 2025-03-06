""" Q) What is setup.py?
In Python, setup.py is a module used to build and distribute Python packages. 
It typically contains information about the package, such as its name, version, and dependencies,
as well as instructions for building and installing the package. This information is used by the
pip tool, which is a package manager for Python that allows users to install and manage Python 
packages from the command line. By running the setup.py file with the pip tool, you can build and 
distribute your Python package so that others can use it.

"""

from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_req(filename:str) -> List[str]:
   """
   this function will return the list of requirements
   """
   requirement = []
   with open(filename) as f:
      requirement = f.readlines()
      requirement = [line.replace("\n","") for line in requirement]
      
      ## We also need to remove hyphen-e-dot
      if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
      
   return requirement


setup(
   name='loan-prediction',
   version='1.0',
   packages=find_packages(),
   author="Koelin",
   author_email= "koelinkrishh@gmail.com",
   install_requires=get_req('requirements.txt')
)







