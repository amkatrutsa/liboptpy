from setuptools import setup

setup(
    name='liboptpy',
    version='0.0.1',
    description='Implementation of various optimization methods for research and study purposes',
    author='Alexandr Katrutsa',
    author_email="aleksandr.katrutsa@phystech.edu",
    packages=['liboptpy', 'liboptpy.unconstr_solvers','liboptpy.unconstr_solvers.fo', 'liboptpy.unconstr_solvers.so',
             'liboptpy.constr_solvers'],
    install_requires=['numpy>=1.12', 'scipy>=1.0'],
    keywords=[ 'Convex optimization', 'numerical optimization', 
              'Python', 'Numpy', 'Scipy'],
    url='https://github.com/amkatrutsa/liboptpy',
    license='MIT',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.5'],
)
