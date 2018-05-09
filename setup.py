from setuptools import setup

setup(
    name='liboptpy',
    version='0.0.1',
    description='Implementation of various optimization methods',
    author='Alexandr Katrutsa',
    author_email="aleksandr.katrutsa@phystech.edu",
    packages=['liboptpy', 'liboptpy.unconstr_solvers'],
    install_requires=['numpy>=1.12', 'future>=0.15.2', 'scipy>=1.0'],
    keywords=[ 'Convex optimization', 'numerical optimization', 
              'Python', 'Numpy', 'Scipy'],
    url='https://github.com/amkatrutsa/liboptpy',
    license='MIT',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5'],
)
