from setuptools import setup

setup(
    version='1.0',
    name='r3d2',
    packages=['r3d2'],
    license='MIT',
    description='Relativistic Reactive Riemann problem solver for Deflagrations and Detonations',
    install_requires=['scipy','numpy','matplotlib'],
    author='Alice Harpole, Ian Hawke',
    author_email='A.Harpole@soton.ac.uk, I.Hawke@soton.ac.uk',
    url='https://github.com/harpolea/r3d2',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics',
      ],
    keywords='relativity riemann solver',
)
