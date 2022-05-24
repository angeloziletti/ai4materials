from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
# Other stuff
import sys, os, fileinput
import versioneer

here = os.path.dirname(os.path.realpath(__file__))


def main():
    # Start package setup

    # Get the long description from the README file
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    setup(

        # template at https://github.com/pypa/sampleproject/blob/master/setup.py
        name='ai4materials',

        # Versions should comply with PEP440.  For a discussion on single-sourcing
        # the version across setup.py and the project code, see
        # https://packaging.python.org/en/latest/single_source_version.html
        # version=get_property('__version__'),
        version="0.1",
        description='Data-analytics modeling of materials science data', long_description=long_description,

        zip_safe=True,

        # The project's main homepage.
        url='https://https://github.com/angeloziletti/ai4materials',

        # Author details
        author='Ziletti, Angelo and Leitherer, Andreas', author_email='angelo.ziletti@gmail.com, andreas.leitherer@gmail.com',

        # Choose your license
        license='Apache License 2.0',

        # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        classifiers=[# How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Physics :: Materials science :: Machine learning :: Deep learning :: Data analytics',

            # Pick your license as you wish (should match "license" above)
            'License :: Apache Licence 2.0',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            # 'Programming Language :: Python :: 2.7',
            # 'Programming Language :: Python :: 3.5',
            # 'Programming Language :: Python :: 3.6',
            # 'Programming Language :: Python :: 3.7'
        ],

        # What does your project relate to?
        keywords='Data analytics of materials science data.',

        # You can just specify the packages manually here if your project is
        # simple. Or you can use find_packages().
        packages=['ai4materials', 'ai4materials.dataprocessing', 'ai4materials.descriptors',
                  'ai4materials.interpretation', 'ai4materials.visualization',
                   'ai4materials.models', 'ai4materials.utils', 'ai4materials.external'],
        #packages=find_packages(include=['ai4materials']),

        package_dir={'ai4materials': 'ai4materials'},

        # Alternatively, if you want to distribute just a my_module.py, uncomment
        # this:
        #   py_modules=["my_module"],

        # List run-time dependencies here.  These will be installed by pip when
        # your project is installed. For an analysis of "install_requires" vs pip's
        # requirements files see:
        # https://packaging.python.org/en/latest/requirements.html
        install_requires=['ase==3.19.0', 'tensorflow==2.6.4', 'keras==2.2.4', 'scikit-learn>=0.17.1', 'pint', 'future',
                          'pandas<=0.25.0', 'enum34', 'pymatgen==2020.3.13', 'keras-tqdm', 'seaborn', 'paramiko',
                          'scipy', 'nose>=1.0', 'numpy', 'h5py<=2.9.0', 'cython>=0.19',  'Jinja2'],
        #         
        #'ase==3.15.0',  # neighbors list does not work for ase 3.16
        #    'scikit-learn >=0.17.1', 'tensorflow==1.8.0', 'pint', 'future', 'pandas',
        #                  'bokeh',
        #    'enum34', 'pymatgen', 'keras==1.2.0', 'pillow>=2.7.0', 'mendeleev', 'keras-tqdm', 
        #                  'seaborn', 'paramiko', 'scipy', 'nose>=1.0', 'sqlalchemy', 'theano==0.9.0',
        #    'numpy', 'h5py', 'cython>=0.19', 'pyshtools', 'Jinja2'],

        # 'bokeh==0.11.0',

        # 'multiprocessing',

        # , 'asap3'],
                          #'mayavi', 'weave'],

        #setup_requires=['nomadcore', 'atomic_data'],
        # List additional groups of dependencies here (e.g. development
        # dependencies). You can install these using the following syntax,
        # for example:
        # $ pip install -e .[dev,test]
        extras_require={
        #    'dev': ['check-manifest'],
           'test': ['pytest', 'coverage'],
        },
        # https://mike.zwobble.org/2013/05/adding-git-or-hg-or-svn-dependencies-in-setup-py/
        # add atomic_data and nomadcore
        dependency_links=['https://github.com/libAtoms/QUIP',
                          'https://github.com/FXIhub/condor.git'],

        # If there are data files included in your packages that need to be
        # installed, specify them here.  If using Python 2.6 or less, then these
        # have to be included in MANIFEST.in as well.
        package_data={
            'ai4materials': ['descriptors/descriptors.nomadmetainfo.json', 
                        'data/nn_models/*.h5', 'data/nn_models/*.json', 'utils/units.txt', 'utils/constants.txt',
                        'data/PROTOTYPES/*/*/*.in', 'data/training_data/*.pkl', 'data/training_data/*.json'
                        ]},

        # Although 'package_data' is the preferred approach, in some case you may
        # need to place data files outside of your packages. See:
        # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
        # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
        # data_files=[('my_data', ['data/data_file'])],

        # To provide executable scripts, use entry points in preference to the
        # "scripts" keyword. Entry points provide cross-platform support and allow
        # pip to create the appropriate form of executable for the target platform.
        # entry_points={
        #     'console_scripts': [
        #         'condor=condor.scripts.condor_script:main',
        #     ],
        # },

        # test_suite = "condor.tests.test_all",

        project_urls={  # Optional
            'Bug Reports': 'https://gitlab.com/ai4materials/issues', 'Source': 'https://gitlab.com/ai4materials/', },

    )


# Run main function by default
if __name__ == "__main__":
    main()
