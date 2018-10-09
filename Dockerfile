#FROM jupyter/datascience-notebook
#FROM jupyter/minimal-notebook
FROM python:2.7

# extra metadata
LABEL version="0.9.0"
LABEL description="First image docker."

#USER root
MAINTAINER Angelo Ziletti

RUN apt-get update -qq --fix-missing
RUN apt-get install -y -qq cmake
RUN apt-get install -y -qq libtiff5-dev
RUN apt-get install -y -qq libfftw3-dev
RUN apt-get install -y -qq gsl-bin
RUN apt-get install -y -qq libgsl0-dev
RUN apt-get install -y -qq swig

#ENV PY_SITE=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
#ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/local/lib:${HOME}/local/lib64:${PY27_PREFIX}/lib
ENV PYTHONPATH=${HOME}/localpy:${PYTHONPATH}

# install hdf5
RUN cd $HOME
    #  - if [ ! -d "$HOME/hdf5-1.10.0-patch1-linux-centos7-x86_64-gcc485-shared/include" ]; then wget https://support.hdfgroup.org/ftp/HDF5/current/bin/linux-centos7-x86_64-gcc485/hdf5-1.10.0-patch1-linux-centos7-x86_64-gcc485-shared.tar.gz && tar xvzf hdf5-1.10.0-patch1-linux-centos7-x86_64-gcc485-shared.tar.gz; else echo 'Using hdf5 from cached directory'; fi
    # - export HDF5_DIR=${HOME}/hdf5-1.8.19-linux-centos7-x86_64-gcc485-shared/
    # - if [ ! -d "${HDF5_DIR}/include" ]; then wget https://support.hdfgroup.org/ftp/HDF5/current18/bin/linux-centos7-x86_64-gcc485/hdf5-1.8.19-linux-centos7-x86_64-gcc485-shared.tar.gz && tar xvzf hdf5-1.8.19-linux-centos7-x86_64-gcc485-shared.tar.gz; else echo 'Using hdf5 from cached directory'; fi
ENV HDF5_DIR=${HOME}/hdf5-1.8.20-linux-centos7-x86_64-gcc485-shared/
RUN if [ ! -d "${HDF5_DIR}/include" ]; then wget https://support.hdfgroup.org/ftp/HDF5/current18/bin/hdf5-1.8.20-linux-centos7-x86_64-gcc485-shared.tar.gz && tar xvzf hdf5-1.8.20-linux-centos7-x86_64-gcc485-shared.tar.gz; else echo 'Using hdf5 from cached directory'; fi

    #  - export LD_LIBRARY_PATH=${HOME}/hdf5-1.10.0-patch1-linux-centos7-x86_64-gcc485-shared/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${HDF5_DIR}/lib:${LD_LIBRARY_PATH}
RUN ls ${HOME}/
RUN ls ${HDF5_DIR}
RUN ls ${HDF5_DIR}/include
RUN ls ${HDF5_DIR}/lib

# install numpy and scipy
RUN pip install numpy
RUN pip install scipy

# install configparser with pip (this is to make condor compatible with Python 3, which no longer has the ConfigParser module)
RUN pip install configparser

    # install h5py (we want an h5py that is built with the new hdf5 version, that is why)
    #  - export HDF5_DIR=${HOME}/hdf5-1.10.0-patch1-linux-centos7-x86_64-gcc485-shared/
    #  - export HDF5_DIR=${HOME}/hdf5-1.8.19-linux-centos7-x86_64-gcc485-shared/
ENV HDF5_DIR=${HOME}/hdf5-1.8.20-linux-centos7-x86_64-gcc485-shared/
RUN pip install h5py

    # testing imports
RUN python -c "import numpy; print(numpy.__file__)"
RUN python -c "import scipy; print(scipy.__file__)"
RUN python -c "import h5py; print(h5py.__file__)"

    # install libspimage
RUN cd $HOME \
&& git clone https://github.com/FXIhub/libspimage \
&& mkdir -p libspimage/build && cd libspimage/build \
&& git pull \
&& cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DUSE_CUDA=OFF -DPYTHON_WRAPPERS=ON -DHDF5_INCLUDE_DIR=${HDF5_DIR}/include -DHDF5_LIBRARY=${HDF5_DIR}/lib/libhdf5.so -DCMAKE_INSTALL_PREFIX=${HOME}/virtualenv/python${TRAVIS_PYTHON_VERSION} -DPYTHON_INSTDIR=${PY_SITE} .. \
&& make \
    #; -j 2 VERBOSE=1
&& make install

# install NFFT
RUN cd $HOME
RUN if [ ! -d "$HOME/nfft-3.2.3/include" ]; then wget https://www-user.tu-chemnitz.de/~potts/nfft/download/nfft-3.2.3.tar.gz && tar xvzf nfft-3.2.3.tar.gz; cd nfft-3.2.3 && ./configure --prefix=${HOME}/local && make && make install; else echo 'Using NFFT from cached directory'; fi

    # install spsim
RUN cd $HOME
RUN git clone https://github.com/FXIhub/spsim
RUN mkdir -p spsim/build && cd spsim/build
#RUN git pull
RUN cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DBUILD_LIBRARY=ON -DUSE_CUDA=OFF -DUSE_NFFT=OFF -DPYTHON_WRAPPERS=ON -DHDF5_INCLUDE_DIR=${HDF5_DIR}/include -DHDF5_LIBRARY=${HDF5_DIR}/lib/libhdf5.so -DCMAKE_INSTALL_PREFIX=${HOME}/virtualenv/python${TRAVIS_PYTHON_VERSION} -DPYTHON_INSTDIR=${PY_SITE}  ..
RUN make VERBOSE=1
RUN make install
RUN ls -alh $HOME/local/lib

    # install condor
RUN cd $HOME
RUN git clone https://github.com/FXIhub/condor.git
RUN cd condor
RUN python setup.py install --nfft-include-dir=$HOME/local/include --nfft-library-dir=$HOME/local/lib

    # --------- end condor and condor installation dependencies --------------------#

#    - cd $CI_PROJECT_DIR
#    - pip install -e .
#    - mkdir $HOME/dependencies
#    - cd $HOME/dependencies
#    - git clone https://gitlab.mpcdf.mpg.de/nomad-lab/python-common.git
#    - cd python-common
#    - pip install -r requirements.txt
#    - pip install -e .
#    - cd ../
#    #- git clone https://gitlab.mpcdf.mpg.de/nomad-lab/atomic-data.git
#    #- cd atomic-data
#    #- pip install -e .
#    #- cd $HOME
#
#  script:
#    # manually install ai4materials deps
#    #- pip install numpy scipy matplotlib flask psycopg2-binary
#    # extra packages for testing
#    # using "install from source" instructions
#    #- export PATH=$PATH:$CI_PROJECT_DIR/bin
#    #- echo $PATH
#    #- export PYTHONPATH=$CI_PROJECT_DIR
#    #- echo $PYTHONPATH
#    # tests
#    #- pip install -e .
#    - python --version
#    - pip freeze
#    - pip install pyflakes
#    - cd $CI_PROJECT_DIR
#    - nosetests -v --nocapture --logging-level=INFO --cover-package=ai4materials --with-coverage
#    - pyflakes ai4materials


#RUN apt-get update --fix-missing \
#&& apt-get install -y libpng-dev \
##&& apt-get install -y libtiff5-dev \
#&& apt-get install -y libfftw3-dev \
#&& apt-get install -y libhdf5-serial-dev \
#&& apt-get install -y cmake \
#&& apt-get install -y gsl-bin \
#&& apt-get install -y libgsl0-dev \
#&& apt-get install -y swig
#
#
#RUN export PY_SITE=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
#
#ENV PATH="/opt/conda/bin:${PATH}"
#ENV PYTHONPATH="/opt/conda/:${PYTHONPATH}"
#
#ENV HDF5_LIBRARY="/opt/conda/lib/libhdf5.so"
#ENV HDF5_INCLUDE_DIR="/opt/conda/include"
#
#
#RUN pip install numpy
#RUN pip install scipy
#RUN pip install h5py
#
##RUN mkdir -p /root/Sources
##RUN cd /root/Sources
##RUN git clone https://github.com/FilipeMaia/libspimage.git && cd libspimage
##RUN mkdir build && cd build && cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DUSE_CUDA=OFF -DPYTHON_WRAPPERS=ON ..
#
#
#RUN mkdir -p /root/Sources && cd /root/Sources \
#&& git clone https://github.com/FilipeMaia/libspimage.git && cd libspimage && mkdir build && cd build \
#&& cmake -DHDF5_LIBRARY=HDF5_LIBRARY -DHDF5_INCLUDE_DIR=HDF5_INCLUDE_DIR .. \
#&& make && make install && cd .. && rm -rf libspimage \
