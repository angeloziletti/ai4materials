FROM jupyter/datascience-notebook

# extra metadata
LABEL version="0.9.0"
LABEL description="First image docker."

USER root
MAINTAINER Angelo Ziletti

# copy things needed for the tutorial
# # images
#COPY analytics-toolkit-tutorials/example-data/face-of-crystals/ /home/beaker/.beaker/v1/web/

RUN apt-get update --fix-missing \
&& apt-get install -y swig \
&& apt-get install -y cmake 
# tensorflow deps (fftw3) \
#&& apt-get -y install libhdf5-dev && apt-get -y install libfftw3-dev libfftw3-doc \
# Install nfft deps (libtiff) \
#&& apt-get -y install libtiff5 libtiff5-dev \
# clean up packages` \
#&& apt-get clean

RUN mkdir -p /root/Sources

#RUN apt-get update --fix-missing 
#RUN apt-get -y install libhdf5-dev && apt-get -y install libfftw3-dev libfftw3-doc && pip install numpy scipy h5py
#&& apt-get install -y cmake \
#RUN apt-get -y install libtiff5 libtiff5-dev
RUN cd /root/Sources && git clone https://github.com/FilipeMaia/libspimage.git && cd libspimage && mkdir build && cd build && cmake .. && make && make install && cd /root/Sources && git clone https://github.com/FilipeMaia/spsim.git && cd spsim && mkdir build && cd build && cmake .. && make && make install && cd /root/Sources && git clone https://github.com/mhantke/condor.git && cd condor && python setup.py install
#RUN cat /home/beaker/additionalReq/requirements.txt | xargs /home/beaker/py3k/bin/pip install --upgrade


#RUN wget http://www.cmake.org/files/v3.5/cmake-3.5.2.tar.gz && tar xf cmake-3.5.2.tar.gz && cd cmake-3.5.2 && ./configure && make

#RUN mkdir -p /root/Sources && cd /root/Sources \
#&& git clone https://github.com/FilipeMaia/libspimage.git && cd libspimage && mkdir build && cd build && cmake .. && make && make install && cd .. && rm -rf libspimage \
#&& cd /root/Sources && git clone https://github.com/FilipeMaia/spsim.git && cd spsim && mkdir build && cd build && cmake .. && make && make install && cd .. && rm -rf spsim \
#&& cd /root/Sources && git clone https://github.com/mhantke/condor.git && cd condor && python setup.py install && git clean -dxf && git reset --hard && /home/beaker/py3k/bin/python setup.py install && cd .. && rm -rf condor

#RUN mkdir -p /root/Sources && cd /root/Sources && wget https://www-user.tu-chemnitz.de/~potts/nfft/download/nfft-3.2.3.tar.gz && tar xvzf nfft-3.2.3.tar.gz && cd nfft-3.2.3 && ./configure && make && make install && apt-get -y install libtiff5 libtiff5-dev


# Install condor with dependencies
#RUN cd /root/Sources && git clone https://github.com/FilipeMaia/libspimage.git && cd libspimage && mkdir build && cd build && cmake .. && make && make install && cd .. && rm -rf libspimage 
#RUN cd /root/Sources && git clone https://github.com/FilipeMaia/spsim.git && cd spsim && mkdir build && cd build && cmake .. && make && make install && cd .. && rm -rf spsim
#RUN cd /root/Sources && git clone https://github.com/mhantke/condor.git && cd condor && python setup.py install

# remove current version tensorflow and install 0.9.0
#RUN pip uninstall -y protobuf
#RUN pip uninstall -y tensorflow
#ENV TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
#RUN pip install --upgrade $TF_BINARY_URL

# remove current Keras and install 1.2.0
#RUN rm -rf /usr/local/lib/python2.7/dist-packages/keras/
#RUN pip install --upgrade keras==1.2.0

# change default keras backend to Theano
#RUN mkdir /home/beaker/.keras
#COPY keras_tf.json /home/beaker/.keras/keras.json 

# other pip dependencies
#RUN pip install pyshtools
#RUN pip install pyquaternion
#RUN pip install keras-tqdm

