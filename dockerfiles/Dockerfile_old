#from labdev-nomad.esc.rzg.mpg.de:5000/nomadlab/notebook:v1.8.0-183-g0b5445c4-dirty 
from face-of-crystals:v1.0.0

# copy things needed for the tutorial
# # images
COPY analytics-toolkit-tutorials/example-data/face-of-crystals/ /home/beaker/.beaker/v1/web/
# # nomad_sim code
ADD analysis-tools/structural-similarity/python-modules/nomad_sim /usr/lib/python2.7/nomad_sim
# # nomad_sim visualization code
ADD analysis-tools/structural-similarity/python-modules/nomad_sim_visualization /usr/lib/python2.7/nomad_sim_visualization
# # copy data from Archive needed for the tutorial
ADD parsed/production/VaspRunParser1.2.0-3-g4facbeb /parsed/production/VaspRunParser1.2.0-3-g4facbeb
# # copy data from example-data needed for the tutorial
ADD analytics-toolkit-tutorials/example-data/face-of-crystals  /home/beaker/test/face-of-crystals
# #  copy beaker notebook in tutorial folder
COPY analysis-tools/structural-similarity/tutorials/face_of_crystals.bkr /home/beaker/notebooks/face_of_crystals.bkr

RUN apt-get update --fix-missing && apt-get install -y swig

# remove libspimage, spsim and condor
RUN cd /root/Sources && rm -rf libspimage && rm -rf spsim && rm -rf condor

# Install condor with dependencies
RUN cd /root/Sources && git clone https://github.com/FilipeMaia/libspimage.git && cd libspimage && mkdir build && cd build && cmake .. && make && make install && cd /root/Sources && git clone https://github.com/FilipeMaia/spsim.git && cd spsim && mkdir build && cd build && cmake .. && make && make install && cd /root/Sources && git clone https://github.com/mhantke/condor.git && cd condor && python setup.py install

# remove current version tensorflow and install 0.9.0
RUN pip uninstall -y protobuf
RUN pip uninstall -y tensorflow
ENV TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
RUN pip install --upgrade $TF_BINARY_URL

# remove current Keras and install 1.2.0
RUN rm -rf /usr/local/lib/python2.7/dist-packages/keras/
RUN pip install --upgrade keras==1.2.0

# change defalt keras backend to Theano
RUN mkdir /home/beaker/.keras
COPY keras_tf.json /home/beaker/.keras/keras.json 

# other pip dependencies
RUN pip install pyshtools
RUN pip install pyquaternion
RUN pip install keras-tqdm

