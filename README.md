[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


ARISE: Crystal-structure recognition via Bayesian deep learning
========================================================

![](./assets/ARISE_logo.svg)


This branch provides code for reproducing the results of 

    A. Leitherer, A. Ziletti, and L.M. Ghiringhelli,
    Robust recognition and exploratory analysis of crystal structures via Bayesian deep learning, arXiv:2103.09777 (2021)


You can proceed with the installation steps as described below or directly proceed to a tutorial available at

    http://analytics-toolkit.nomad-coe.eu/tutorial-ARISE
    
within the NOMAD analytics toolkit (https://nomad-lab.eu/AItutorials) where you do not have to install any software.

The code of this branch uses functionalities of ai4materials (https://github.com/angeloziletti/ai4materials) that is currently under development.

ai4materials allows to perform complex analysis of materials science data, using machine learning techniques. It also
provide functions to pre-process (on parallel processors), save and subsequently load materials science datasets,
thus easing the traceability, reproducibility, and prototyping of new models.

Code author: Angelo Ziletti, Ph.D. (angelo.ziletti@gmail.com; ziletti@fhi-berlin.mpg.de), Andreas Leitherer, Ph.D. student (andreas.leitherer@gmail.com, leitherer@fhi-berlin.mpg.de)



------------------
Installation
------------------

We recommend to create a virtual python 3.7 environment (for instance, with conda), and then execute

    git clone https://github.com/angeloziletti/ai4materials.git 
    cd ai4materials
    git checkout ARISE
    pip install -e .

To reproduce the results in arXiv:2103.09777, you need to install the quippy package  (https://github.com/libAtoms/QUIP) 
to be able to compute the SOAP descriptor.

---------------
ARISE - Usage
---------------

For global or local analysis of single- or polycrystalline systems, one just needs to define the corresponding geometry file and load a pretrained model for prediction:

    from ai4materials.models import ARISE

    geometry_files = [ file_1, file_2, ... ]

    predictions, uncertainty = ARISE.analyze(geometry_files, mode='global') 

    predictions, uncertainty = ARISE.analyze(geometry_files, mode='local',
                                              stride=[[4.0, 4.0, 4.0], ...], box_size=[12.0, ...])
                                              
Please refer to  http://analytics-toolkit.nomad-coe.eu/tutorial-ARISE and the associated publication for more details.
