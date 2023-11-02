# syntax=docker/dockerfile:1
# You can choose the version of freesurfer
FROM freesurfer/freesurfer:7.4.1
# the OS installed is a centos 8

COPY ./.config_envs/license_freesurfer.txt /usr/local/freesurfer/.license
COPY ./.config_envs/* /root/Epilepsy-dMRI-VNS/.config_envs/

WORKDIR /root

# install miniconda and configure conda-forge and the solver libmamba (limamba is faster to solve the conflicts)
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/condabin/conda init bash
RUN ~/miniconda3/condabin/conda config --add channels conda-forge
RUN ~/miniconda3/condabin/conda config --set channel_priority strict
RUN ~/miniconda3/condabin/conda update --yes -n base conda
RUN ~/miniconda3/condabin/conda install --yes -n base conda-libmamba-solver
RUN ~/miniconda3/condabin/conda config --set solver libmamba
RUN ~/miniconda3/condabin/conda config --set auto_activate_base False

# install fsl 6.0
RUN ~/miniconda3/bin/python Epilepsy-dMRI-VNS/.config_envs/fslinstaller.py -d ~/fsl/ -n
RUN echo 'FSLDIR=~/fsl/' >> ~/.bashrc
RUN echo 'PATH=${FSLDIR}/share/fsl/bin:${PATH}' >> ~/.bashrc
RUN echo 'export FSLDIR PATH' >> ~/.bashrc
RUN echo '. ${FSLDIR}/etc/fslconf/fsl.sh' >> ~/.bashrc

# Create the enviroment and activate it
RUN ~/miniconda3/condabin/conda env update -p ~/fsl/ --file Epilepsy-dMRI-VNS/.config_envs/environment.yml --prune
RUN source ~/miniconda3/bin/activate fsl

# Install git to import the project
RUN dnf -y install git

# install microstructure_fingerprinting
RUN git clone https://github.com/rensonnetg/microstructure_fingerprinting.git
WORKDIR /root/microstructure_fingerprinting
RUN ~/fsl/bin/python setup.py install
WORKDIR /root

# install elikopy
RUN git clone https://github.com/Hyedryn/elikopy.git
RUN ~/fsl/bin/python -m pip install ~/elikopy/

COPY . /root/Epilepsy-dMRI-VNS

WORKDIR /root/Epilepsy-dMRI-VNS
