# syntax=docker/dockerfile:1
# This building dockerfile takes more than 1 hour to build
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
RUN ~/miniconda3/condabin/conda create -n dMRI pip packaging python=3.10
RUN source ~/miniconda3/bin/activate dMRI

# install fsl 6.0
RUN ~/miniconda3/envs/dMRI/bin/python Epilepsy-dMRI-VNS/.config_envs/fslinstaller.py -d ~/fsl/ -n
RUN echo 'FSLDIR=~/fsl/' >> ~/.bashrc
RUN echo 'PATH=${FSLDIR}/share/fsl/bin:${PATH}' >> ~/.bashrc
RUN echo 'export FSLDIR PATH' >> ~/.bashrc
RUN echo '. ${FSLDIR}/etc/fslconf/fsl.sh' >> ~/.bashrc

# Install git to import the project
RUN dnf -y install git

# install elikopy
RUN git clone https://github.com/Hyedryn/elikopy.git
RUN ~/miniconda3/envs/dMRI/bin/python -m pip install ~/elikopy/

# install microstructure_fingerprinting
RUN git clone https://github.com/rensonnetg/microstructure_fingerprinting.git
RUN ~/miniconda3/envs/dMRI/bin/python -m pip install ~/microstructure_fingerprinting

# Update the enviroment and activate it
# RUN ~/miniconda3/condabin/conda update -f Epilepsy-dMRI-VNS/.config_envs/environment.yml --prune
RUN ~/miniconda3/envs/dMRI/bin/pip install -r Epilepsy-dMRI-VNS/.config_envs/requirements.txt -U
RUN ~/miniconda3/condabin/conda install --yes -c mrtrix3 mrtrix3

WORKDIR /root/Epilepsy-dMRI-VNS
