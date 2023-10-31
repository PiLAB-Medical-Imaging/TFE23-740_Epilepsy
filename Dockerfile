# syntax=docker/dockerfile:1
# You can choose the version of freesurfer
FROM freesurfer/freesurfer:7.4.1
# the OS installed is a centos 8

COPY license_freesurfer.txt /usr/local/freesurfer/.license

WORKDIR /root

# install python3.10
# RUN dnf update
RUN dnf -y install wget yum-utils make gcc openssl-devel bzip2-devel libffi-devel zlib-devel
## here you can choose your favourite version of python
RUN wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz 
RUN tar xzf Python-3.10.8.tgz
WORKDIR /root/Python-3.10.8 
RUN ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions
RUN make -j ${nproc}
RUN make altinstall
WORKDIR /root
RUN rm Python-3.10.8.tgz

# install miniconda and configure conda-forge and the solver libmamba (limambe is faster to solve the conflicts)
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
RUN ~/miniconda3/bin/conda config --add channels conda-forge
RUN ~/miniconda3/bin/conda config --set channel_priority strict
RUN ~/miniconda3/bin/conda update --yes -n base conda
RUN ~/miniconda3/bin/conda install --yes -n base conda-libmamba-solver
RUN ~/miniconda3/bin/conda config --set solver libmamba
RUN ~/miniconda3/bin/conda config --set auto_activate_base False

# Install git to import the project
RUN dnf -y install git
RUN git clone https://github.com/micerr/Epilepsy-dMRI-VNS.git

# Create the enviroment and activate it
RUN ~/miniconda3/bin/conda env create -f Epilepsy-dMRI-VNS/.config_envs/environment.yml
RUN source ~/miniconda3/bin/activate dMRI

# install fsl
# RUN python Epilepsy-dMRI-VNS/.config_envs/fslinstaller.py
# 
# # install microstructure_fingerprinting
# RUN git clone https://github.com/rensonnetg/microstructure_fingerprinting.git
# WORKDIR /root/microstructure_fingerprinting
# RUN python setup.py install -U
# WORKDIR /root
# 
# # install elikopy
# RUN git clone https://github.com/Hyedryn/elikopy.git
# RUN python elikopy/setup.py install -U
# 