# Prediction of response to VNS in DRE
The aim of this thesis is the understanding of which microstructual features differ between responders and non-responders to **Vagus Nerve Stimulation** (VNS) in **Drug-Resistant Epileptic** (DRE) patients.

The dataset we have is made up of 19 patients, for each we have **T1**, **T2** and **DWI** volumes of their brain *after the implantation* of the VNS.
# Setup environment
## Setup environment on your own computer
This thesis was composed by different tools and softwares, which many are listed in ``.config_envs/enviroment.yml`` or ``.config_envs/requirements.txt``, while others are specified in ``Dockerfile``.

A non esaustive list is reported here:
- FreeSurfer
  - TRACULA
- Elikopy
  - FSL
  - Microstructure Fingerprinting
- ANTs
- MRtrix3
- Radiomics
- PyTorch

To setup all the enviroment without going mad a Docker images was created with the ready environment. 

You can download the image from Docker Hub through:
```
docker pull micerr\Epilepsy-DMRI-VNS:1.0.0
```

The same image can be buit by using the ``Dockerfile`` it takes ~1hour.
Inside the ``Dockerfile`` is possible to change the version of FreeSurfer or update the version of Elikopy.

Then create and run a container:

```
docker run -i -v /absolute/path/to/Epilepsy-dMRI-VNS/:/root/Epilepsy-dMRI-VNS/ -t test/latest
```

As soon as it ends, you have to activate the enviroment with:
```
conda activate dMRI
```

If everything works the ``~\Epilepsy-dMRI-VNS`` folder in the container will be the same as your folder.

## Setup environment on CECI
Many of the software and libraries are already installed on CECI cluster.
- Install Elikopy on CECI: [elikopy docs](https://elikopy.readthedocs.io/en/latest/installation.html#using-elikopy-on-the-ceci-cluster)

During the installation of Elikopy on the clusters in ``/CECI/proj/pilab/Software/config_elikopy.bash`` are loaded different modules:
- MRtrix3
- MisterI
- ANTs
- DIAMOND
- FSL
- FreeSurfer
- C3D
- Microstructure Fingerprinting
- Elikopy

The other software can be installed following the CECI docs on pre-installed softwares: [CECI docs](https://support.ceci-hpc.be/doc/_contents/UsingSoftwareAndLibraries/UsingPreInstalledSoftware/index.html).

To intall a software that is not in the pre-installed software follow always the [CECI docs](https://support.ceci-hpc.be/doc/_contents/UsingSoftwareAndLibraries/InstallingSoftwareByYourself/index.html).

# dMRI preprocessing with ElikoPy
The preprocessing of diffusion images is done by using **ElikoPy**. The full **documentation** is [here](https://elikopy.readthedocs.io/en/latest/), while the **repository** is [here](https://github.com/Hyedryn/elikopy). It's maintained by the amazing guys of [PiLAB-Medical-Imaging](https://github.com/PiLAB-Medical-Imaging) of [UCLouvain](https://uclouvain.be/en/index.html).

I suggest to **read all the [documentation](https://elikopy.readthedocs.io/en/latest/)** before use my python script, and take it as example. The parameters used in the preprocessing and metric models highly depend on the study that you are doing. Many new parameters and changes are done every month by the team, so take it with grain of salt.

Read the script in ``src/0-dMRI-proproc/preproc.py``, it is full of comments to drive you in the understanding.

``preproc.py`` takes the parameter ``-f`` that is the relative path to the study folder (how the study folder must be composed is explained in Elikopy [docs](https://elikopy.readthedocs.io/en/latest/elikopy_project.html) ).
It takes also the parameter ``-CECI`` if it runs of CECI cluster.

In this step must be present at least the folder ``study/data_1/`` where are stored:
- ``acqparams.txt``
- ``index.txt``
- ``subj0.bval``
- ``subj0.bvec``
- ``subj0.json``
- ``subj0.nii.gz``
- ``subj1.bval``
- ...
- ``subjN.nii.gz``

If everything is setted you can run the python script
```
python src/0-dMRIpreproc/preproc.py -f ./study/
```
# Brain segmenation with FreeSurfer

Brain segmentation is used to extract Region of Interest (ROI) from **T1 images**.

Two scripts are present ``sub_seg.py`` for CECI users, to segment the brain on the cluster, ``seg.sh`` to run the segmentation on you machine.

The two scripts run automatically in parallel the segmentation to speedup the process. Each brain segmentation can take **from 8 to 10 hours**, therefore parallelizing is necessary.

In ``seg.sh`` can be set the number of jobs in parallel. I always suggest ``njobs=nsubjects`` to reduce the time to a maximum of 10 hours.

The T1 volumes should be located in ``study/T1/``. Furthermore, to increase the accuracy of the segmentation you can use the T2 volumes during the computation with ``-T2`` and ``-T2pial`` parametes.
- Documentation of ``recon-all`` command [here](https://freesurfer.net/fswiki/recon-all)
- Very good tutorial on FreeSurfer segmenation [here](https://andysbrainbook.readthedocs.io/en/latest/FreeSurfer/FreeSurfer_Introduction.html)

A further more precise segmentation of the Thalamus can be done by using ``segmentThalamicNuclei.sh`` explained in the [FreeSurfer docs](https://freesurfer.net/fswiki/ThalamicNuclei). The team of FreeSurfer is working on it and soon this tool will be implemented directly in FreeSurfer.

Read the comments on ``src/1-brainSegFreeSurfer/sub-seg`` for more hints.