N_CPUS = 8

import os

os.environ["OMP_NUM_THREADS"] = f"{N_CPUS}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{N_CPUS}"
os.environ["MKL_NUM_THREADS"] = f"{N_CPUS}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{N_CPUS}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{N_CPUS}"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"

import torch

torch.set_num_threads(N_CPUS)  # For intra-op parallelism
torch.set_num_interop_threads(N_CPUS)  # For inter-op parallelism

import torchio as tio
import pandas as pd
import lightning.pytorch as L
import torch
import numpy as np
import monai

from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner


class LitBasicModel(L.LightningModule):
    def __init__(self, model, learning_rate, scheduler_step_size, scheduler_gamma):
        super().__init__()
        self.example_input_array = torch.Tensor(8, 1, 110, 110, 68)

        self.learning_rate = learning_rate
        self.model = model
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.batch_size = None

        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.sigmoid = torch.nn.Sigmoid()

        # TODO rimuovi validate_args per speeduppare
        self.acc = torch.nn.ModuleList([BinaryAccuracy(threshold=0.5, validate_args=True) for _ in range(3)])
        self.auc_roc = torch.nn.ModuleList([BinaryAUROC(validate_args=True) for _ in range(3)])
        # Crea la balanced accuracy 

        self.association = {
            "train": 0,
            "val": 1,
            "test": 2
        }

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        outputs = self.model(x)
        return outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    # def __predict(self, probabilities):
    #     return torch.where(probabilities >= 0.5, 1, 0)
    
    def __step(self, batch, batch_idx, name):
        inputs = batch[0]
        targets = batch[1]
        outputs = self.model(inputs)
        probabilities = self.sigmoid(outputs)
        losses = self.loss_fn(probabilities, targets)
        avg_loss = losses.mean()
        self.acc[self.association[name]](probabilities, targets)
        self.auc_roc[self.association[name]](probabilities, targets)
        self.log_dict({
            f"{name}_loss": avg_loss,
            f"{name}_acc" : self.acc[self.association[name]],
            f"{name}_auc-roc": self.auc_roc[self.association[name]],
            }, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.batch_size,
            # sync_dist=True # Remove if are not using ddp
        )
        return avg_loss
    
    def training_step(self, train_batch, batch_idx):
        return self.__step(train_batch, batch_idx, "train")
    
    def validation_step(self, val_batch, batch_idx):
        return self.__step(val_batch, batch_idx, "val")
    
    def test_step(self, test_batch, batch_idx):
        return self.__step(test_batch, batch_idx, "test")
    
    def predict_step(self, batch):
        inputs = batch[0]
        outputs = batch[1]
        probabilities = self.sigmoid(outputs)
        return probabilities
    
class DiffusionMRIDataModule(L.LightningDataModule):
    def __init__(self, val_subjs_idx, test_subjs_idx, study_path, batch_size, num_workers, multiplier):
        super().__init__()

        self.val_subjs_idx = val_subjs_idx
        self.test_subjs_idx = test_subjs_idx

        self.study_path = study_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = multiplier

        self.save_hyperparameters()

    @staticmethod
    def __findSizeCrop(subjects_list, image_to_use):
        transformation = tio.Compose([
            tio.transforms.ToCanonical(),
            tio.transforms.Resample(image_to_use),
            tio.transforms.CropOrPad(mask_name="aparc_aseg"),
            tio.transforms.CopyAffine(image_to_use),
        ])
        max = [0, 0, 0]
        for subj in subjects_list:
            subj_transformed = transformation(subj)
            for i in range(3):
                if subj_transformed.shape[i+1] > max[i]:
                    max[i] = subj_transformed.shape[i+1]

        return max

    @staticmethod
    def __load_subjects(study_path):
        info_df = pd.read_csv(f"{study_path}/stats/info.csv").dropna()

        subjects_list = []

        for _, row in info_df.iterrows():
            if "VNSLC" not in row.ID:
                continue
            
            subj_dict = {
                "id": row.ID,
                "resp": row.resp, # Prova cambiando in 0, 0.75, 1
                # "age": row.age,
                # "sex": row.sex,
                # "epilepsy_type": row.epilepsy_type,
                # "epilepsy_onset_age": row.epilepsy_onset_age,
                # "therapy_duration": row.therapy_duration,
                # "AEDs": row.AEDs,
                # "FA": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/dti/{row.ID}_FA.nii.gz"),
                # "MD": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/dti/{row.ID}_MD.nii.gz"),
                # "AD": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/dti/{row.ID}_AD.nii.gz"),
                # "RD": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/dti/{row.ID}_RD.nii.gz"),
                # "wFA": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/diamond/{row.ID}_diamond_wFA.nii.gz"),
                # "wMD": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/diamond/{row.ID}_diamond_wMD.nii.gz"),
                # "wAD": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/diamond/{row.ID}_diamond_wAD.nii.gz"),
                # "wRD": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/diamond/{row.ID}_diamond_wRD.nii.gz"),
                # "diamond_frac_csf": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/diamond/{row.ID}_diamond_frac_csf.nii.gz"),
                # "icvf": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/noddi/{row.ID}_noddi_icvf.nii.gz"),
                # "odi": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/noddi/{row.ID}_noddi_odi.nii.gz"),
                # "fextra": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/noddi/{row.ID}_noddi_fextra.nii.gz"),
                # "fiso": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/noddi/{row.ID}_noddi_fiso.nii.gz"),
                "wfvf": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/mf/{row.ID}_mf_wfvf.nii.gz"),
                # "fvf_tot": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/mf/{row.ID}_mf_fvf_tot.nii.gz"),
                # "mf_frac_csf": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/dMRI/microstructure/mf/{row.ID}_mf_frac_csf.nii.gz"),
                # "WM_mask": tio.LabelMap(f"{study_path}/freesurfer/{row.ID}/dlabel/diff/White-Matter++.bbr.nii.gz"),
                "aparc_aseg": tio.LabelMap(f"{study_path}/freesurfer/{row.ID}/dlabel/diff/aparc+aseg+thalnuc.bbr.nii.gz"),
                # "tract_prob": tio.ScalarImage(f"{study_path}/freesurfer/{row.ID}/dpath/mergedX2_3D_avg16_syn_bbr.nii.gz"),
                # "t1": tio.ScalarImage(f"{study_path}/subjects/{row.ID}/registration/{row.ID}_T1_brain_reg.nii.gz"),
            }

            subjects_list.append(tio.Subject(subj_dict))

        return subjects_list

    @staticmethod
    def getPreprocessingTransform(reference_image, maxSize):
        return tio.Compose([
            ## Preprocessing ##
            # Spatial
            tio.transforms.ToCanonical(),
            tio.transforms.Resample(reference_image),
            tio.transforms.CropOrPad((maxSize[0], maxSize[1], maxSize[2])),
            tio.transforms.CopyAffine(reference_image),
            tio.transforms.EnsureShapeMultiple(8, method="crop"),
            # Voxel Intensity
            tio.transforms.Mask(masking_method="aparc_aseg"),
            # tio.transforms.RescaleIntensity(percentiles=(0.5, 99.5), masking_method="aparc_aseg"),
            tio.ZNormalization(masking_method="aparc_aseg"),
        ]) 
    
    @staticmethod
    def getTrainingTransform():
        return tio.Compose([
            tio.OneOf({
                tio.RandomAffine(scales=0.25, degrees=20, translation=5, check_shape=True): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }),
            tio.RandomAnisotropy(p=0.1),
            tio.RandomBiasField(p=0.1),
            tio.RandomBlur(p=0.1),
            tio.RandomGamma(p=0.1),
            tio.RandomNoise(p=0.2),
            tio.OneOf((
                tio.RandomMotion(),
                tio.RandomGhosting(),
                tio.RandomSpike(),
            ), p=0.1),
            tio.RandomSwap(patch_size=5, p=0.1),
        ])
    
    @staticmethod
    def getValidationTransform():
        return tio.Compose([
            tio.OneOf({
                tio.RandomAffine(scales=0.25, degrees=20, translation=5, check_shape=True): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }),
            tio.RandomAnisotropy(p=0.1),
            tio.RandomBiasField(p=0.1),
            tio.RandomBlur(p=0.1),
            tio.RandomGamma(p=0.1),
            tio.RandomNoise(p=0.2),
            tio.OneOf((
                tio.RandomMotion(),
                tio.RandomGhosting(),
                tio.RandomSpike(),
            ), p=0.1),
        ])
    
    @staticmethod
    def getTestingTransform():
        return tio.Compose([
            tio.RandomAffine(scales=0.05, degrees=5, check_shape=True),
            tio.RandomAnisotropy(downsampling=(1, 2), p=0.1),
            tio.RandomBiasField(coefficients=0.25, p=0.1),
            tio.RandomBlur(std=(0, 2), p=0.2),
            tio.RandomGamma(log_gamma=(-1.5, 1.5), p=0.2),
            tio.RandomNoise(std=(0, 0.1), p=0.2),
        ])

    def setup(self, stage:str) -> None:
        subjects_list = self.__load_subjects(self.study_path)

        test = self.test_subjs_idx
        val = self.val_subjs_idx
        train = set(range(19))
        for el_test in test+val:
            train.remove(el_test)
        train = list(train)

        maxSize = self.__findSizeCrop(subjects_list, "wfvf")
        preprocessing_transform = self.getPreprocessingTransform("wfvf", maxSize)
        k = self.k

        if stage == "fit" or stage == "validate":
            if stage == "fit":
                training_subjects = []
                for idx in train:
                    training_subjects.append(subjects_list[idx])
                training_transform = self.getTrainingTransform()
                self.training_set =   tio.SubjectsDataset(training_subjects*k,   transform=tio.Compose([preprocessing_transform, training_transform]))

            validation_subjects = []
            for idx in val:
                validation_subjects.append(subjects_list[idx])
            validation_transform = self.getValidationTransform()
            self.validation_set = tio.SubjectsDataset(validation_subjects*k, transform=tio.Compose([preprocessing_transform, validation_transform]))

        if stage == "test":
            testing_subjects = []
            for idx in test:
                testing_subjects.append(subjects_list[idx])
            testing_transform = self.getTestingTransform()
            self.testing_set =    tio.SubjectsDataset(testing_subjects*k,    transform=tio.Compose([preprocessing_transform, testing_transform]))

    def train_dataloader(self):
        return DataLoader(
            self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.validation_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.testing_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict):
            inputs = batch["wfvf"][tio.DATA].to(device)
            targets = batch["resp"][np.newaxis].T.to(torch.float).to(device)
            return (inputs, targets)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return batch
    
def basicModel():
    num_epochs = 100
    batch_size = 4
    num_workers = N_CPUS
    multiplier = 100
    learning_rate= 1e-3
    scheduler_gamma = 0.1
    scheduler_step_size = 30

    test_subjs_idx = [2, 16, 13, 12]
    val_subjs_idx = [0, 7, 10, 14]

    dMRI = DiffusionMRIDataModule(val_subjs_idx, test_subjs_idx, "../../study", batch_size, num_workers, multiplier)

    model = monai.networks.nets.ResNet(
        block="bottleneck",
        layers=[3, 8, 36, 3],
        block_inplanes=[64, 128, 256, 512],
        n_input_channels=1,
        num_classes=1,
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        save_weights_only=False,
    )

    basicModel = LitBasicModel(model, learning_rate, scheduler_step_size, scheduler_gamma)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=num_epochs,
        default_root_dir="./",
        # num_sanity_val_steps=-1,
        # profiler="simple",
        callbacks=[checkpoint_callback],
    )
    tuner = Tuner(trainer)
    tuner.lr_find(
        model=basicModel,
        datamodule=dMRI,
    )
    tuner.scale_batch_size(
        model=basicModel,
        datamodule=dMRI,
        steps_per_trial=10,
        max_trials=40
    )

    trainer.fit(
        model=basicModel,
        datamodule=dMRI,
    )

def main():
    return basicModel()

if __name__ == "__main__":
    exit(main())
