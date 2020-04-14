# An Auto-Encoder Strategy for Adaptive Image Segmentation

## Abstract
Deep neural networks are powerful tools for biomedical image segmentation. These models are often trained with heavy supervision, relying on pairs of images and corresponding voxel-level labels. However, obtaining segmentations of anatomical regions on a large number of cases can be prohibitively expensive. Furthermore, models trained with heavy supervision are often sensitive to shifts in image characteristics, for instance, due to a routine upgrade in scanner software. Thus there is a strong need for deep learning-based segmentation tools that do not require heavy supervision and can continuously adapt.  In this paper, we propose a novel perspective of segmentation as a discrete representation learning problem, and present a variational autoencoder segmentation strategy that is flexible and adaptive. Our method, called Segmentation Auto-Encoder (SAE), leverages all available unlabeled scans and merely requires a segmentation prior, which can be a single unpaired segmentation image. In experiments, we apply SAE to brain MRI scans. Our results show that SAE can produce good quality segmentations, particularly when the prior is good. We demonstrate that a Markov Random Field prior can yield significantly better results than a spatially independent prior. 

## Requirements
The code was tested on:
- python 3.6
- pytorch 1.1
- torchvision 0.3.0
- scikit-image 0.15.0
- scikit-learn 0.19.1
- matplotlib 3.0.2
- numpy 1.15.4
- tqdm 4.38.0

## Instruction

### Prepocessing 
T1-weighted 3D brain MRI scans was preprocessed using FreeSurfer [1]. This includes skull stripping, bias-field correction, intensity normalization, affine registration to Talairach space, and resampling to 1 mm3 isotropic resolution. We provide some example volumes in `./data/vols/` and automatic segmentations in `./data/labels/`. These images are derived from the OASIS dataset [2]. We also provide the probabilistic atlas used in our experiment in the data folder. 

### Training
Once your MRI has been preprocessed, the next step is to obtain a good initialization for the encoder. This is accomplish by first mapping your training brain MRI to the probabilistic atlas. As an example, we provide our initialization in `./weights/pretrained_encoder.pth.tar`. However, we recommend pretraining your own encoder for your dataset in order to obtain the best results.

To train SAE, run `python train.py 0 2 --compute_dice`

One important parameters in the script is `args.sigma`. Setting `args.sigma = 2` allows you to estimate the variance σ
in 1/(2σ<sup>2</sup>) ||x-x<sup>'</sup>|| as described in the paper. Setting `args.sigma = 0` allows you to set a fixed weight 
α to weight the reconstruction term α||x-x<sup>'</sup>||

The parameter  `args.beta` puts weight on the L<sub>mrf</sub>

## References
[1] Bruce Fischl. Freesurfer. Neuroimage, 62(2):774–781, 2012

[2] Marcus et al. Open access series of imaging studies (OASIS): cross-sectional MRI data
in young, middle aged, nondemented, and demented older adults. Journal of Cognitive
Neuroscience, 2007.
