# An Auto-Encoder Strategy for Adaptive Image Segmentation
An Auto-Encoder Strategy for Adaptive Image Segmentation

## Abstract:
Deep neural networks are powerful tools for biomedical image segmentation. These models are often trained with heavy supervision, relying on pairs of images and corresponding voxel-level labels. However, obtaining segmentations of anatomical regions on a large number of cases can be prohibitively expensive. Furthermore, models trained with heavy supervision are often sensitive to shifts in image characteristics, for instance, due to a routine upgrade in scanner software. Thus there is a strong need for deep learning-based segmentation tools that do not require heavy supervision and can continuously adapt.  In this paper, we propose a novel perspective of segmentation as a discrete representation learning problem, and present a variational autoencoder segmentation strategy that is flexible and adaptive. Our method, called Segmentation Auto-Encoder (SAE), leverages all available unlabeled scans and merely requires a segmentation prior, which can be a single unpaired segmentation image. In experiments, we apply SAE to brain MRI scans. Our results show that SAE can produce good quality segmentations, particularly when the prior is good. We demonstrate that a Markov Random Field prior can yield significantly better results than a spatially independent prior. 

## Requirements:


