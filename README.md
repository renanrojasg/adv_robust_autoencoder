# Inverting Adversarially Robust Networks for Image Synthesis
### [Renan A. Rojas-Gomez](http://renanar2.web.illinois.edu/)<sup>1</sup>, [Raymond A. Yeh](https://www.raymond-yeh.com/)<sup>2</sup>, [Minh N. Do](https://minhdo.ece.illinois.edu/)<sup>1</sup>, [Anh Nguyen](https://anhnguyen.me/research/)<sup>3</sup><br>
University of Illinois at Urbana-Champaign<sup>1</sup>, Purdue University<sup>2</sup>, Auburn University<sup>3</sup><br/>
### [[Paper (ACCV 2022)]](https://openaccess.thecvf.com/content/ACCV2022/papers/Rojas-Gomez_Inverting_Adversarially_Robust_Networks_for_Image_Synthesis_ACCV_2022_paper.pdf)

This is the official implementation of "Inverting Adversarially Robust Networks for Image Synthesis" accepted at ACCV 2022. Our code includes training, prediction and downstream task routines.

## Description
We empirically show that using adversarially robust (AR) representations as an image prior greatly improves the reconstruction accuracy of feature inversion models, and propose a robust encoding-decoding network for image synthesis and enhancement tasks.

This repository includes scripts to reproduce the results reported in our paper. These include:

- ### **Feature inversion**
![](readme/inv.jpg)
---

- ### **Style transfer**
![](readme/st.jpg)
---

- ### **Image denoising**
![](readme/den.jpg)
---

- ### **Anomaly detection**
![](readme/ad.jpg)
---


## Getting Started

Our code uses the [robustness](https://github.com/MadryLab/robustness) library for adversarially robust training and dataloading purposes. We also use [piqa](https://github.com/francois-rozet/piqa) to compute accuracy metrics. 

Please refer to the `conda_reqs.txt` and `pip_reqs.txt` for the full list of required packages to run our demo.


## Usage

The *scripts* folder contains bash scripts for training and prediction. Relevant parameters for each script are as follows:

1. `train_classifier.sh`: Trains either a standard or an AR AlexNet classifier.
    + **data**: training / validation set path.
    + **arch**: encoder architecture.
    + **dataset**: dataset identifier (e.g. ImageNet).
    + **adv-train**: [True] use adversarial training ot [False] standard training.
    + **constraint**: l-p constraint for adversarial attacks.
    + **attack-lr**: attack step size.
    + **attack-steps**: number of PGD steps.
    + **eps**: size of constrained l-p ball.
    + **epoch**: number of training epochs.
    + **lr**: learn rate for stochastic gradient descent.
    + **step-lr**: learn rate.
    + **batch-size**
    + **seed**: fixed random seed.
    
2. `train_generator.sh`: trains a feature inverter (image decoder) based on a pre-trained (standard or AR) using different optimization criteria.
    + **load_classifier/comparator**: location of pretrained models.
    + **data**: training dataset location.
    + **dataset**: dataset identifier.
    + **arch**: inverter architecture.
    + **upsample_mode**: unpooling strategy.
    + **adversarial_loss**: flag to apply GAN loss.
    + **adversarial_loss_weight**
    + **feature_loss**: norm used for feature loss (None is unused).
    + **feature_loss_weight**
    + **pixel_loss**: norm used for pixel loss (None is unused).
    + **pixel_loss_weight**
    + **gen/disc_lr**: Learn rate for genarator and discriminator.
    + **epochs**: training epochs.
    
3. `train_generator_singleclass.sh`: A particular case of training a feature inverter, where the training set corresponds to samples of a single class. Useful for anomaly detection.

    Same parameters as 2, plus:
    + **single_class_dataset**: dataset from which samples of a single class are taken.
    + **train_single_class**: class identifier (positives). The rest of classes are taken as out-of-distribution (negatives).
    

4. `train_denoiser.sh`: Trains a generator to recover clean images from features of images corrupted by clamped additive white gaussian noise. The script allows to equip the model with skip connections, as explained in the manuscript.

    Same parameters as 2, plus:
    + **wavelet_pooling**: flag to skip connections (Wavelet pooling).
    + **noise_level**: standard deviation of clamped AWGN applied to inputs during training.

5. `predict_generator.sh`: Takes a pre-trained autoencoder and uses it to predict on a dataset of interest (e.g. ImageNet).
    + **load_classifier**: location of pretrained classifier model.
    + **load_generator**: location of pretrained generator model.
    + **dataset**: dataset identifier.
    + **upsample_mode**: unpooling strategy.

6. `anomaly_detection.sh`: Takes a pre-trained feature inverter and uses it to identify between inliers and outliers (one vs. all anomaly detection), as explained in the manuscript.
    + **load classifier/comparator**: location of pretrained models.
    + **load generator**: location of pretrained generator model.
    + **in_class_dataset**: Unlabeled dataset from which positive and negative samples are classified.
    + **upsample_mode**: unpooling strategy.
    + **iterations**: number of optimization loops.
    + **feature_loss_weight**
    + **pixel_loss_weight**

7. `stylize.sh`: Takes three AlexNet autoencoders, each trained to invert a specific pooling level, and uses them for style transferring purposes.
    + **content / style images**: paths of reference datasets.
    + **load_convX_generator**: weights of pre-trained autoencoders.
    + **load_classifier/comparator**: location of classifier (encoder) and comparator (for Gram loss computation) pretrained weights.
    + **stylize_layers**: set of layers to use for feature alignment.
    + **compare_layers**: set of layers to use to compute the gram loss.
    
8. `denoise.sh`: Takes a pre-trained denoising autoencoder and uses it to reconstruct noisy images.
    + **load_classifier**: location of pretrained models.
    + **load_generator**: location of pretrained generator model.
    + **wavelet_pooling**: flag to use skip connections (Wavelet pooling).
    + **upsample_mode**: unpooling strategy.
    + **noise_level**: standard deviation of AWGN applied to inputs during evaluation.

Please refer to `./parsing.py` for a full set of input arguments for predict, anomaly_detection, stylization and denoise. Similarly, refer to the input arguments included in `./train_classifier` and `./train_generator` for their full set of training arguments.

## Checkpoints
[Pre-trained weights](https://drive.google.com/drive/folders/1fYHGvu0S9NLBsE8tj_jyDpKOeMNg7Ijd?usp=sharing) must be stored in the `./checkpoints` folder.
+ Feature inversion: Store in `./checkpoints/alexnet_inv` folder.
+ Anomaly detection: Store in `./checkpoints/alexnet_ad` folder.
+ Denoising: Store in `./checkpoints/alexnet_den` folder.
+ Style Transfer: Store in `./checkpoints/alexnet_st` folder.
+ VGG-19 model (Torchvision): Store in `./checkpoints/ext/tvis` folder.

## External References
Besides the use of the [robustness](https://github.com/MadryLab/robustness) library, our implementation of the wavelet pooling (skip connections) and feature alignment (whitening and coloring transformation) are inspired by the official [WCT2](https://github.com/clovaai/WCT2) implementation.

## Citation
If you use this software, please consider citing:

    @article{rojas_2021_inverting,
      title={Inverting Adversarially Robust Networks for Image Synthesis},
      author={Rojas-Gomez, Renan A and Yeh, Raymond A and Do, Minh N and Nguyen, Anh},
      journal={arXiv preprint arXiv:2106.06927},
      year={2021}
    }
