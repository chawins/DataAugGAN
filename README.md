# DataAugGAN: Data Augmentation with GAN

This repository is part of Princeton University's computer vision (COS 429) course. It is my final project which aims to explore a capability of GAN as a method for data augmentation. For more information, please see the [project report](https://drive.google.com/file/d/1Zd8qNYBNzC3FvGKA1PTvEuTsnm1_X3_r/view?usp=sharing).

### Code Description
- `run_acgan_mnist.py`: train ACGAN on MNIST
- `run_acgan_cifar10.py`: train ACGAN on CIFAR-10
- `data-aug-gan.ipynb`: Carry out all the experiments described in the report
- Most of utility, model definition, etc. are in `lib/`

### Contributor
Chawin Sitawarin (chawins@princeton.edu)

The GAN code is adapted from:  
1. https://github.com/lukedeo/keras-acgan   
2. https://github.com/King-Of-Knights/Keras-ACGAN-CIFAR10
