# Fool-X
Fool-X: A Novel Adversarial Attack against Deep Learning Models

Fool-X is a new adversarial machine learning attack that uses the maximum hyperplane distance along with the gradient sign to calculate the minimum perturbation. Adversarial perturbations are small, imperceptible modifications in input data that cause a deep learning classifier to detect data clearly belonging to one class as another. This is commonly applied to images, where an image that is obviously of one object can be forced to be seen by a classifier as a different object. Fool-X using an input image, a classifier, and an epsilon Ɛ scaling value sucessfully creates a small but powerful perturbation.

This repository contains the code for the implementation and analysis of the Fool-X attack.

- Batch testing python files for testing against large groups of the CIFAR and ILSVRC dataset (ILSVRC-2012 required to be downloaded).
- Single image testing for running Fool-X on a single image at a time.
- Immunity training using Fool-X examples followed by batch testing to analyze robustness.
- Transferability testing to test the ability of Fool-X adversarial examples to work between network architectures.
- The Fool-X Algorithm itself. 
