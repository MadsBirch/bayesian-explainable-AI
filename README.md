# EXPLAINABLE CLASSIFICATION OF X-RAY MEDICAL IMAGES USING BAYESIAN DEEP LEARNING

## ABSTRACT
Deep Neural Networks (DNNs) have often been found to be poorly calibrated due to overconfident predictions. At the same time the increasing complexity of modern DNNs have led to these models being considered black boxes. For that reason, various explanation methods have been proposed to uncover what features influence the predictions of DNNs. Bayesian Deep Neural Networks (BNNs) infer the entire posterior distribution of the weights, meaning that uncertainty about predictions is inherent. In this code we investigate how a Bayesian approach can improve the calibration and explainability of modern DNNs. We implemented and trained a Convolutional Neural Network (CNN) on the MURA dataset to perform a classification task and found that the Bayesian framework resulted in a significant reduction in calibration error and improved the interpretability of the implemented visual explanation methods.

Please refer to our paper [here](https://github.com/MadsBirch/Bayesian_Explainable_AI/blob/614bbe7238ed773177b1a27d2f290324121c0ecd/report.pdf) for further and detailed explantions.

This repository contains the code base for the project for the Advanced Machine Learning (02460) course at DTU.

The collaborators were:
- Grigor Spalj
- Alessandro Contini
- Anders David Lægdsgaard Lassen
- Mads Birch Sørensen


## Results

### Numeric Results

![Figure 1](https://github.com/MadsBirch/Bayesian_Explainable_AI/blob/main/cam_ensemble_rotation/figures/aggregation_results.png?raw=true) <br />
*Figure 1*
