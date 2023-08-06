# EXPLAINABLE CLASSIFICATION OF X-RAY MEDICAL IMAGES USING BAYESIAN DEEP LEARNING

This repository contains the code base for the project for the Advanced Machine Learning (02460) course at DTU.

Please refer to our paper [here](https://github.com/MadsBirch/Bayesian_Explainable_AI/blob/614bbe7238ed773177b1a27d2f290324121c0ecd/report.pdf) for further and detailed explantions.

The collaborators were:
- Grigor Spalj
- Alessandro Contini
- Anders David Lægdsgaard Lassen
- Mads Birch Sørensen


## ABSTRACT
Deep Neural Networks (DNNs) have often been found to be poorly calibrated due to overconfident predictions. At the same time the increasing complexity of modern DNNs have led to these models being considered black boxes. For that reason, various explanation methods have been proposed to uncover what features influence the predictions of DNNs. Bayesian Deep Neural Networks (BNNs) infer the entire posterior distribution of the weights, meaning that uncertainty about predictions is inherent. In this code we investigate how a Bayesian approach can improve the calibration and explainability of modern DNNs. We implemented and trained a Convolutional Neural Network (CNN) on the MURA dataset to perform a classification task and found that the Bayesian framework resulted in a significant reduction in calibration error and improved the interpretability of the implemented visual explanation methods.


## Results

### Numeric Results
We found that the calibration error of the Bayesian CNN was dramatically lower.

<figure>
    <img src="https://github.com/MadsBirch/Bayesian_Explainable_AI/blob/main/figures/numeric_results.png"  width="50%">
    <figcaption> Fig. 1 - Accuracy and calibration error for the MAP and Laplace and Ensemble models. </figcaption>
</figure>


<figure>
    <img src="https://github.com/MadsBirch/Bayesian_Explainable_AI/blob/main/figures/heat.png"  width="50%">
    <figcaption> Fig. 2 - Individual CAMs for each of the five models in the ensemble. _Top_: </figcaption>
</figure>


<figure>
    <img src="https://github.com/MadsBirch/Bayesian_Explainable_AI/blob/main/figures/ensemble.png"  width="50%">
    <figcaption> Fig. 3 - Aggregated CAMs for the models in the ensemble.</figcaption>
</figure>
