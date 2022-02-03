# Tensorflow Implementation For "SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction"

Project for [Vision and Perception](https://sites.google.com/diag.uniroma1.it/alcorlab-diag/teaching-thesis?authuser=0#h.bvp6qx4bvrrm), DIAG, Sapienza University in Roma

![sapienza-big](https://user-images.githubusercontent.com/24941293/152373391-ac062aac-750a-45cd-bf40-9851cf2911f1.png)

## Table of Contents
  - [Introduction](#Introduction)
  - [Running](#Running)
  - [Team](#Team)

## Introduction

![image](https://user-images.githubusercontent.com/24941293/152379633-983f49ce-4b44-4790-bee9-d9514b204deb.png)


This project is the **tensorflow implementation** for paper "SGCN:Sparse Graph Convolution Network for Pedestrian Trajectory Prediction" in CVPR 2021, and we also use a new dataset [MOT-15](https://motchallenge.net/data/MOT15/) to test it.

[Here](https://github.com/shuaishiliu/SGCN) is the original pytorch code. We rewrite the original author's code by tensorflow.

[Paper](https://arxiv.org/pdf/2104.01528.pdf)

## Running

Keep this file structure and run the file "Main_SGCN_MOT15.ipynb" directly by [jupyter notebook](https://jupyter.org/).

"metrics.py" implements loss functions.

"model.py" implements network model.

"utils.py" processes dataset.

"Visualization.ipynb"  shows the visualization of the trojectory.

"dataset/data" file includes MOT-15 dataset，which has been processed.

"dataset/ETH" file includes eth dataset.

"dataset/hotel" file includes hotel dataset.

## Team

- PK
- SCC



















