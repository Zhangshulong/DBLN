# DBLN
![Overall architecture of the DBLN](figures/DBLN.png)

## Introduction
DBLN (Deep Bidirectional Learning Network) is a deep learning model for time series modeling, suitable for multivariate time series classification and forecasting tasks. This project implements DBLN and related experiments, supporting various public datasets.

## Features
- Supports multivariate time series modeling
- End-to-end training, easy to extend
- Multiple data augmentation and normalization methods
- Compatible with mainstream dataset formats

## Installation & Dependencies
```bash
pip install -r requirements.txt
```
Python 3.8+ is recommended. It is suggested to use a virtual environment or Anaconda.

## Quick Start
Example for training:
```bash
python run.py --dataset BasicMotions
```
Parameters can be adjusted in run.py or related config files.

## Dataset
The `data/UEA/` directory contains only a subset of datasets for demonstration. The complete UEA multivariate time series archive can be downloaded from the official website:

[https://www.timeseriesclassification.com/](https://www.timeseriesclassification.com/)

## Training & Testing
You can train and test the model using the run.py script, with customizable command-line arguments.

## Results
Experimental results and visualizations will be supplemented later, or refer to the original paper.

```
