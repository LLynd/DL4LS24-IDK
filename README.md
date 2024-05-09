# DL4LS24-IDK
**Cell type classification based on gene expression data**

**Team Assignment for Deep Learning for Life Sciences 2024 course @MIM UW**

# Introduction

## Problem description

This project is focused on assigning cell types to specific cells using images prepared by a technique called Imaging Mass Cytometry (IMC). The main goal is to accurately identify and categorize different cell types present in the images. To achieve this, we use different machine learning methods.

The IMC technique was performed on a panel of 40 markers. These markers are specific characteristics or features that help us distinguish between different cell types. For this project, we’re focusing on three types of cells: tumor cells, mural cells, and leukocytes (a type of white blood cell).

## Data

The data used in this project includes several components:
- Images: These are the pictures of the cells that we get from the IMC.
- Expression Matrix: This is a table that shows the amount of each marker in each cell. Each row in the table represents a cell, and the columns represent the different markers.
- Ground Truth Cell Annotation: This is our reference information that tells us the actual type of each cell.

## Methodology

...

## Results

...

# Usage

## Setting up

Pre-requirements: Linux system or Windows with WSL2 installed

To setup the virtual environment, run the setup.sh script with:

`./setup.sh`

To activate the created virtual environment, run:

`source venv_DL4LS24/bin/activate`

## Running your own experiments (Pre - Training)

To run experiments, You first need to create a free account on WanDB.ai.

Experiments are controlled by instances of the Config dataclass. In such a config, you specify all the constants of a particular experiment, including the method (xgboost, linear, starling or logistic), wandb project and test set size. To run the experiment you then parse the created config to the `run_experiment` function. In case this is your first time running the experiments on a particular machine, you will need to login into wandb through a prompt that will appear in the terminal. You will find the results of the training in the wandDB dashboard. 

You can find an example script for running a pretraining experiment at the scripts subdirectory.

## Running inference on new data

Similarly to experiments, inference runs are controlled by instances of the Config dataclass. In such a config, You need to specify the path to a pre trained model, the path to the new data and the method (make sure it aligns with the model). To run the inference parse the created config to the `run_inference` function. The results are saved in the results subfolder.

# Authors
## Course Information
Deep Learning in Life Sciences @MIM UW

Code 1000-2M23DLS

2023/2024 Summer semester

Course coordinator: Bartosz Wilczyński

Tutors: Marcin Możejko,  Maciej Sikora

https://deeplife4eu.github.io

## Author contributions
**Team IDK:**

Asia Dąbrowska - EDA

Antoni Janowski - Coding

Miriam Lipniacka - Coding

Łukasz Niedźwiedzki - Coding

Anna Szymik - EDA

Maciej Wiśniewski - EDA

## Contact
...