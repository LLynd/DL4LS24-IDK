# DL4LS24-IDK
**Cell type classification based on gene expression data**

**Team Assignment for Deep Learning for Life Sciences 2024 course @MIM UW**

# Introduction

## Problem description

This project is focused on assigning cell types to specific cells using images prepared by a technique called Imaging Mass Cytometry (IMC). The main goal is to accurately identify and categorize different cell types present in the images. To achieve this, we use different machine learning methods.

The IMC technique was performed on a panel of 40 markers. These markers are specific characteristics or features that help us distinguish between different cell types. For this project, we’re focusing on three types of cells: tumor cells, mural cells, and leukocytes (a type of white blood cell).

The data used in this project includes several components:
Images: These are the pictures of the cells that we get from the IMC.

Expression Matrix: This is a table that shows the amount of each marker in each cell. Each row in the table represents a cell, and the columns represent the different markers.

Ground Truth Cell Annotation: This is our reference information that tells us the actual type of each cell.

## Data

## Methodology

## Results

# Usage

## Setting up

Pre-requirements: Linux system or Windows with WSL2 installed

To setup the virtual environment, run the setup.sh script with:

`./setup.sh`

To activate the created virtual environment, run:

`source venv_DL4LS24/bin/activate`

## Running your own experiments (Pre - Training)

To run experiments, You need to create a free account on WanDB.ai. 

Experiments are controled by instances of the Config dataclass. In this config, you specify all the constants of a particular experiment, including the method, wandb project and test set split. To run the experiment you then parse the created config to the `run_experiment` function. In case this is your first time running the experiments on a particular machine, you will need to login into wandb.

- example scripts

## Running inference on new data

The results are saved in the results subfolder.

# Authors
## Course Information
...

Tutors: Marcin Możejko, ...

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