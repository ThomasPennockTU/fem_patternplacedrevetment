# Finite element model for pattern-placed revetments
This repository contains a finite element model developed for the publication 'Estimating the effect of assumed initial damage to the hydraulic stability of pattern-placed revetments on dikes using finite element modeling' by van der Vegt et al. (2024) ([article](https://doi.org/10.1016/j.coastaleng.2024.104484)). This study focuses on impact of damage on the structural integrity of pattern-placed revetments. The provided scripts are able to generate a batch of FEM models with different configurations of the model using Abaqus Explicit 2019. The model simulates five wave impacts on a Basalton STS+ revetment.

## Model Description
The model is specifically designed for a top layer consisting of Basalton STS+ elements with a granular filter layer. The geometric definitions for the top layer elements can be found in the 'BasaltonSTSplus' folder. Additionally, it is possible to model the joint filling material as solids, which are defined in the 'BasaltonSTSplus_jointfilling' folder.

The model allows for the simulation of various types of damage in addition to an intact pattern-placed revetment. The supported damage scenarios include deformation, washed-out joint filling, a missing element, and a deformed toe.

## Usage
The script needs to be executed using the Python console within Abaqus Explicit 2019. The main script file for the model is named 'RevetmentModel.py'. Within this Python file there is a path to an Excel file where you can define the models.

Before running the model, you can modify the input parameters by editing the input Excel file. An example of the input Excel file is provided in 'example.xlsx'.

## Example Models
Several example models, corresponding to the configurations defined in 'example.xlsx', are included in the 'Example models' folder. These examples can serve as references for setting up and running simulations using the model.
