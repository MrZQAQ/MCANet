#  MCANet
  
Source code of paper: "MultiheadCrossAttention based network model for DTI prediction"
##  Model Overview
  
<img src="assets/Model_Overview.jpg"  width="800px"  height="400px"  title="Model Overview" >
##  Dependencies
  
Quick install: `pip install -r requirements.txt`
  
Dependencies:
- python 3.8+
- pytorch >=1.2
- numpy
- sklearn
- tqdm
- prefetch_generator
  
##  Usage
  
`python main.py <dataset> [-m,--model] [-s,--seed] [-f,--fold]`
  
Parameters:
- `dataset` : `DrugBank`, `Davis` or `KIBA`
- `-m` or `--model` : select `<model name>` from `MCANet`, `MCANet-B`, `onlyMCA` or `onlyPolyLoss`, *optional*, *default:*`MCANet`
- `-s` or `--seed` : set random seed, *optional*
- `-f` or `--fold` : set K-Fold number, *optional*
  
##  Project Structure
  
- DataSets: Data used in paper.
- assets: Readme resources.
- utils: A series of tools.
- config.py: model config.
- LossFunction.py: Loss function used in paper.
- main.py: main file of project.
- model.py: Proposed model in paper.
- README.md: this file
- requirements.txt: dependencies file
- RunModel.py: Train, validation and test programs.
  