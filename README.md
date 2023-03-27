#  MCANet

Paper is available at [Briefing in Bioinformatics](https://doi.org/10.1093/bib/bbad082)

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
- `dataset` : `DrugBank`, `Davis` , `KIBA` , `Enzyme` , `GPCRs` or `ion_channel`
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
  
## Citation

```
@article{10.1093/bib/bbad082,
    author = {Bian, Jilong and Zhang, Xi and Zhang, Xiying and Xu, Dali and Wang, Guohua},
    title = {MCANet: shared-weight-based MultiheadCrossAttention network for drug–target interaction prediction},
    journal = {Briefings in Bioinformatics},
    volume = {24},
    number = {2},
    pages = {bbad082},
    year = {2023},
    month = {03},
    issn = {1477-4054},
    doi = {10.1093/bib/bbad082}
}
```
