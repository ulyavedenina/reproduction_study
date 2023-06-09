# Neural Pragmatic NLG project

This repository contains the reproduction attempt of a [pragmatic model](https://arxiv.org/abs/2004.14451) built by Nie et al. and its test on the A3DS data set. Both data sets (cub and A3DS) as well as the model training notebook (ran on a paid Google Colab runtime) and the final report are attached.

Most of the following instructions were taken from https://github.com/windweller/Pragmatic-ISIC and https://github.com/salaniz/pytorch-gve-lrcn .

## Installation
Following the repository (https://github.com/windweller/Pragmatic-ISIC), this implementation uses Python 3, PyTorch and pycocoevalcap.  
All dependencies can be installed into a conda environment with the provided environment.yml file (original from Windweller and also dependencies needed to make the sentence classifier work).

1. Clone the repository
```shell
git clone https://github.com/ulyavedenina/reproduction_study.git
cd reproduction_study
```
2. Create conda environment
```shell
conda env create -f environment.yml
```
3. Activate environment
```shell
conda activate gve-lrcn
```

4. Download pre-trained model and data
```bash
sh rsa-file-setup.sh 
```

5. Install other packages

```bash
pip install -r requirements.txt
```
Note: Most likely one must also run: 
```bash
python -m spacy download en
``` 
to make the code work.

6. Download A3DS Dataset: Since the dataset is too big to upload in GitHub, one can download it by following the instructions here: 
https://github.com/polina-tsvilodub/3dshapes-language . Before downloading the dataset, one must create a file data/A3DS in their reproduction_study repository. Note: we used the whole A3DS dataset and not only the sandbox.

## Usage
1. Train GVE on A3DS
* To train GVE on A3DS we first need a sentence classifier:
```
python main.py --model sc --dataset 3d
```
* Copy the saved model to the default path (or change the path to your model file) and then run the GVE training:
```
cp ./checkpoints/sc-3d-D<date>-T<time>-G<GPUid>/best-ckpt.pth ./data/3d/sentence_classifier_ckpt.pth
python main.py --model gve --dataset 3d --sc-ckpt ./data/3d/sentence_classifier_ckpt.pth
```

3. Evaluation
* By default, model checkpoints and validation results are saved in the checkpoint directory using the following naming convention: `<model>-<dataset>-D<date>-T<time>-G<GPUid>`
* To make a single evaluation on the test set, you can run
```
python main.py --model gve --dataset 3d --eval ./checkpoints/gve-3d-D<date>-T<time>-G<GPUid>/best-ckpt.pth
```

NOTE: The original code from the Windweller repository and our replication code is only working on linux and Mac. Trying to make it run in windows is not recommended.

## Table of contents

1. main.py: this is the first python code that is called. It contains the main function, the creation of a dataset and a dataloader object. It also creates a model and a trainer. 
2. models folder: this folder contains definitions of the gve class, the lrcn class and also the sentence_classifier class amongst other classes.
3. ResultsCUB folder: contains the results we obtained from evaluting the captioner on the CUB dataset.
4. train folder: this folder contains the actual trainer code for each class. 
5. utils folder: all other help functions and classes are located here. In the subfolder data one can find the classes for the different datasets.
6. colabScript.ipynb: this is a script to run the code of the repository on google colab. It presupposes that the repository is uploaded on Google Drive(including the dataset)
7. Report.pdf: file containg our summary for the project.
