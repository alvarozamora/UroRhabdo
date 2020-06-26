# Deep Learning Enabled Prediction of 5-Year Survival in Pediatric Genitourinary Rhabdomyosarcoma

This repository is for our 2020 paper entitled, "Deep Learning Enabled Prediction of 5-Year Survival in Pediatric Genitourinary Rhabdomyosarcoma" (under review).

## Usage

### Requirement:
PyTorch 1.4 <br />
Python 3.6 <br />
Numpy 1.6 <br />
Matplotlib 3.1

### Getting started: 

Clone the repository:
```
git clone https://github.com/alvarozamora/UroRhabdo
cd UroRhabdo
```
* Open preprocessing.py with your text editor and change the uro.csv file path on line 9 as appropriate.  

* Execute the following command to preprocess the data stored in uro.csv
```
python preprocess.py 
```
* Execute the following command to train the model (Note: model architecture is stored in model.py)
```
python train.py 
```
