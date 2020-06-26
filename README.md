# Deep Learning Enabled Prediction of 5-Year Survival in Pediatric Genitourinary Rhabdomyosarcoma

This repository is for our 2020 paper entitled, "Deep Learning Enabled Prediction of 5-Year Survival in Pediatric Genitourinary Rhabdomyosarcoma" (under review).

## Usage

### Requirements:
PyTorch 1.4 <br />
Python 3.6 <br />
Numpy 1.16 <br />
Matplotlib 3.1 

### Getting started: 

Clone the repository:
```
git clone https://github.com/alvarozamora/UroRhabdo
cd UroRhabdo
```

* Using your text editor, adjust the path name of uro.csv in the 9th line of preprocessing.py as appropriate.

* Execute the following to preprocess the data

```
python preprocessing.py
```

* Execute the following to train the models. Model architecture is stored in model.py

```
python train.py
```
