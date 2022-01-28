# DataMiningForBigDataProject

- [DataMiningForBigDataProject](#dataminingforbigdataproject)
- [Running the code](#running-the-code)
  - [Download the data](#download-the-data)
  - [Create the environment](#create-the-environment)
  - [Run the code](#run-the-code)
    - [With the Python script](#with-the-python-script)
    - [With the Python interactive notebook](#with-the-python-interactive-notebook)
  - [Outputs](#outputs)

# Running the code

Here are the explaination to run the code.

## Download the data

Download the data and put it in the root of the project directory.

## Create the environment

Create env, activate it and install packages

**Windows (Powershell)**

```powershell
python -m venv .venv
.venv\Scripts\activate.ps1
python -m pip install -r requirements.txt
```

**Unix (Bash)**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the code

### With the Python script

Then run the python file [`test.py`](test.py) with the command

```powershell
python test.py
```

### With the Python interactive notebook

You can also run the script with the notebook [test.ipynb](test.ipynb).

## Outputs

The output files are locate in the `big_data_project` directory. You will find

- [`output.csv`](big_data_project/output.csv) - the [`test.csv`](big_data_project) file merged with the prediction
- [`cmatrix.png`](big_data_project/cmatrix.csv) - the confusion matrix
- [`roc.png`](big_data_project/roc.csv) - the roc curve
