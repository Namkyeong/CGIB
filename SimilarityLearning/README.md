# Drug-Drug Interaction Prediction Task

### Download and Create datasets
- Download AIDS OpenSSL dataset from https://github.com/cszhangzhen/H2MN.
- Download IMDB dataset from https://github.com/yunshengb/SimGNN.
- Put each datasets into ``datasets``.

### Hyperparameters
Following Options can be passed to `main_regression.py` and `main_classification.py`

`--dataset:`
Name of the dataset. Supported names are: AIDS700nef, IMDBMulti, and openssl_min50.  
usage example :`--dataset AIDS700nef`

`--lr:`
Learning rate for training the model.  
usage example :`--lr 0.001`

`--epochs:`
Number of epochs for training the model.  
usage example :`--epochs 500`

`--beta:`
Hyperparameters for balance the trade-off between prediction and compression.  
usage example :`--beta 1.0`

`--tau:`
Temperature hyperparameter for $\text{CGIB}_{\text{cont}}$.  
usage example :`--tau 1.0`