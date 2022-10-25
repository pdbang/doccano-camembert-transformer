# Data project with Quinten: Posology extraction from patient reports – Named Entity Recognition 🏥 

## Introduction 💊

Doccano is an excellent text labeling tool for named entity recognition, but the library that processes the output of this software is not very flexible and is not updated anymore.
This library has been developed in order to make it possible to use data from Doccano with Camembert using pandas and its dataframes.
Some simple modifications should allow to use other models. You just have to modify the tokenizer and the model in doccano_to_camembert.py and the character management in utils.

An example is available on the notebook ner_model.ipynb


## Tokenization 🪙

We used the platform Doccano for manual text labellisation, and CamemBERT for tokenization, which is a state-of-the-art language model for French based on the RoBERTa model. CamemBERT can only deal with numerical class values, so we had to adapt the initial output of Doccano.

The scoring of the final model was made with Kaggle, so we had to convert our output to a specific submission model.

The functions for tokenization and labels management are stored in [main/code/tokens_labels_management](main/code/tokens_labels_management)

## Running the NER CamemBERT-based model

The final NER model is located at [code/model/ner_model.ipynb](code/example_model/ner_model.ipynb).

