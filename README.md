# BERT for Question Answering (SQuAD)

BERT for Question Answering using SQuAD v1.1.
This implementation uses the pretrained BERT-base model (i.e. BertModel.from\_pretrained('bert-base-uncased')) and builds a simple fully-connected layer on top for predicting answer spans given SQuAD instances.

## Run

	python train.py

## Dataset

Using the SQuAD v1.1 dataset downloaded from the [original website](https://rajpurkar.github.io/SQuAD-explorer/) .
