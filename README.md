# CS60075-Team-11-Task-1

Lexical Complexity Prediction Shared Task Codes for Team 11

## Installation and running

* Requirements - **Python 3.6** (Tested in this environment)

* All the below commands are for a Linux system

* Create a virtual environment (Highly recommended to prevent dependency conflicts)

```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
```

**Note**: Ensure pip is updated in the virtual enviroment by executing the last command above. Some dependencies require the latest version of pip

* Basic Requirements

```sh
    pip install -r requirements.txt
```

* GloVE Embeddings

```sh
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove*.zip
```

* Running the code

```sh
    python lcp_shared_task_overall_model.py
```

## Trained Models

* We saved certain models during training (such as biLSTM for context probability prediction), to help reduce training the same models while experimenting. These models are available in the [TrainedModels](./TrainedModels) subdirectory.

* These trained and saved models are used while running the overall model script

## Experiments

* Transformer Models and the Neighbourhood Aggregate Model with the Google Colab links are located in the [Experiments](./Experiments) folder
