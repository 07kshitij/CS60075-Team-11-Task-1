# CS60075-Team-11-Task-1

Lexical Complexity Prediction Shared Task Codes for Team 11

## Installation and Requirements

* **Requirements**

  * **Python 3.6** (Tested in this environment)
  * Cuda

* Note - All the below commands are for a Linux system

* Clone this repository

```sh
    git clone https://github.com/07kshitij/CS60075-Team-11-Task-1.git

    cd CS60075-Team-11-Task-1/
```

* Create a **Virtual Environment** (**Highly recommended** to prevent dependency conflicts)

```sh
    python3 -m venv venv

    source venv/bin/activate
    # On Windows - venv\Scripts\activate

    pip install --upgrade pip
```

**Note**: Ensure pip is updated in the virtual enviroment by executing the last command above. Some dependencies require the latest version of pip. Please ensure to execute all the 3 commands above

* Download the required dependencies

```sh
    pip install -r requirements.txt
```

* Download GloVE Embeddings

```sh
    wget http://nlp.stanford.edu/data/glove.6B.zip

    unzip glove*.zip
```

## Running the code

* Using the best models saved during training

  * We saved the best checkpoints during training. These models are available in the [TrainedModels](./TrainedModels) subdirectory.

  * To run the Overall Model using the best checkpoints saved, execute the below command.

    ```sh
        python lcp_shared_task_overall_model.py
    ```

* Retraining everything from scratch

  * To retrain all the constitutent models and run the whole architecture from scratch, execute the below command.

  * **Note**:  Slightly different results can be obtained than reported in the report due to certain non-determinism in PyTorch computations. We tried to seed as much random number generators as possible, but some non-determinism still exists which we can't control through external arguments to PyTorch.

    ```sh
        python lcp_shared_task_overall_model.py 0
    ```

## Running the code (Using Google Colab Notebook)

* The Overall solution model can also be run following the instructions in [this](https://colab.research.google.com/drive/1ZrP0q5qa3zjU8yBdw7qB6IRShFNBO3sv?usp=sharing) Colab Notebook

## Experiments

* Transformer Models and the Neighbourhood Aggregate Model with the Google Colab links are located in the [Experiments](./Experiments) subdirectory.

## Final Presentation Video

* The final video can be accessed by the file [cs60075_team11_FinalPresentation.mp4](./cs60075_team11_FinalPresentation.mp4)
