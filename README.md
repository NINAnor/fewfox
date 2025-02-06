# fewfox: creating an efficient model using very few data

## Setup

### Install the requirements

To install the packages we use the `poetry` package manager.

```bash
poetry install
```

### Get some data

At the moment, the pipeline works with the ESC50 dataset. You can download it using the script `/data/get_ESC50.sh`. The [ESC50 dataset](https://github.com/karolpiczak/ESC-50) is composed of **50 semantical** classes with **40 sounds per class**.

Once you have downloaded the ESC50 dataset, we can create a "miniESC50" dataset. The miniESC50 dataset is composed of 5 classes. Each class contains 5 sounds for training, 5 sounds for validating and the remaining sounds (30) for testing the model. 

After setting the correct MINI_ESC50_PATH in `configs/paths/default.yaml`, create the miniESC50 dataset using the following command:

```bash
poetry run python miniesc50.py 
```

### Change the config

The most important is to adapt `/configs/paths/default.yaml` using your paths

### Run the pipeline

You can train the prototypical model using the following command:

```bash
poetry run python src/protopipeline.py
```

The model weights will be stored in the `lightning_logs` folder.

### Evaluate your model

You can evaluate the performance of your model using the command:

```bash
poetry run python src/evaluate.py
```

This should return model performance and a 2D image displaying the embeddings and the prototypes




