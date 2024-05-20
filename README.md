# Introduction

This is the repository for the following research paper:

```text
Inference-Based No Learning Approach on Pre-trained BERT Model Retrieval using k-Nearest Neighbour algorithm
```

## Utilities

We provide python scripts to download tasks included in the benchmark dataset.

### Preparation

Create and activate a python virtual environment

For Unix

```bash
python -m venv .venv
source .venv/activate
```

For Windows

```powershell
python -m venv .venv
.venv¥Scripts¥activate.bat
```

Install required dependencies

```bash
pip intall -r requirements.txt
```

### Download tasks data

All tasks data can be download at the same time using the following command

```bash
python main.py datasets download --path <parent directory to save tasks data>
```

Task name can also be specified

```bash
python main.py datasets download --dataset <task 1> --dataset <task 2> --path <parent directory to save tasks data>
```

List of task name can be acquired using the following command

```bash
python main.py datasets list
```

## Benchmark dataset

Benchmark dataset csv file is provided inside the `benchmark-dataset` directory.

The structure of the csv file is as follows:

1. dataset_index: Index of the task (0~27)
2. dataset_name: Name of the task
3. model_name: Index of the pre-trained BERT model
4. test_acc: Accuracy score of the corresponding dataset's test split, calculated with the corresponding model fine-tuned, using the dataset's train and validation split
5. test_rel_acc: Relative (Normalized) accuracy score, with the accuracy score of best model normalized to 1 
6. rank: rank of the corresponding model based on the test_acc column.