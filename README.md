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

