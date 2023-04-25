from pydantic import BaseModel, BaseSettings
from typing import Tuple, List, Optional
from datasets import DatasetDict, Dataset, load_dataset
import os
import pandas as pd

class DatsetSetting(BaseSettings):
    train_val_test_split: Tuple[int, int, int] = (8,1,1)
    train_val_split: Tuple[int, int] = (9,1)
    train_test_split: Tuple[int, int] = (9,1)

class DatasetInfo(BaseModel):
    name: str
    train_split_name: str = "train"
    validation_split_name: Optional[str] = "validation"
    test_split_name: Optional[str] = "test"
    data_column: str = "text"
    label_column: str = "label"

AVAILABLE_DATASETS = {
    "imdb": DatasetInfo(name="imdb", train_split_name="train", validation_split_name=None, test_split_name="test"),
    "rotten_tomatoes":  DatasetInfo(name="rotten_tomatoes"),
}

def get_available_datasets_names() -> List[str]:
    """データセット名のリストの取得
    
    Returns
    -------
    List[str]
        データセット名のリスト 
    """
    return list(AVAILABLE_DATASETS.keys())

def get_dataset_info(name: str) -> Optional[DatasetInfo]:
    """データセット情報を取得

    Parameters
    ----------
    name: str
        データセット名
    
    Returns
    -------
    Optional[DatasetInfo]:
        データセット情報
    """
    return AVAILABLE_DATASETS.get(name, None)

def get_dataset_dict(name: str) -> DatasetDict:
    return load_dataset(name)

def split_dataset_dict(dataset_dict: DatasetDict, dataset_info: DatasetInfo) -> DatasetDict:
    """データセットを適切なフォーマットにsplitする

    Parameters
    ----------
    dataset_dict : DatasetDict
        hugging_faceのデータセット辞書
    dataset_info : DatasetInfo
        データセット情報

    Returns
    -------
    DatasetDict
        splitしたデータセット辞書
    """
    if dataset_info.validation_split_name is None and dataset_info.test_split_name is None:
        ds_train = dataset_dict[dataset_info.train_split_name]
        dsdict_train_val_test = ds_train.train_test_split(test_size=0.2, seed=0)
        ds_train, ds_val_test = dsdict_train_val_test["train"], dsdict_train_val_test["test"]
        dsdict_val_test = ds_val_test.train_test_split(test_size=0.5, seed=0)
        ds_val, ds_test = dsdict_val_test["train"], dsdict_val_test["test"]

    elif dataset_info.validation_split_name is None and dataset_info.test_split_name is not None:
        ds_train = dataset_dict[dataset_info.train_split_name]
        dsdict_train_val = ds_train.train_test_split(test_size=0.2, seed=0)
        ds_train, ds_val = dsdict_train_val["train"], dsdict_train_val["test"]
        ds_test = dataset_dict[dataset_info.test_split_name]

    elif dataset_info.validation_split_name is not None and dataset_info.test_split_name is None:
        ds_train = dataset_dict[dataset_info.train_split_name]
        dsdict_train_test = ds_train.train_test_split(test_size=0.2, seed=0)
        ds_train, ds_test = dsdict_train_test["train"], dsdict_train_test["test"]
        ds_val = dataset_dict[dataset_info.validation_split_name]
    
    else:
        ds_train = dataset_dict[dataset_info.train_split_name]
        ds_val = dataset_dict[dataset_info.validation_split_name]
        ds_test = dataset_dict[dataset_info.test_split_name]
    
    return DatasetDict({
        "train": ds_train,
        "val": ds_val,
        "test": ds_test
    })
        
def save_dataset(root_dir: str, dataset_dict: DatasetDict, dataset_info: DatasetInfo):
    """データセットの保存

    Parameters
    ----------
    dataset_dict : DatsetDict
        _description_
    dataset_info : DatasetInfo
        _description_
    """
    dataset_dir = os.path.join(root_dir, dataset_info.name)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for split in dataset_dict:
        df = pd.DataFrame(dataset_dict[split])
        split_csv_path = os.path.join(dataset_dir, split + ".csv")
        df.to_csv(split_csv_path, index=False)


    