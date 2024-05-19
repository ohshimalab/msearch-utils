import os
from typing import List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, BaseSettings


class DatasetSetting(BaseSettings):
    train_val_test_split: Tuple[int, int, int] = (8, 1, 1)
    train_val_split: Tuple[int, int] = (9, 1)
    train_test_split: Tuple[int, int] = (9, 1)


class DatasetInfo(BaseModel):
    name: str
    subset_name: Optional[str]
    train_split_name: str = "train"
    validation_split_name: Optional[str] = "validation"
    test_split_name: Optional[str] = "test"
    data_column: str = "text"
    label_column: str = "label"


AVAILABLE_DATASETS = {
    "imdb": DatasetInfo(
        name="imdb",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "rotten_tomatoes": DatasetInfo(name="rotten_tomatoes"),
    "tweet_eval_emoji": DatasetInfo(
        name="tweet_eval",
        subset_name="emotion",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "tweet_eval_emotion": DatasetInfo(
        name="tweet_eval",
        subset_name="emoji",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "tweet_eval_hate": DatasetInfo(
        name="tweet_eval",
        subset_name="hate",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "tweet_eval_irony": DatasetInfo(
        name="tweet_eval",
        subset_name="irony",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "tweet_eval_offensive": DatasetInfo(
        name="tweet_eval",
        subset_name="offensive",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "tweet_eval_sentiment": DatasetInfo(
        name="tweet_eval",
        subset_name="sentiment",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "sst2": DatasetInfo(
        name="sst2",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name=None,
        data_column="sentence",
    ),
    "ag_news": DatasetInfo(
        name="ag_news", train_split_name="train", validation_split_name=None, test_split_name="test"
    ),
    "yelp_review_full": DatasetInfo(
        name="yelp_review_full", train_split_name="train", validation_split_name=None, test_split_name="test"
    ),
    "ethos_binary": DatasetInfo(
        name="ethos", subset_name="binary", train_split_name="train", validation_split_name=None, test_split_name=None
    ),
    "ade_corpus_v2_classification": DatasetInfo(
        name="ade_corpus_v2",
        subset_name="Ade_corpus_v2_classification",
        train_split_name="train",
        validation_split_name=None,
        test_split_name=None,
    ),
    "banking77": DatasetInfo(
        name="banking77", train_split_name="train", validation_split_name=None, test_split_name="test"
    ),
    "poem_sentiment": DatasetInfo(
        name="poem_sentiment",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
        data_column="verse_text",
    ),
    "snips_built_in_intents": DatasetInfo(
        name="snips_built_in_intents", train_split_name="train", validation_split_name=None, test_split_name=None
    ),
    "sms_spam": DatasetInfo(
        name="sms_spam", data_column="sms", train_split_name="train", validation_split_name=None, test_split_name=None
    ),
    "dair_ai_emotion": DatasetInfo(
        name="dair-ai/emotion",
        data_column="unsplit",
        train_split_name="train",
        validation_split_name=None,
        test_split_name=None,
    ),
    "onestop_english": DatasetInfo(
        name="onestop_english", train_split_name="train", validation_split_name=None, test_split_name=None
    ),
    "emo": DatasetInfo(name="emo", train_split_name="train", validation_split_name=None, test_split_name="test"),
    "twitter_financial_news_sentiment": DatasetInfo(
        name="zeroshot/twitter-financial-news-sentiment",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "financial_phrasebank": DatasetInfo(
        name="financial_phrasebank",
        subset_name="sentences_allagree",
        train_split_name="train",
        validation_split_name=None,
        test_split_name=None,
        data_column="sentence",
    ),
    "climatebert_climate_detection": DatasetInfo(
        name="climatebert/climate_detection",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "trec": DatasetInfo(
        name="trec",
        label_column="coarse_label",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "fancyzhx_dbpedia_14": DatasetInfo(
        name="fancyzhx/dbpedia_14",
        data_column="content",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "hate_speech18": DatasetInfo(
        name="hate_speech18", train_split_name="train", validation_split_name=None, test_split_name=None
    ),
    "clinc_oos": DatasetInfo(
        name="clinc_oos",
        subset_name="plus",
        label_column="intent",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "climatebert_tcfd_recommendations": DatasetInfo(
        name="climatebert/tcfd_recommendations",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "climatebert_climate_specificity": DatasetInfo(
        name="climatebert/climate_specificity",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "climatebert_climate_commitments_actions": DatasetInfo(
        name="climatebert/climate_commitments_actions",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "deysi_spam-detection-dataset": DatasetInfo(
        name="Deysi/spam-detection-dataset",
        train_split_name="train",
        validation_split_name=None,
        test_split_name="test",
    ),
    "hate_offensive": DatasetInfo(
        name="hate_offensive",
        data_column="tweet",
        train_split_name="train",
        validation_split_name=None,
        test_split_name=None,
    ),
    "OxAISH-AL-LLM_wiki_toxic": DatasetInfo(
        name="OxAISH-AL-LLM/wiki_toxic",
        data_column="comment_text",
        train_split_name="train",
        validation_split_name="validation",
        test_split_name="test",
    ),
    "mattymchen_mr": DatasetInfo(
        name="mattymchen/mr", train_split_name="test", validation_split_name=None, test_split_name=None
    ),
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


def get_dataset_dict(name: str, subset: Optional[str] = None) -> DatasetDict:
    if not subset:
        return load_dataset(name)
    return load_dataset(name, subset)


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
        ds_train, ds_val_test = (
            dsdict_train_val_test["train"],
            dsdict_train_val_test["test"],
        )
        dsdict_val_test = ds_val_test.train_test_split(test_size=0.5, seed=0)
        ds_val, ds_test = dsdict_val_test["train"], dsdict_val_test["test"]

    elif dataset_info.validation_split_name is None and dataset_info.test_split_name is not None:
        ds_train = dataset_dict[dataset_info.train_split_name]
        dsdict_train_val = ds_train.train_test_split(test_size=0.1, seed=0)
        ds_train, ds_val = dsdict_train_val["train"], dsdict_train_val["test"]
        ds_test = dataset_dict[dataset_info.test_split_name]

    elif dataset_info.validation_split_name is not None and dataset_info.test_split_name is None:
        ds_train = dataset_dict[dataset_info.train_split_name]
        dsdict_train_test = ds_train.train_test_split(test_size=0.1, seed=0)
        ds_train, ds_test = dsdict_train_test["train"], dsdict_train_test["test"]
        ds_val = dataset_dict[dataset_info.validation_split_name]

    else:
        ds_train = dataset_dict[dataset_info.train_split_name]
        ds_val = dataset_dict[dataset_info.validation_split_name]
        ds_test = dataset_dict[dataset_info.test_split_name]

    return DatasetDict({"train": ds_train, "val": ds_val, "test": ds_test})


def save_dataset(root_dir: str, dataset_dict: DatasetDict, dataset_info: DatasetInfo):
    """データセットの保存

    Parameters
    ----------
    dataset_dict : DatsetDict
        _description_
    dataset_info : DatasetInfo
        _description_
    """
    if not dataset_info.subset_name:
        dataset_dir = os.path.join(root_dir, dataset_info.name)
    else:
        dataset_dir = os.path.join(root_dir, dataset_info.name + "_" + dataset_info.subset_name)
    dataset_dir = dataset_dir.replace("/", "_")
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for split in dataset_dict:
        df = pd.DataFrame(dataset_dict[split])
        df = df.rename(columns={dataset_info.label_column: "label", dataset_info.data_column: "text"})
        df = df[["text", "label"]]
        split_csv_path = os.path.join(dataset_dir, split + ".csv")
        df.to_csv(split_csv_path, index=False)


def download_dataset(root_dir: str, dataset_name: str):
    """指定のデータセットをダウンロードする

    Parameters
    ----------
    root_dir : str
        _description_
    dataset_name : str
        _description_
    """
    dataset_info = get_dataset_info(dataset_name)
    dataset_dict = get_dataset_dict(name=dataset_info.name, subset=dataset_info.subset_name)
    dataset_dict_splitted = split_dataset_dict(dataset_dict, dataset_info)
    save_dataset(root_dir, dataset_dict_splitted, dataset_info)
