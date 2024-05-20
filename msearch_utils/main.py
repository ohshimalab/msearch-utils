import os
from pathlib import Path
from typing import List, Optional

import typer

from msearch_utils.utils import download_dataset, get_available_datasets_names

app = typer.Typer()

datasets_app = typer.Typer()

app.add_typer(datasets_app, name="datasets")


@datasets_app.command("list")
def list():
    """利用可能なデータセットを表示する"""
    available_datasets = get_available_datasets_names()
    print(f"There are {len(available_datasets)} datasets available:")
    print("-" * 50)
    for dataset_name in available_datasets:
        print(dataset_name)


@datasets_app.command("download")
def download(
    dataset: Optional[List[str]] = typer.Option(None),
    path: Optional[Path] = typer.Option(None),
):
    """データセットをダウンロードする

    Parameters
    ----------
    dataset_name: Optional[List[str]]
        ダウンロード対象データセット名
    path : Optional[Path], optional
        保存パス

    """
    if path is None:
        print("No rootdir")
        raise typer.Abort()
    if path.is_file():
        print("Root dir can not be file")
        raise typer.Abort()
    if not path.exists():
        os.makedirs(path)
    available_datasets = get_available_datasets_names()
    if not dataset:
        for dataset_name in available_datasets:
            download_dataset(path, dataset_name)
    else:
        not_supported_datasets = [dataset_name for dataset_name in dataset if dataset_name not in available_datasets]
        if not_supported_datasets:
            print("サポートされていないデータセット:", ",".join(not_supported_datasets))
            raise typer.Abort()
        else:
            for dataset_name in dataset:
                download_dataset(path, dataset_name)
