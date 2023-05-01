from pathlib import Path
import os
from typing import Optional

import typer

from msearch_utils.utils import get_available_datasets_names, download_dataset

app = typer.Typer()

datasets_app = typer.Typer()

app.add_typer(datasets_app, name="datasets")

@datasets_app.command("download")
def download(path: Optional[Path] = typer.Option(None)):
    if path is None:
        print("No rootdir")
        raise typer.Abort()
    if path.is_file():
        print("Root dir can not be file")
        raise typer.Abort()
    if not path.exists():
        os.makedirs(path) 
    available_datasets = get_available_datasets_names()
    for dataset_name in available_datasets:
        download_dataset(path, dataset_name)
    
    
    
if __name__ == "__main__":
    app()
