import PySimpleGUI as sg
import os
from typing import List
from datasets import DatasetDict, load_dataset

download_dir = ""
dir_selection_row = [
    sg.Text("Directory"),
    sg.In(key="-DIR-", enable_events=True),
    sg.FolderBrowse(key="-DIR BROWSE-"),
]
model_selection_row = [
    sg.Listbox(
        values=["imdb", "rotten_tomatoes"],
        key="-DATASET LIST-",
        size=(40, 20),
        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
    ),
    sg.VerticalSeparator(),
    sg.MLine(key="-LOG-" + sg.WRITE_ONLY_KEY, size=(40, 20), disabled=True),
]
download_row = [
    sg.Button("Download", size=(80, 1), enable_events=True, key="-DOWNLOAD-"),
]
progress_row = [sg.ProgressBar(key="-PROGRESS-", max_value=100)]
layout = [dir_selection_row, model_selection_row, download_row, progress_row]
window = sg.Window("Demo", layout)


def download_datasets(dataset_names: List[str]):
    window["-DOWNLOAD-"].update(disabled=True)
    window["-DATASET LIST-"].update(disabled=True)
    window["-DIR-"].update(disabled=True)
    window["-DIR BROWSE-"].update(disabled=True)
    window["-PROGRESS-"].update(current_count=0)
    for i, dataset_name in enumerate(dataset_names):
        window["-LOG-" + sg.WRITE_ONLY_KEY].print(
            "Start downloading", dataset_name, "..."
        )
        load_dataset(dataset_name)
        window["-LOG-" + sg.WRITE_ONLY_KEY].print("Finish downloading", dataset_name)
        window["-LOG-" + sg.WRITE_ONLY_KEY].print(
            "Start saving", dataset_name, "to disk..."
        )
        window["-LOG-" + sg.WRITE_ONLY_KEY].print(
            "Finish saving", dataset_name, "to disk"
        )

        window["-PROGRESS-"].update(current_count=(i + 1) * 100 // len(dataset_names))
    window["-DOWNLOAD-"].update(disabled=False)
    window["-DATASET LIST-"].update(disabled=False)
    window["-DIR-"].update(disabled=False)
    window["-DIR BROWSE-"].update(disabled=False)


while True:
    event, values = window.read()
    if event == "-DOWNLOAD-":
        download_dir = values["-DIR-"]
        dataset_names = values["-DATASET LIST-"]
        if not os.path.isdir(download_dir):
            sg.popup_error("Invalid Directory")
        elif not dataset_names:
            sg.popup_error("Please select one model or more")
        else:
            window.perform_long_operation(
                lambda: download_datasets(dataset_names), "-DOWNLOAD DATASETS-"
            )

    elif event == "OK" or event == sg.WIN_CLOSED:
        break
