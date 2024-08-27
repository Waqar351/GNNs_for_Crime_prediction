import os
import pathlib
import zipfile

import gdown

output_csv_path = pathlib.Path("data/raw/sp")
output_csv_path.mkdir(parents=True, exist_ok=True)

dataset_ids = {
    # "crimes.h5": "1dvCtaWlciKtUeiVHmvtmyZR2Kwk5J1AQ",
    "crimes.h5": "1iQeOTMdJVbTIa9PlvINM4dhH4sSuL-Ak",
    "limites_administrativos.shp": "1iNY4qIQODyTJg6bE9g19qe57CWt9cdZD",
    "limites_administrativos.shx": "1C5Ra-OaCiFrjzAGTzvCS7xQbpqjE4_Av",
    "geosampa.zip": "1LICFIi9S0e8dQ14TQBi4xzBMTJ2Q-f-9",
    "weather_data.zip": "1brn_1tUJDwDqO5Z38WgJv5D0H_WMG7xU",
}

for name, dataset_id in dataset_ids.items():
    output_file = str(output_csv_path) + "/" + name
    gdown.download(id=dataset_id, output=output_file)
    if output_file[-3:] == "zip":
        with zipfile.ZipFile(output_file, "r") as zip_ref:
            zip_ref.extractall(output_csv_path)
        os.remove(output_file)
