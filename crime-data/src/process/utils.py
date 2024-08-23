import os

import pandas as pd


def process_station_data(station_name, crime_data_path):
    NUM_HEADER_LINES = 8  # Number of lines considered as header

    """
        Function designed to process weather data from a station.
        Search into the directory to find the respective files and process it.
        In the header of the csv (first 8 lines) are some metadata. Example:
            'REGIAO:;SE\n',                      # Station's region
             'UF:;SP\n',                         # Station's UF
             'ESTACAO:;SAO PAULO - MIRANTE\n',   # Station's name (respective city)
             'CODIGO (WMO):;A701\n',             # Station's code
             'LATITUDE:;-23,49638888\n',         # Station's Latitude
             'LONGITUDE:;-46,61999999\n',        # Station's Longitude
             'ALTITUDE:;785,64\n',               # Station's Altitude
             'DATA DE FUNDACAO:;25/07/06\n'      # Station's foundation date
    """

    # Search for file names
    station_files_name = [f for f in os.listdir(crime_data_path / "weather_data") if station_name in f.lower()]

    # Extract the metadata and data from each file
    data = []

    for fn in station_files_name:
        # Open the file
        with open(crime_data_path / ("weather_data/" + fn), encoding="iso-8859-1") as f:
            for i in range(NUM_HEADER_LINES):
                line_information = f.readline()
                if "latitude" in line_information.lower():
                    latitude = float(line_information.split(";")[1].replace("\n", "").replace(",", "."))
                elif "longitude" in line_information.lower():
                    longitude = float(line_information.split(";")[1].replace("\n", "").replace(",", "."))

        # Importing csv table
        csv_table = pd.read_csv(
            crime_data_path / ("weather_data/" + fn), skiprows=[0, 1, 2, 3, 4, 5, 6, 7], encoding="iso-8859-1", sep=";"
        )

        # Pre-processing
        csv_table["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"] = (
            csv_table["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"].str.replace(",", ".").astype("float64")
        )
        csv_table["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"] = (
            csv_table["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"].str.replace(",", ".").astype("float64")
        )
        csv_table["VENTO, RAJADA MAXIMA (m/s)"] = (
            csv_table["VENTO, RAJADA MAXIMA (m/s)"].str.replace(",", ".").astype("float64")
        )
        csv_table["VENTO, VELOCIDADE HORARIA (m/s)"] = (
            csv_table["VENTO, VELOCIDADE HORARIA (m/s)"].str.replace(",", ".").astype("float64")
        )

        # Aggregation by day and calculating features
        csv_table = csv_table.groupby(["Data"], as_index=False).agg(
            precipitacao__sum=("PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "sum"),
            precipitacao__mean=("PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "mean"),
            precipitacao__max=("PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "max"),
            precipitacao__min=("PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "min"),
            precipitacao__std=("PRECIPITAÇÃO TOTAL, HORÁRIO (mm)", "std"),
            temperatura__mean=("TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", "mean"),
            temperatura__max=("TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", "max"),
            temperatura__min=("TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", "min"),
            temperatura__std=("TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)", "std"),
            umidade_relativa__mean=("UMIDADE RELATIVA DO AR, HORARIA (%)", "mean"),
            umidade_relativa__max=("UMIDADE RELATIVA DO AR, HORARIA (%)", "max"),
            umidade_relativa__min=("UMIDADE RELATIVA DO AR, HORARIA (%)", "min"),
            umidade_relativa__std=("UMIDADE RELATIVA DO AR, HORARIA (%)", "std"),
            vento_rajada__mean=("VENTO, RAJADA MAXIMA (m/s)", "mean"),
            vento_rajada__max=("VENTO, RAJADA MAXIMA (m/s)", "max"),
            vento_rajada__min=("VENTO, RAJADA MAXIMA (m/s)", "min"),
            vento_rajada__std=("VENTO, RAJADA MAXIMA (m/s)", "std"),
            vento_velocidade__mean=("VENTO, VELOCIDADE HORARIA (m/s)", "mean"),
            vento_velocidade__max=("VENTO, VELOCIDADE HORARIA (m/s)", "max"),
            vento_velocidade__min=("VENTO, VELOCIDADE HORARIA (m/s)", "min"),
            vento_velocidade__std=("VENTO, VELOCIDADE HORARIA (m/s)", "std"),
        )

        # Transform date type
        csv_table["Data"] = pd.to_datetime(csv_table["Data"])

        # Column name to lowercase
        csv_table.columns = [c.lower() for c in csv_table.columns]
        # Appending to list
        data.append(csv_table)

    return data, (latitude, longitude)
