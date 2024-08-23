# Crime Data

## Basic Usage

All python packages necessary to run our code are in the `pyproject.toml` file. We suggest the use of Poetry for package dependency management. `Python >= 3.10` is required.

### Using Poetry

To configure our project using Poetry, you just need to go to the root folder and run the following commands:

1. `poetry shell`
This will create a python env for the crime-data project.
2. `poetry update` This will install all necessary python packages for our project. **This command should be rerun frequently (similarly to git pull) to update the Poetry project if there are any changes.**

If you want to add a new python package, run `poetry add {package name}`. After you install a new package using Poetry, the new package name and its version is going to be add to the  `pyproject.toml` and `poetry.lock` files, so you will need to commit this changes to our repository. 

**Attention:** Every time you intend to run or commit a file, ensure that it's done within a terminal session with Poetry's shell activated. This means you should run `poetry shell` before executing any commands or making changes to the code. This process is akin to using the `conda activate` command.

## Data

### How to Download Sao Paulo (SP) Dataset

To download SP dataset, go to the root project directory and run the following script:

```
python src/download/download_sp_data.py
```

This will create a folder `data/raw/sp` in the root directory with the raw dataset.

### How to Process SP Dataset

To process SP dataset, go to the root project directory and run the following script:

```
python src/process/process_sp_data.py
```

This will create a folder `data/processed/sp` in the root directory with all processed data.

You can also run the process dataset with the following flags:

1. `--overwrite (bool)`: This will overwrite any existing processed file. Default is False.
2. `--max_distance (float)`: This will ignore crimes with greater distance to the closest street segment than `max_distance.` Default value is 50.
3. `--network_id (str)`: This will be used as the network_id type when downloading graphs from osmnx. The default value is `all_private`.
4. `--process_id (str)`: This will be used as the folder name containing processed data. If `None`, use datetime.now() as id. Default is None
5. `--region (str)`: Region to donwnload using Osmnx function get_from_place. Default is `None`.
6. `--polygon (str)`: Key name for entry in `src/process/polygons_coordinates.json` file containing polygon coordinates to use in osmns get_from_polygon. Default is `sp-centro`.

**Note:** The `--polygon` flag restricts the graph to nodes inside a polygon. New polygons can be add, you just have to add its coordiantes in the `src/process/polygons_coordinates.json` and then use its key name as the entry for the flag.

### Processed Data

As output of our processing script, we have the following files:

1. `crime_nodes_datarame.pickle`: It is a Pickle file containing a GeoDataFrame with crime information and its respective graph information.
2. `graph-crossing_as_nodes.pickle`: The pickle file contains a networkx graph in which `street crossings` are used as `nodes` and `street segments` as `edges.`
3. `graph-streets_as_nodes.pickle`: The pickle file contains a networkx graph in which `street segments` are used as `nodes` and `street crossing` as `edges.`
4. `parameters.json`: This `json` file contains information to reproduce the processed data, such as the flags used to process the data and the date it was processed.
5. `raw_graph.pickle`: Pickle file containing the raw networkx graph we got from osmnx.
6. `street_nodes_datagrame.pickle`: A pickle file containing a GeoDataFrame with street information, such as geospatial features and their respective graph information.

## Data Source

1. A good portion of the criminal data was taken from [Portal SSP](https://www.ssp.sp.gov.br/estatistica/consultas).

