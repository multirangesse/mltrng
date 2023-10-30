<p align="center">
  <a href="" rel="noopener">
 <img width=400px height=200px src="docs\Public_key_encryption_keys.svg.png" alt="Project logo"></a>
</p>

<h1 align="center">Organizing Records for Retrieval in Multi-Dimensional Range Searchable Encryption</h1>


## 📝 Table of Contents

- [About](#about)
- [Dependencies](#dependencies)
- [Command Line Arguments](#Command_Line_Arguments)


##  About <a name = "about"></a>
This project has the capability to encrypt multidimensional scientific data.

## 🏁 Dependencies <a name = "dependencies"></a>

In order to run the code, you must have Python 3.10.12 or newer versions installed. You can find the installation source [here][def].<br />
The `req.txt` file located in the main directory lists all the required dependencies for executing the project. You can install these dependencies by using the command `pip install -r req.txt`.

## 🏁 Command Line Arguments <a name = "Command_Line_Arguments"></a>

The following command line arguments can be used in the script:

- `-dg` or `--datagen`:
   - Default value: None
   - Choices: ['txt', 'hdf5', 'kdtree']
   - Description: Specifies the type of data generation. Choose from 'txt', 'hdf5', or 'kdtree'.

- `-en` or `--encrypt`:
   - Action: BooleanOptionalAction
   - Description: Enables encryption. Use this flag to enable encryption.

- `-kd` or `--kdisk`:
   - Action: BooleanOptionalAction
   - Description: Enables kdtree disk-based indexing. Use this flag to enable k-disk indexing.

- `-sh` or `--shuffle`:
   - Default value: None
   - Choices: ['sh', 'sh0', 'sh1', 'sh2']
   - Description: Specifies the shuffling method. Choose from 'sh', 'sh0', 'sh1', or 'sh2'.

- `-re` or `--report`:
   - Action: BooleanOptionalAction
   - Description: Enables reporting. Use this flag to enable reporting.

- `-te` or `--test`:
   - Action: BooleanOptionalAction
   - Description: Enables testing. Use this flag to enable testing.

- `-dd` or `--datadim`:
   - Type: int
   - Number of arguments: 5
   - Required: False
   - Default value: [4, 10, 10, 10, 10]
   - Description: Specifies the dimensions of the data. Provide five integers as arguments.

- `-qu` or `--query`:
   - Default value: None
   - Choices: ['txt', 'hdf5', 'kdtree']
   - Description: Specifies the type of query. Choose from 'txt', 'hdf5', or 'kdtree'.

- `-rq` or `--rangequery`:
   - Type: int
   - Number of arguments: 8
   - Required: False
   - Default value: [0, 5, 1, 6, 0, 2, 0, 10]
   - Description: Specifies the range query parameters. Provide eight integers as arguments.

These command line arguments provide options and functionality for the Python script.



[def]: https://www.python.org/downloads/source/
