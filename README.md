<p align="center">
  <a href="" rel="noopener">
 <img width=400px height=200px src="docs\Public_key_encryption_keys.svg.png" alt="Project logo"></a>
</p>

<h1 align="center">Organizing Records for Retrieval in Multi-Dimensional Range Searchable Encryption</h1>


## 📝 Table of Contents

- [About](#about)
- [Dependencies](#dependencies)
- [Command Line Arguments](#experiments)


##  About <a name = "about"></a>
This project has the capability to encrypt multidimensional scientific data.

## 🏁 Dependencies <a name = "dependencies"></a>

In order to run the code, you must have Python 3.10.12 or newer versions installed. You can find the installation source [here][def].<br />
The `req.txt` file located in the main directory lists all the required dependencies for executing the project. You can install these dependencies by using the command `pip install -r req.txt`.

## 🏁 Running the Experiments <a name = "experiments"></a>


### Configurations
You can change the configurations for the experiments in each individual experiment file based on the following parameters:
    sample_num = 1000
    datadim = [Number of dimensions,D1_size, D2_size, D3_size, D4_size]
    slabdim = [4, 4, 4, 4]--> For slab-based shuffling
    slab_size = 256 --> For row-based shuffling
The first element in datadim is the number of dimensions, and the following elements are the dimensions' sizes. 
sample_num is the number of sample random queries of the query shape.

### Isotropic queries
Run the command python3 src/isoqreport.py
To change the configuration of the dataset and slab size: change the following configurations in isoqreport.py:

### Bisected Anisotropic queries
Run the command python3 src/baqreport.py
To change the configuration of the dataset and slab size, change the configurations in isoqreport.py:

### Gradual Anisotropic queries
Run the command python3 src/gaqreport.py
To change the configuration of the dataset and slab size, change the configurations in isoqreport.py:

### Outlier queries
Run the command python3 src/oaqreport.py
To change the configuration of the dataset and slab size, change the following configurations in isoqreport.py:





[def]: https://www.python.org/downloads/source/
