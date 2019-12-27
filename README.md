# eg_clustering_code
This repo contains the code for clustering, postprocessing, and mapping the clusters of villages which we generated for EG. 

## Important files and workflow of code:

### Kmeans_IP_formulation.py: 
This file contains the classes and functions necessary to run the initial clustering algorithm. It implements a modified K-means algorithm in which the cluster assignment step is an embedded integer program solved using the PuLP package. The algorithm is run using the **Kmeans_IP_run.py** file. The results of this algorithm are stored in the Excel files in the folder Excel_results/Kmean_IP. In addition, some graphs to help diagnose whether the algorithm is converging properly are saved to the Kmeans_functionality_figures folder.

### Systematic_postprocessing_formulation.py: 
This file contains the functions necessary to run a series of sequential postprocessing algorithms on the K-means/IP output and generate summary statistics of the clusters at each step. The postprocessing algorithms are run using the **Systematic_postprocessing_run.py** file. The results of these algorithms are stored in the Postprocessing and Renumbered subfolders of the Excel_results folder. Some basic summary statistics are printed to the console, while more detailed summary statistics are printed to each excel file which is saved in the excel_results folder. In addition, basic histograms are generated for each step of the postprocessing and saved in the stats_figures folder. *Note: The postprocessing_small_if_few() function in systematic_postprocessing_formulation.py should be modified to not exclude cluster 4 if this algorithm is run on new data.*

### Colored_by_block.ipynb: 
This jupyter notebook uses the gmaps python package to create an interactive Google Map with the clustering results plotted on top of it. Using the ipywidgets infrastructure, notebook exports the map as a .html file that is saved in maps/html_files. A legend of block colors and names is exported to maps/legends. 

**Running the notebook**: From terminal, cd into the EG_clustering_code folder of the local github repository and copy the following commands to open the notebook on a local server:

```
cd maps/maps_code
jupyter notebook
```

**The exported map has the following elements:**
- [ ] Village markers sized in proportion to the number of predicted OOSC. Purple markers represent unclustered villages and navy markers represent clustered villages we recommend EG expand into. Markers can be clicked to display a textbox with the village name, block name, and cluster ID.
- [ ] Polygons representing the outer border of clusters. Polygons are colored such that clusters in the same block are the same color.
	
**Important items to keep updated in Colored_by_block.ipynb:**
- [ ] **Dataframe being mapped:** This code is setup to currently map the final postprocessed results, but this can be changed by modifying the filepath indicated in the ipynb to load a different results dataframe from Excel_results. If you change the filepath also change the stage variable at the top of the notebook to reflect the postprocessing step you're mapping. This will modify the figure file name appropriately and allow the saving of multiple maps at once. If the same postprocessing stage is mapped multiple times, the map associated with that stage will be overwritten by the most recent map generated.
- [ ] **The districts being mapped:** The notebook prints two maps, one of Shivpuri district and another of the Khargone/Khandwa districts (combining the maps exceeds the capacity of the gmaps plotting package and creates html files that are very slow to load). To specify which map you'd like to generate, follow the instructions outlined at the top of the notebook.

**Important note on .html file size**: Each time the notebook is used to generate a .html map file, **restart the kernel**, otherwise the .html file size becomes excessively large.
	
*Alternative mapping option*: To run a map which is colored to help diagnose clustering deficiencies run the *Diagnostic_coloring.ipynb* file instead of the *Colored_by_block.ipynb*
