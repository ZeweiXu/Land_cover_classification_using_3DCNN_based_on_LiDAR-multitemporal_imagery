Data and scripts for the scalable deep learning framework for land cover classification
==============================================
Original data source: Raw LiDAR data from 34 counties of Illinois, can be downloaded from Illinois Geospatial Clearning house: http://clearinghouse.isgs.illinois.edu/data/elevation/illinois-height-modernization-ilhmp-lidar-data

Preprocessed data: extracted .npy files of .las, where the x,y,z, and intensity values are extracted

Reference data: 93,250 training, validation, and testing data generated from visual interpretation

Classes: agriculture, developed, wetlands, forest, herbaceous, shrub,water

Environment: Python2.7

Library dependencies: cuda7.0, pdal 1.2, sklearn 0.18.1, Voxnet 1.0 (https://github.com/dimatura/voxnet)

Steps:

1. Run LiDAR_preprocessing.sh with the command below to generate model training, validation, and testing data.

Example command:

	> bash preprocessing.sh $1 $2 $3

$1 numer 0 - 99 corresponds to the index of the divided 100 portions of all the reference dataset
$2 'I' or 'n'  
$3 'train','trainnoshuffle','test' or 'validation' represents the data set that will be genreated by the script

2. Run train.py to train the model

Example command:

	> bash train.bash

3. Run test.py to get accuracy evaluation of the prediction results

Example command:

	> python test.py $1

$1 The GPU ID on the machine


Folder structure:
----------------
3DCNN

- data: Training and testing samples of both intensity and occupancy grids selected from the 34-county area and saved as .tar files. Each sample has a resolution of 30m.
    |-- totallas.npy: dictionary of all lasfile and its bounding box coordinates
    |-- scale_up_newreference_locations.npy: contains coordinates of all streamline in references dataset
    |-- testN.tar/testI.tar: occupancy and intensity grid testing samples randomly selected for accuracy assessment
    |-- trainN.tar/trainI.tar: occupancy and intensity grid training samples randomly selected for model training    
    |-- validationN.tar/validationI.tar: occupancy and intensity grid training samples randomly selected for model validation
    |-- training_data/validation_data/testing_data: combined LiDAR features with multitemporal spectral bands for the final SVM classification
    |-- testing_label9.npy/validation_label.npy/training_label.npy: True label of testing,validation, and training dataset with 9 rotations
    |-- testing_label.npy: True label of testing data without rotation
    
- scripts: scripts for preprocessing, training, and testing the deep learning framework for land cover classificaiton.
    |--preprocessing.py: preprocessing python script
    |--preprocessing.sh: preprocessing bash script to submit the job to batch system
    |--npytar.py: required library for the preprocessing script
    |--npytar.pyc: the compiled bytecode of npytar.py
    |--train.py: model training for LiDAR feature extraction
    |--configaug/shapenet10.py: parameters
    |--test.py: accuracy evaluation of the generated land cover map
 
Publication:
----------------
Xu, Z., Guan, K., Casler, N., Peng, B., & Wang, S. (2018). A 3D convolutional neural network method for land cover classification using LiDAR and multi-temporal Landsat imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 144, 423-434.



