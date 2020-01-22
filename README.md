Data and scripts for 3DCNN-based land cover classification

==============================================

Original data source: Raw LiDAR data of Williamson county, IL can be downloaded from Illinois Geospatial Clearning house: http://clearinghouse.isgs.illinois.edu/data/elevation/illinois-height-modernization-ilhmp-lidar-data

Classes: agriculture, developed, wetlands, forest, herbaceous, shrub, water

Data format: .tar of .mat files
Contents:  
- landscapetest_singleaug.tar/landscape_test_singleaugI.tar: 2489 occupancy and intensity grid testing samples randomly selected for accuracy assessment

- landscape_train_singleaug.tar/landscape_train_singleaugI.tar: 6453 occupancy and intensity grid training samples randomly selected for model training

Environment: Python2.7
Library dependencies: cuda7.0, pdal 1.2, sklearn 0.18.1, Voxnet 1.0 (https://github.com/dimatura/voxnet)


Steps:
If you want to directly use the preprocessed data, please skip steps 1-3.  

1. Download the raw dataset from the link above and save to your working directory 

2. Run las2mattotal.py for LiDAR samples extraction with Pdal functions.

3. Created input of zipped .mat files by running convert_shapenet10augall.py.

4. Train the model by running bash scripts train_norepaugI.sh and test the model by running testoutfeaturesI.sh, please change the path variable to your work folder accordingly.

5. Run classificiton.py to get testing results and accuracy evaluation result

Folder structure:

----------------
3DCNN
    
- data: Training and testing samples of both intensity and occupancy grids selected from Williamson county and saved as tar files. Each sample has a resolution of 30m.
      
- landscapetest_singleaug.tar/landscape_test_singleaugI.tar: 2489 occupancy and intensity grid testing samples randomly selected for accuracy assessment
      
- landscape_train_singleaug.tar/landscape_train_singleaugI.tar: 6453 occupancy and intensity grid training samples randomly selected for model training    
      
- feature128te/tr:output 128 features for testing and training when combined with imagery dataset
      
- testlabelaug.npy/trainlabelaug.npy: True label of testing and training dataset with 9 rotations
      
- testlabel.npy: True label of testing data without rotation
      
- trainlidarspectral/testlidarspectral:testing and training data of 128 features of lidar with 72 bands of imagery(12 images each with 6 bands)
      
- out_test_I_prediction.npz: 128 features 
      - weights.npz: weights learned and used for prediction
    
- scripts: scripts for training and testing the 3DCNN model for land cover classificaiton.
     
 -classificaiton.py: run svm classification and provide accuracy assessment 
      
-config/shapenet10.py: store parameter dictionary
     
-testoutfeaturesI.py: testing script
      
-train_norepaugI.py: training script
      
-testoutfeaturesI.sh: testing bash script
      
-train_norepaugI.sh: training bash script

Publication:
----------------
Xu, Z., Guan, K., Casler, N., Peng, B., & Wang, S. (2018). A 3D convolutional neural network method for land cover classification using LiDAR and multi-temporal Landsat imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 144, 423-434.
