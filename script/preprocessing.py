# Import all required libs
from random import shuffle
import scipy.io as sio
from path import Path
import numpy as np
import argparse
import logging
import random
import npytar
import laspy
import glob
import math
import pdb
import sys
import os

sample_ind = int(sys.argv[1])
Iorn = sys.argv[2]
all_val = sys.argv[3]

##########################################################################################
###################### STEP 1 Build the tree structure of the LAS patches#################
##########################################################################################
# The list below shows number of LAS patches of each county                              #
# alexander223;clark3864;clay3628;coles3994;crawford3512;cumberland2734;dewitt3098;      #
# douglas3183;effingham3741;fayette5458;gallatin397;hardin134;jasper3828;johnson2006;    #
# lawrence2901;massac1848;moultrie2800;pulaski1625;richland3085;saline482;union319;      #
# vermilion6790;wabash1963;wayne5271;white657;edwards1826;macon4374;pope3316;shelby5787; #
# edgar4684;hamilton567;jefferson632;piatt4074;williamson569                             #
##########################################################################################
print('Start step 1')
totallas = '/data/cigi/common/zeweixu2/land_cover_scale_up/data/totallas.npy'

#Array of the starting patch number of each
cosum = [   0,   223,  4087,  7715, 11709, 15221, 17955, 21053, 24236,
       27977, 33435, 33832, 33966, 37794, 39800, 42701, 44549, 47349,
       48974, 52059, 52541, 52860, 59650, 61613, 66884, 67541, 69367,
       73741, 77057, 82844, 87528, 88095, 88727, 92801, 93370]

class Rtree:
    def __init__(self,bbox,size):
        self.bbox = bbox
        self.size = size
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    def query(self,qbbox):
        def ifoverlap(node,qbbox):
            o = node.bbox
            if not (o[2]<qbbox[0] or o[0]>qbbox[2] or o[3]<qbbox[1] or o[1]>qbbox[3]):
                return True
            return False

        re = []
        for i in range(self.size):
            if ifoverlap(self.children[i],qbbox):
                node = self.children[i]
                for j in range(node.size):
                    if ifoverlap(node.children[j],qbbox):
                        re.append(cosum[i]+j)
        return re

def construct():
    cc = Rtree([275874.41,4092256.97,459324.41,4483726.97],34)
    total = np.load(totallas)
    for i in range(34):
        sib = [min(total[cosum[i]:cosum[i+1]][:,0]),min(total[cosum[i]:cosum[i+1]][:,1]),max(total[cosum[i]:cosum[i+1]][:,2]),max(total[cosum[i]:cosum[i+1]][:,3])]
        cc.add_child(Rtree(sib,cosum[i+1]-cosum[i]))
        for j in range(cosum[i],cosum[i+1]):
            cc.children[i].add_child(Rtree(total[j],0))
    tree = cc
    return tree
#Build the actual tree
total_LAS_Rtree = construct()

#query example: all patches the overlap with the input bounding box [0,0,0,0]
#result = total_tree.query([0,0,0,0])

print('End step 1')
##########################################################################################
############################### STEP 2 Extract samples ###################################
##########################################################################################
print('Start step 2')

matpath = '/data/cigi/scratch/nattapon/'
total_dic = '/data/cigi/scratch/Nattapon/scaleup/reference/totallas.npy'
las_file_path = '/data/cigi/scratch/nattapon/scale_up_renamed_lasnpy/'
sample_file = '/data/cigi/scratch/Nattapon/scaleup/reference/scale_up_newreference_locations.npy'

def sample_rotation(data, angle, bbox):
    #print('sample_rotation')
    """
    Functions takes in 3 arguments:
        data: 2D array of points in the form [[x_0, y_0, z_0], [x_1, y_1, z_1], [x_2, y_2, z_2], ...]
        angle: In the form of degrees
        bbox: Min and max values of x, y in the form [x_min, y_min, x_max, y_max]
        
    Function returns data input rotated around center of bbox
    """
    #ensure the data are all in type 'float'
    data_in = data[:,-1]
    data_float = data[:,:3]
    #calculate shift values of data to shift to new origin
    xshift = (bbox[0] + bbox[2]) / 2
    yshift = (bbox[1] + bbox[3]) / 2
    #shift x, y values to new origin
    shift = np.tile([xshift, yshift, 0], (len(data_float), 1)).astype(float)
    data_float = data_float - shift
    #transpose to be able to use a rotation matrix
    data_transpose = np.ndarray.transpose(data_float)
    #create rotation matrix
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s, 0], [s, c, 0], [0, 0, 1]])
    #multiply matrices
    #rotated = np.matmul(R, data_transpose).astype(float)
    rotated = np.dot(R, data_transpose).astype(float)
    #transpose back to original form and shift back to original origin
    data_float = np.ndarray.transpose(rotated) + shift    
    r_data = np.concatenate((data_float, data_in.reshape(-1,1)),axis = 1)
    return r_data

def voxelization(data,sample,sample_ind,i,label,matpath,voxel_ind):

    ### output path matpath/I or n/1 or 2 or 3 or 4 or 5 or 6 or 7/    
    outmatI = matpath +'I/'+str(label)+'/'+str(sample_ind)+'_'+str(i)+'_'+str(int(label))+'.mat'
    outmat = matpath +'n/'+str(label)+'/'+str(sample_ind)+'_'+str(i)+'_'+str(int(label))+'.mat'
    pfi = str(sample_ind)+'_'+str(i)+'_'+str(int(label))
    zd = 50
    D3array=np.zeros((30,30,zd))
    I3array=np.zeros((30,30,zd))
    if voxel_ind != 0:
        xv=data[:,0].astype(float).reshape(-1,1)
        yv=data[:,1].astype(float).reshape(-1,1)
        zv=data[:,2].astype(float).reshape(-1,1)
        intensity=data[:,3].astype(float).reshape(-1,1)
        if zv.size:
            zv=zv-min(zv)
            if max(zv)<zd:
                for j in range(len(xv)):
                    vx=int(math.floor(xv[j]-sample[0]))
                    vy=int(math.floor(yv[j]-sample[1]))
                    vz=int(math.floor(zv[j]))
                    if vx==30:
                        vx=29
                    if vy==30:
                        vy=29
                    if vz==zd:
                        vz=zd-1
                    D3array[vx,vy,vz]=1
                    I3array[vx,vy,vz]=I3array[vx,vy,vz]+intensity[j]
                I3array[vx,vy,vz]=I3array[vx,vy,vz]/float(len(xv))
            else:
                nzv=zv*(zd-0.0001)/(max(zv))
                for j in range(len(xv)):
                    vx=int(math.floor(xv[j]-sample[0]))
                    vy=int(math.floor(yv[j]-sample[1]))
                    vz=int(math.floor(nzv[j]))
                    if vx==30:
                        vx=29
                    if vy==30:
                        vy=29
                    if vz==zd:
                        vz=zd-1
                    D3array[vx,vy,vz]=1
                    I3array[vx,vy,vz]=I3array[vx,vy,vz]+intensity[j]
                I3array[vx,vy,vz]=I3array[vx,vy,vz]/float(len(xv))
            sio.savemat(outmat, mdict={pfi: D3array})
            sio.savemat(outmatI, mdict={pfi: I3array})
	    #print('saved')
        else:
            sio.savemat(outmat, mdict={pfi: D3array})
            sio.savemat(outmatI, mdict={pfi: I3array})
    else:
        sio.savemat(outmat, mdict={pfi: D3array})
        sio.savemat(outmatI, mdict={pfi: I3array})

def query(total_dic,las_file_path,sample_file,sample_ind):


    sample_boundaries = np.load(sample_file)

    #read the sample labels
    label = int(float(sample_boundaries[sample_ind][-1]))
    #read the sample points
    sample = sample_boundaries[sample_ind][:4]
    #create sample patches
    samplen = sample + [-7.5,-7.5,7.5,7.5]
	#query all LAS file that intersect with the sample patch
    C =  total_LAS_Rtree.query(samplen)

    try:
        for i in range(len(C)):
            las_file = las_file_path+str(C[i])+".npy"	
            # inFile = laspy.file.File(las_file, mode = "r")	
            inFile = np.load(las_file)
  	        #### Get arrays of points which indicate valid X, Y values.
            X_valid = np.logical_and((samplen[0] < inFile[:,0]),(samplen[2] > inFile[:,0]))
            Y_valid = np.logical_and((samplen[1] < inFile[:,1]),(samplen[3] > inFile[:,1]))
            valid_indices = np.where(np.logical_and(X_valid, Y_valid))[0]	
            parray = []
            count = -1
            if len(valid_indices) != 0:
                count += 1
                temp = np.concatenate((inFile[:,0][valid_indices].reshape(-1,1),inFile[:,1][valid_indices].reshape(-1,1),inFile[:,2][valid_indices].reshape(-1,1),inFile[:,3][valid_indices].reshape(-1,1)),axis = 1)
                if count == 0:
                    parray = temp
                else:
                    parray = np.concatenate((parray,temp),axis = 0)    
        if len(parray) != 0:
            voxel_ind = 1
            rotated_sample = [parray]
            for i in range(1,9):
                degree = 360/9*i
                rotated_sample.append(sample_rotation(parray,degree,samplen))
            # crop real sample boundaries
            for i in range(len(rotated_sample)):
                ssa = rotated_sample[i]
                X_valid = np.logical_and((sample[0] < ssa[:,0]),(sample[2] > ssa[:,0]))
                Y_valid = np.logical_and((sample[1] < ssa[:,1]),(sample[3] > ssa[:,1]))
                valid_indices = np.where(np.logical_and(X_valid, Y_valid))[0]
                rsc = ssa[valid_indices]        
                voxelization(rsc,sample,sample_ind,i,label,matpath,voxel_ind)
        else:
            voxel_ind = 0
            for i in range(9):
                voxelization([],[],sample_ind,i,label,matpath,voxel_ind)
    except:
        #print(sample_ind)
	True

# sample_index ranges from 0 - 99 corresponds to the index of the divided 100 portions of all the reference dataset

#for sample_index in range(100):
sample_index = sample_ind
total_sample = len(np.load(sample_file))
unit = int(total_sample/100)
#find starting index
start = unit*sample_index
#find ending index
if sample_index == 99:
    end = total_sample
else:
    end = start + unit
#generate actual sample patches
for i in range(start,end):
    query(total_dic,las_file_path,sample_file,i)

print('End step 2')
##########################################################################################
######## STEP 3 split generated samples into training, validatiaion, and testing #########
##########################################################################################
print('Start step 3')

os.chdir('/data/cigi/scratch/nattapon/')
pat = '/data/cigi/scratch/nattapon/'
pat2 = '/data/cigi/scratch/nattapon/'

# classes of samples
classes = ['water','developed','barren','forest','shrub','herbaceous','agriculture','wetlands']
# reshuffle and create .txt order files
na = ['matI','matn']
# array that define the amount of training, validatiaion, and testing samples of each class
toarray = np.array([
    [    0,   965,  1126,  1609],
    [    0,  2853,  3328,  4754],
    [    0,   172,   201,   287],
    [    0, 10185, 11883, 16976],
    [    0,  2474,  2886,  4123],
    [    0,  3481,  4061,  5801],
    [    0, 34658, 40435, 57765],
    [    0,  1333,  1555,  2222]
    ])

for i,aClass in enumerate(classes):
    with open(aClass+'_order9.txt', 'r') as the_file:
        data = the_file.readlines()

    fi = [x.strip() for x in data]
    for ii in ['I','n']:
        count = 0

        for jj in range(toarray[i][0],toarray[i][1]):
            count += 1   
            oor=list('000000000')
            oor[-len(str(count)):]=list(str(count))
            oor="".join(oor)
            strr=aClass +'_'+oor
            for it in range(1,10):
                raw = pat+ii+'/'+str(i+1)+'/'+fi[jj]+'_'+str(it-1)+'_'+str(i+1)+'.mat'
                name = strr + '_' + str(it) + '.mat'
                os.system('cp '+raw+' '+pat2+'mat'+ii+'9/train/'+aClass+'/'+name)

        for jj in range(toarray[i][1],toarray[i][2]):
            count += 1   
            oor=list('000000000')
            oor[-len(str(count)):]=list(str(count))
            oor="".join(oor)
            strr=aClass +'_'+oor
            for it in range(1,10):
                raw = pat+ii+'/'+str(i+1)+'/'+fi[jj]+'_'+str(it-1)+'_'+str(i+1)+'.mat'
                name = strr + '_' + str(it) + '.mat'
                os.system('cp '+raw+' '+pat2+'mat'+ii+'9/validation/'+aClass+'/'+name)

        for jj in range(toarray[i][2],toarray[i][3]):
            count += 1   
            oor=list('000000000')
            oor[-len(str(count)):]=list(str(count))
            oor="".join(oor)
            strr=aClass +'_'+oor
            for it in range(1,10):
                raw = pat+ii+'/'+str(i+1)+'/'+fi[jj]+'_'+str(it-1)+'_'+str(i+1)+'.mat'
                name = strr + '_' + str(it) + '.mat'
                os.system('cp '+raw+' '+pat2+'mat'+ii+'9/test/'+aClass+'/'+name)

print('End step 3')
##########################################################################################
###################### STEP 4 Pack all the samples into tar files ########################
##########################################################################################
print('Start step 4')

def write(records, fname):
    writer = npytar.NpyTarWriter(fname)
    classlist=['water','developed','forest','shrub','herbaceous','agriculture','wetlands']
    for (classname, instance, rot, fname) in records:
        class_id = int(classlist.index(classname)+1)
        name = '{:03d}.{}.{:03d}'.format(class_id, instance, rot)
        try:
            arrt = sio.loadmat(fname)
        except:
            import pdb
            pdb.set_trace()
        arrt.pop('__version__', None)
        arrt.pop('__header__', None)
        arrt.pop('__globals__', None)
        artem=arrt.keys()
        arttem=arrt[artem[0]]
        arr=arttem.astype(np.uint8)
        arrpad = np.zeros((32,32,52), dtype=np.uint8)
        arrpad[1:-1,1:-1,1:-1] = arr
        writer.add(arrpad, name)
    writer.close()

# I_or_n = ['I','n']
# all_val = ['train','trainnoshuffle','test','validation']

# for ind in I_or_n:
#     for val in all_val:

ind = Iorn
val = all_val

base_dir = Path('/data/cigi/scratch/nattapon/mat'+ind+'9/').expand()
base_tar_dir = '/data/cigi/scratch/nattapon/tar/'
records = {'train': [], 'test': [], 'validation':[]}

logging.info('Loading .mat files')
countt=0
for fname in sorted(base_dir.walkfiles('*.mat')):
    countt=countt+1
    if fname.endswith('test_feature.mat') or fname.endswith('train_feature.mat'):
        continue
    elts = fname.splitall()
    instance_rot = Path(elts[-1]).stripext()
    instance = instance_rot[:instance_rot.rfind('_')]
    rot = int(instance_rot[instance_rot.rfind('_')+1:])
    split = elts[-3]
    classname = elts[-2].strip()
    records[split].append((classname, instance, rot, fname))

logging.info('Saving train npy tar file')
train_records = records['train']
if val == 'trainnoshuffle':
    train_records = sorted(train_records, key=lambda x: x[2])
    train_records = sorted(train_records, key=lambda x: x[1])
    write(train_records, base_tar_dir+'landscape_train_'+ind+'noshuffle9_original.tar')
elif val == 'train':
    random.shuffle(train_records)
    write(train_records, base_tar_dir+'landscape_train_'+ind+'9_original.tar')
elif val == 'validation':
    logging.info('Saving test npy tar file')
    validation_records = records['validation']
    validation_records = sorted(validation_records, key=lambda x: x[2])
    validation_records = sorted(validation_records, key=lambda x: x[1])
    write(validation_records, base_tar_dir+'landscape_validation_'+ind+'9_original.tar')
else:
    logging.info('Saving test npy tar file')
    test_records = records['test']
    test_records = sorted(test_records, key=lambda x: x[2])
    test_records = sorted(test_records, key=lambda x: x[1])
    write(test_records, base_tar_dir+'landscape_test_'+ind+'9_original.tar')

print('End step 4')
