# %%
# Import libraries
import pandas as pd
import numpy as np
import os
import time
import pickle as pkl
from pathlib import Path
np.set_printoptions(precision=2)

tstart = time.time()                                                                   # start time: the time now (recorded in computer)
##############################################
##############################################
#TASK 1: Load Exiobase 3, monetary IOT, product by product

##############################################
# Create folder path 1) Change the current working directory to reflect the location in your computer
# 2) Run os.getcwd() to find out what that is, and 3) fill in the path to 'Path'

# Folder to store the MRIO data for python
os.getcwd()
mrio_dir = Path("C:\\Users\\lik6\\OneDrive - Universiteit Leiden\\4. Leiden Univ\\2021-WN EIOA course by Ranran\\IGA\\IOT_2011_pxp")                                                   # Fill " " with your working directory path 

# Exiobase folder
exio_dir = Path("C:\\Users\\lik6\\OneDrive - Universiteit Leiden\\4. Leiden Univ\\2021-WN EIOA course by Ranran\\IGA\\IOT_2011_pxp")
 
# Population file folder
pop_dir = Path("C:\\Users\\lik6\\OneDrive - Universiteit Leiden\\4. Leiden Univ\\2021-WN EIOA course by Ranran\\IGA\\Other data")

##############################################
# Load labels (final demand, product, region, extensions, etc.)

# list of final demand
labels_final_str = exio_dir.joinpath('finaldemands.txt')                                 # the entire path to locate the finaldemand.txt file
labels_final_pd = pd.read_csv(labels_final_str, sep='\t', index_col=[3], header=[0])     # read the labels into DataFrame
labels_final_pd = labels_final_pd.drop(labels=['Number'],axis=1)                         # remove the 'Number' column 
n_final = labels_final_pd.count()[0]                                                     # number of final demand components


# list of products
labels_prod_str = exio_dir.joinpath('products.txt')  
labels_prod_pd = pd.read_csv(labels_prod_str, sep='\t', index_col=[3], header=[0])
labels_prod_pd = labels_prod_pd.drop(labels=['Number'],axis=1)
n_prod = labels_prod_pd.count()[0]

# region list of categories including population
labels_reg_str = pop_dir.joinpath('exiobase_2011_population.csv')  
labels_reg_pd = pd.read_csv(labels_reg_str, sep=',', index_col=[0], header=[0])
labels_reg_pd = labels_reg_pd.iloc[:, list(range(0, 4))]
n_reg = labels_reg_pd.count()[0]

# unit of monetary flows
unit_monetary = 'M.EUR'

# list of 'environmental' extensions
labels_ext_str = exio_dir.joinpath('satellite/unit.txt')
labels_ext_pd = pd.read_csv(labels_ext_str, sep='\t')
labels_ext_pd.columns.values[0] = 'Name'                                                 # the column name was missing  

# list of value added components 
pos_va = [0,1,2,3,4,5,6,7,8]                                                             # position of value added in extensions
labels_va_pd = labels_ext_pd.iloc[pos_va]
labels_co2_pd = labels_ext_pd[labels_ext_pd.Name.str.contains('CO2', case=False)]        # filter by extention Name, including 'CO2'
pos_co2 = labels_co2_pd.index                                                            # position of the CO2-related extension indicators


##############################################
#Load values

# final demand matrix
Y_str = exio_dir.joinpath('Y.txt')
Y_pd = pd.read_csv(Y_str, sep='\t', index_col=[0,1], header=[0,1])
Y = np.array(Y_pd)

# extensions (including household direct 'emissions')
H_str = exio_dir.joinpath('satellite/F_hh.txt')
H_pd = pd.read_csv(H_str, sep='\t', index_col=[0], header=[0,1])
H = np.array(H_pd)                                                                         # household domestic emissions 1104*343

H_co2 = H[pos_co2,:]
H_co2 = np.sum(H_co2,0)                                                                    # obtain a CO2 extension vector for 343 


F_str = exio_dir.joinpath('satellite/F.txt')
F_pd = pd.read_csv(F_str, sep='\t', index_col=[0], header=[0,1])
F = np.array(F_pd)

F_co2 = F[pos_co2,:]
F_co2 = np.sum(F_co2,0)                                                                    # obtain a CO2 extension vector 9800 cols

F_va = F[pos_va,:]   
F_va = np.sum(F_va,0)                                                                      # obtain a VA extension vector                                                         

tend = time.time()                                                                         # end time
print('Done reading everything except intersectoral flows in %5.2f s\n'% (tend - tstart))  # time passed (since the last 'start' time)


tstart = time.time()                                                                       # start time

A_str = exio_dir.joinpath('A.txt')
A = np.array(pd.read_csv(A_str, sep='\t', index_col=[0,1], header=[0,1]))

tend = time.time()                                                                         # end time
print('Done reading intersectoral flows in %5.2f s\n'% (tend - tstart))

tstart = time.time()                                                                       # start time

L = np.linalg.inv(np.eye(n_reg*n_prod) - A)    

tend = time.time()                                                                         # end time
print('Done calculating Leontief inverse in %5.2f s\n'% (tend - tstart))




#############################################
#############################################
#TASK 2 Store in dictionary and as pickle

tstart = time.time()

# store all labels and MRIO variables in one dictionary: 
label = {'region': labels_reg_pd, 'product': labels_prod_pd, 'final': labels_final_pd, 'va': labels_va_pd, 'co2': labels_co2_pd}
mrio = {'F': F, 'H': H, 'V': F_va, 'Y': Y, 'L': L, 'F_co2': F_co2, 'H_co2': H_co2, 'label': label}

# %%
#############################################
# save to pickle
mrio_name = 'mrio.pkl'    
mrio_str = os.path.abspath('mrio.pkl')                                                    # .pkl: a Python file type
pkl_out = open(mrio_str,"wb")                                                              # 'open' the created mrio.pkl file
pkl.dump(mrio, pkl_out)                                                                    # 'dump' the mrio data to mrio.pkl
pkl_out.close()

tend = time.time()
print('Done writing in %5.2f s\n'% (tend - tstart))

