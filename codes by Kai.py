"""
Tasks:

    1. Load from pickle
    2. Calculate for a specific country territorial CO2 emissions
    3. Calculate for a specific country consumption-based CO2 emissions
    4. Identify top-5 sectors that contribute to territorial emissions of a specific country
    5. Identify top-5 final demand expenditure on products that contribute to consumption-based emissions of a specific country
    6. Identify top-5 countries where the specific country displaced its emissions to
"""

import numpy as np
import time
import pickle as pkl
import os
from pathlib import Path
np.set_printoptions(precision=2)

##############################################
# Load the pickle of Exiobase V3.8.1(2020)

os.getcwd()
mrio_dir = Path('OneDrive - Universiteit Leiden/4. Leiden Univ/2021-WN EIOA course by Ranran/IGA')  # Fill " " with your working directory path

tstart = time.time()
mrio_name = 'mrio_y2020.pkl'
mrio_str = mrio_dir.joinpath(mrio_name)
pkl_in = open(mrio_str,"rb")                                                        # 'open' the mrio.pkl file
mrio = pkl.load(pkl_in)                                                             # load the mrio.pkl file to mrio
pkl_in.close()

tend = time.time()
print('Done reading in %5.2f s\n'% (tend - tstart))
#%%
##############################################
# category counts

nr = mrio['label']['region'].count()[0]
ns = mrio['label']['product'].count()[0]
ny = mrio['label']['final'].count()[0]

##############################################
# MRIO matrix/vector variables
L = mrio['L']
Y = mrio['Y']
V = mrio['V']
F = mrio['F']
F_co2 = mrio['F_co2']
F_wc = mrio['F_wc']
H = mrio['H']
H_co2 = mrio['H_co2']
H_wc = mrio['H_wc']

##############################################
# calculate total output
x = L @ Y.sum(1)

##############################################
# calculate direct co2 emission intensity and direct water consumption intensity
f_co2 = F_co2/x                                                                   # resulting in na or inf when x = 0
f_co2 = np.nan_to_num(F_co2/x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)       # replace na and inf with 0

f_wc = F_wc/x                                                                     # resulting in na or inf when x = 0
f_wc = np.nan_to_num(f_wc, copy=False, nan=0.0, posinf=0.0, neginf=0.0)           # replace na and inf with 0

##############################################
# Configuration for all following tasks
# position 27 is UK

c = 27

# To locate country columns or rows in A and L: 'c*ns : (c+1)*ns'
# To locate country columns in Y: 'c*ny : (c+1)*ny'

# convert units from kg to Mt (*1e-9,co2) and Mm3 to (, water consumption)
conv = 1e-9

#%% Changes in final demand with the processed beef sector (PB)
##############################################
# final demand of the UK in 2020
y_uk = Y[:, c*ny : (c+1)*ny].sum(1)

# Assuming the p_a percentage of reduction of PB in UK's domestic final demand after the Brexit.
p_a = 0.1
kpb = 42
uk_red = p_a*y_uk[(ns*c+kpb)] 
y_uk[ns*c+kpb] = (1-p_a)*y_uk[ns*c+kpb]                                       # Specifying the postion of PB in the product list of the EXIOBASE
print("UK's domestic final demand reduction for PB:", uk_red, "M.EUR")

# Assuming the p_b(%) percentage of reduction of PB exports from EU27 to UK after the Brexit.
p_b = 0.3
y_AB_EU = 0

for i in range(27):
    
    y_AB_EU += (1-p_b)*y_UK[ns*i+kpb]

# Assuming the p_c(%) percentage of increase of PB exports from Australia to UK after the free tariff agreement.
p_c = 0.4

# Specifying the position of Australia in the country list.
AU = 37

y_AB_AU = (1+p_c)*y_UK[ns*AU+kpb]

print("UK's domestic fimal demand for PB:",np.around(y_AB_UK,3), "M.EUR")
print("PB exports from EU27 to UK:",np.around(y_AB_EU,3), "M.EUR")
print("PB exports from Australia to UK:",np.around(y_AB_AU,3), "M.EUR")

#%% Changes in co2 emissions and water used

label_s = list(mrio['label']['product']['Name']) + ['Household direct']
cba_hh = H_co2[c*ny : (c+1)*ny].sum() *conv
cba_s = f_co2 @ L @ np.diag(y)*conv
cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba_hh)                                                   # append household direct emissions to the end
cba_r = cba_rs.sum(1)                                                               # by emitting regions
cba_r[c] = cba_r[c] + cba_hh                                                     # add HH direct emissions

#%%



# TASK 2. Calculate territorial emissions for a specific country


# We want to report three cols: production-related, households and total, plus a global contribution percentage

pba = np.zeros((1,4))
pba[:,0] = F_co2[c*ns : (c+1)*ns].sum() *conv                                       # direct emissions by producers
pba[:,1] = H_co2[c*ny : (c+1)*ny].sum() *conv                                       # direct emissions by households
pba[:,2] = pba[:,0:2].sum()
pba[:,3] = pba[:,2]/((F_co2.sum()+H_co2.sum())*conv)*100


##############################################
##############################################
# TASK 3. Calculate for a specific country consumption-based CO2 emissions


# We want to report three cols: production-generated, households and total, plus a global contribution percentage

cba = np.zeros((1,4))
y = Y[:, c*ny : (c+1)*ny].sum(1)                                                    # aggregate final demand to one column

cba[:,0] = f_co2 @ L @ y *conv                                                      # footprint traced to production emissions
cba[:,1] = H_co2[c*ny : (c+1)*ny].sum()*conv                                        # footprint traced to household direct emissions
cba[:,2] = cba[:,:2].sum()
cba[:,3] = cba[:,2]/((F_co2.sum()+H_co2.sum())*conv)*100


##############################################
##############################################
# TASK 4. Identify top-5 sectors that contribute to production-based emissions

# pba_s: we want to report in rows each sector plus household

label_s = list(mrio['label']['product']['Name']) + ['Household direct']# forming a list of 201 rows
pba_s = F_co2[c*ns : (c+1)*ns] *conv
pba_s = np.append(pba_s, pba[0,1])                                                  # append household direct emissions to the end

# pba_s_sorted: we want to report in rows the 'sorted' (in descending order) sectoral plus household results;
# 3 cols: sectoral emissions, percent, and cumulative percent
pba_s_sorted = np.zeros((ns+1,3))
rank = np.argsort(pba_s)                                                            # returns indices that'd sort the array low to high
rank = np.flip(rank)                                                                # flip so it goes 'high to low'

pba_s_sorted[:,0] = np.sort(pba_s)                                                  # sort the values from 'low to high'
pba_s_sorted[:,0] = np.flip(pba_s_sorted[:,0])                                      # flip so the values go from high to low
pba_s_sorted[:,1] = pba_s_sorted[:,0]/pba[0,2]*100                                  # percentage contribution
pba_s_sorted[:,2] = np.cumsum(pba_s_sorted[:,1])                                    # cumulative percentage contribution
pba_s_sorted = np.around(pba_s_sorted,3)                                            # evenly round to 3 decimals

# sector names of the top 5 emitters
for i in range(0,5) :
        print("Top", i+1, label_s[rank[i]],"(sector index =", rank[i], "),","CO2 =", pba_s_sorted[i, 0],"Mt;")

##############################################
##############################################
# TASK 5. Identify top-5 final demand spending sectors that contribute to consumption-based emissions
# TASK 6. Identify top-5 countries where the specific country displaced emissions to

label_s = list(mrio['label']['product']['Name']) + ['Household direct']
cba_s = f_co2 @ L @ np.diag(y)*conv
cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba[0,1])                                                   # append household direct emissions to the end
cba_r = cba_rs.sum(1)                                                               # by emitting regions
cba_r[c] = cba_r[c] + cba[0,1]                                                      # add HH direct emissions

# cba_s_sort: we want to report in rows each final demand sector(product) plus household
# 3 cols: sectoral emissions, percent, and cumulative percent
cba_s_sorted = np.zeros((ns+1,3))
rank = np.argsort(cba_s)                                                            # sort from 'low to high' and returns indices
rank = np.flip(rank)                                                                # flip so it goes 'high to low'
cba_s_sorted[:,0] = np.sort(cba_s)                                                  # sort the values from 'low to high'
cba_s_sorted[:,0] = np.flip(cba_s_sorted[:,0])                                      # flip so the values go from high to low
cba_s_sorted[:,1] = cba_s_sorted[:,0]/cba[0,2]*100                                  # percentage contribution
cba_s_sorted[:,2] = np.cumsum(cba_s_sorted[:,1])                                    # cumulative percentage contribution
cba_s_sorted = np.around(cba_s_sorted,3)                                            # evenly round to 3 decimals
# sector names of the top 5 final demand product categories
for i in range(0,5) :
        print("Top", i+1, label_s[rank[i]],"(sector index =", rank[i], "),","CO2 =", cba_s_sorted[i, 0],"Mt;")


# cba_r_sort: we want to report in rows each country of origin
# 3 cols: country emissions, percent, and cumulative percent
label_r = list(mrio['label']['region']['CountryName'])
cba_r_sorted = np.zeros((nr,3))
rank = np.argsort(cba_r)                                                            # sort from 'low to high' and returns indices
rank = np.flip(rank)                                                                # flip so it goes 'high to low'

cba_r_sorted[:,0] = np.sort(cba_r)                                                  # sort the values from 'low to high'
cba_r_sorted[:,0] = np.flip(cba_r_sorted[:,0])                                      # flip so the values go from high to low
cba_r_sorted[:,1] = cba_r_sorted[:,0]/cba[0,2]*100                                  # percentage contribution
cba_r_sorted[:,2] = np.cumsum(cba_r_sorted[:,1])                                    # cumulative percentage contribution
cba_r_sorted = np.around(cba_r_sorted,3)                                            # evenly round to 3 decimals
# country names of the top 5 displacement locations
if c in rank[0:5] :
    for i in range(0,6) :
        if rank[i] != c :
            print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], "),","CO2 =", cba_r_sorted[i, 0],"Mt;")
else :
    for i in range(0,5) :
        print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], "),","CO2 =", cba_r_sorted[i, 0],"Mt;")

#%% Changes in final demand with the processed beef sector (PB)
# final demand of the UK in 2020
y_UK = Y[:, c*ny : (c+1)*ny].sum(1)

# Assuming the p_a percentage of reduction of PB in UK's domestic final demand after the Brexit.
p_a = 0.1
y_AB_UK = (1-p_a)*(y_UK[ns*c+kpb)].sum())

# Assuming the p_b(%) percentage of reduction of PB exports from EU27 to UK after the Brexit.
p_b = 0.3

    # Specifying the postion of PB in the product list of the EXIOBASE
kpb = 43
y_AB_EU = 0

for i in range(27):
    y_AB_EU += (1-p_b)*y_UK[ns*i+kpb]

# Assuming the p_c(%) percentage of increase of PB exports from Australia to UK after the free tariff agreement.
p_c = 0.4

    # Specifying the position of Australia in the country list.
AU = 37

y_AB_AU = (1-p_c)*y_UK[ns*AU+kpb]

print("UK's domestic fimal demand for PB:",np.around(y_AB_UK,3), "M.EUR")
print("PB exports from EU27 to UK:",np.around(y_AB_EU,3), "M.EUR")
print("PB exports from Australia to UK:",np.around(y_AB_AU,3), "M.EUR")

#%% Changes in co2 emissions and water used

label_s = list(mrio['label']['product']['Name']) + ['Household direct']
cba_hh = H_co2[c*ny : (c+1)*ny].sum() *conv
cba_s = f_co2 @ L @ np.diag(y)*conv
cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba_hh)                                                   # append household direct emissions to the end
cba_r = cba_rs.sum(1)                                                               # by emitting regions
cba_r[c] = cba_r[c] + cba_hh                                                     # add HH direct emissions

#%% Consumption-based CO2 emissions in UK agricultural production sectors (2011, MtCO2)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

fig= plt.figure(figsize=(15,15))
sns.set_theme(rc={'ytick.labelsize':12,'xtick.labelsize':10})
cbs_rs_fp = pd.DataFrame(cba_rs[:,0:17], index= label_r, columns= label_s[0:17])
fig = sns.heatmap(cbs_rs_fp,vmin=0, vmax=0.5, cmap="YlGnBu",yticklabels=True, linewidths=.8)
plt.setp(fig.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
fig.set_title(" Consumption-based CO2 emissions in UK agricultural production sectors (2011, MtCO2)")
plt.savefig('/Users/hp/OneDrive - Universiteit Leiden/4. Leiden Univ/2021-WN EIOA course by Ranran/IGA/cbs_rs_fp.png',dpi=300)

#%%  Main CO2 outsouring regions of UK agricultural production sectors (2011, MtCO2)
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10))
cba_rs_fp= cba_rs[:,0:17].sum(1)
cba_rs_fp_eu=cba_rs[0:27,0:17].sum(0)
cba_rs_fp_uk=cba_rs[27,0:17]
cba_rs_fp_rw=cba_rs[28:49,0:17].sum(0)
x_max = 17
x_coords = np.linspace(1, x_max, num=17, endpoint=False)
width = 1
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
ax.bar(
    x_coords,
    cba_rs_fp_uk,
    width=width, label='UK'
)
ax.bar(
    x_coords,
   cba_rs_fp_eu,label='EU',
    width=width
)
ax.bar(
    x_coords,
    cba_rs_fp_rw,label='Rest of the world',
    width=width
)
plt.yticks([0.5,1,1.5,2,2.5,3,3.5])
plt.xticks(np.arange(1, 18, step=1))
ax.set_xticklabels(label_s[0:17])
plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment='right')
plt.legend(loc = "upper right")                          # Lable the plot (check tutorial link above)
plt.title('Main CO2 outsouring regions of UK agricultural production sectors (2011, MtCO2)')
plt.savefig('/Users/hp/OneDrive - Universiteit Leiden/4. Leiden Univ/2021-WN EIOA course by Ranran/IGA/cba_rs_fp_main.png',dpi=300)
plt.show()
plt.close()
#%%
