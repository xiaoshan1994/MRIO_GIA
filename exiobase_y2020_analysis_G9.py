"""
Tasks:

"""

import numpy as np
import time
import pickle as pkl
import os
from pathlib import Path
np.set_printoptions(precision=2)

#%% Load MRIO from Pickle

os.getcwd()
mrio_dir = Path('C:/Users/hp/OneDrive - Universiteit Leiden/4. Leiden Univ/2021-WN EIOA course by Ranran/IGA')  # Fill " " with your working directory path


tstart = time.time()

mrio_name = 'mrio_y2020.pkl'
mrio_str = mrio_dir.joinpath(mrio_name)
pkl_in = open(mrio_str,"rb")                                                        # 'open' the mrio.pkl file
mrio = pkl.load(pkl_in)                                                             # load the mrio.pkl file to mrio
pkl_in.close()

tend = time.time()
print('Done reading in %5.2f s\n'% (tend - tstart))

# category counts

nr = mrio['label']['region'].count()[0]
ns = mrio['label']['product'].count()[0]
ny = mrio['label']['final'].count()[0]

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


# calculate total output
x = L @ Y.sum(1)

# calculate direct emissions intensity
f_co2 = F_co2/x                                                                     # resulting in na or inf when x = 0
f_co2 = np.nan_to_num(f_co2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)           # replace na and inf with 0


# calculate direct water consumption intensity
f_wc = F_wc/x                                                                     # resulting in na or inf when x = 0
f_wc = np.nan_to_num(f_wc, copy=False, nan=0.0, posinf=0.0, neginf=0.0)           # replace na and inf with 0



# change to choose different region: c in 0 to 48
# position 27 is the UK
c = 27

# To locate country columns or rows in A and L: 'c*ns : (c+1)*ns'
# To locate country columns in Y: 'c*ny : (c+1)*ny'

#%% Calculate territorial emissions for a specific country



""""CO2"""

conv = 1e-9

# We want to report three cols: production-related, households and total, plus a global contribution percentage

pba = np.zeros((1,4))
pba[:,0] = F_co2[c*ns : (c+1)*ns].sum() *conv                                       # direct emissions by producers
pba[:,1] = H_co2[c*ny : (c+1)*ny].sum() *conv                                       # direct emissions by households
pba[:,2] = pba[:,0:2].sum()
pba[:,3] = pba[:,2]/((F_co2.sum()+H_co2.sum())*conv)*100


#%%
# Calculate for the UK consumption-based CO2 emissions in 2020 (estimations from the EXIOBASE 2020)


# We want to report three cols: production-generated, households and total, plus a global contribution percentage

cba = np.zeros((1,4))

#y is the final demand of the UK before the treaty
y = Y[:, c*ny : (c+1)*ny].sum(1)                                                    # aggregate final demand to one column

cba[:,0] = f_co2 @ L @ y *conv                                                      # footprint traced to production emissions
cba[:,1] = H_co2[c*ny : (c+1)*ny].sum()*conv                                        # footprint traced to household direct emissions
cba[:,2] = cba[:,:2].sum()
cba[:,3] = cba[:,2]/((F_co2.sum()+H_co2.sum())*conv)*100


# final demand of the UK in 2020 after the treaty 
y_after = Y[:, c*ny : (c+1)*ny].sum(1)
au_fd_sum=0                     # final demand moved to Australia export
#print(y_after[27*200+42])

# Assuming the p_a percentage of reduction of product of meat cattle in UK's domestic final demand after the treaty. (index=42)
p_a = 0.05

#5% will be moving from domestic product in the UK to Australia exports
uk_red= p_a * y_after[27*200+42]

y_after[27*200+42] = (1-p_a)*y_after[27*200+42] #new value



# Assuming the p_b(%) percentage of reduction of PB exports from EU27 to UK after the treaty.
p_b = 0.11

for i in range(27):
    au_fd_sum += p_b * y_after[200*i+42]
   # print(y_after[200*i+42])
    y_after[200*i+42]= (1-p_b)*y_after[200*i+42]
    #print(y_after[200*i+42])

xd=0
for i in range(49):
    xd+=y_after[200*i+42]
 #   print(xd)

print("UK's domestic final demand reduction for products of meat cattle:",uk_red, "M.EUR")
        
print("EU final demand reduction for products of meat cattle:",au_fd_sum, "M.EUR")

au_fd_sum+=uk_red
# Assuming the reductions from EU and the UK lead to an increase of products of meat cattle exports from Australia to UK after the free tariff agreement.

print("Products of meat cattle exports from Australia to UK before agreement:",y_after[37*200+42], "M.EUR")

y_after[37*200+42] = y_after[37*200+42] + au_fd_sum

print("Products of meat cattle exports from Australia to UK after agreement:",y_after[37*200+42], "M.EUR")
print()
#You can see that adding the three first numbers printed gives you the last one: 
#so no change in the consumption patterns of the world

#%% Identify top-5 sectors that contribute to production-based emissions

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

#%%
# Identify top-5 final demand spending sectors that contribute to consumption-based emissions
# Identify top-5 countries where the specific country displaced emissions to

label_s = list(mrio['label']['product']['Name']) + ['Household direct']
cba_s = f_co2 @ L @ np.diag(y)*conv
cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba[0,1])                                                   # append household direct emissions to the end
cba_r = cba_rs.sum(1)                                                               # by emitting regions
cba_r[c] = cba_r[c] + cba[0,1]                                                      # add HH direct emissions
print()
#print(cba_rs.sum())
print()
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
# country names of the top displacement locations
if c in rank[0:20] :
    for i in range(0,23) :
        if rank[i] != c :
            print("Top", i, label_r[rank[i]],"(country index =", rank[i], "),","CO2 =", cba_r_sorted[i, 0],"Mt;")
else :
    for i in range(0,23) :
        print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], "),","CO2 =", cba_r_sorted[i, 0],"Mt;")


#%%  Main CO2 outsouring regions of UK processed food product sectors (2020, MtCO2)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

print()
print("CO2 Plot 1: ")
fig= plt.figure(figsize=(15,15))
sns.set_theme(rc={'ytick.labelsize':12,'xtick.labelsize':10})
cbs_rs_fp = pd.DataFrame(cba_rs[:,42:53], index= label_r, columns= label_s[42:53])
fig = sns.heatmap(cbs_rs_fp,vmin=0, vmax=0.5, cmap="YlGnBu",yticklabels=True, linewidths=.8)
plt.setp(fig.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
fig.set_title(" Consumption-based CO2 emissions in UK processed agricutltural products (2020, MtCO2)")
plt.savefig('co2emissions_per_country_processed2020.svg',bbox_inches='tight',pad_inches = 0)

plt.show()
plt.close()

#%%
# Consumption-based CO2 emissions in UK's products of meat cattle (2020,MtCO2)
fig= plt.figure(figsize=(3,15))
sns.set_theme(rc={'ytick.labelsize':12,'xtick.labelsize':10})
cbs_rs_fp = pd.DataFrame(cba_rs[:,42], index= label_r, columns= ['Products of meat cattle'])
fig = sns.heatmap(cbs_rs_fp,vmin=0, vmax=0.3,cmap="YlGnBu",yticklabels=True, linewidths=.8,annot=True,fmt= '.2e')
plt.setp(fig.get_xticklabels(), ha="right", rotation_mode="anchor")
fig.set_title("Consumption-based CO2 emissions in UK's products of meat cattle (2020,MtCO2)")
plt.savefig('co2emissions_consumption_uk_bf2020.svg',bbox_inches='tight',pad_inches = 0)
plt.show()
plt.close()

#%% Consumption-based CO2 emissions  in UK processed agricutltural products divided in UK, EU and Rest of the World (2020, MtCO2)

cba_rs_fp= cba_rs[:,42:53].sum(0)
print()
print("Total CBA co2 emissions: ", cba_rs.sum())
print("Total CBA co2 emissions of processed food products: ", cba_rs_fp.sum())
print("CBA co2 emissions of products of meat cattle: ", cba_rs_fp[0])

cba_rs_fp_eu=cba_rs[0:27,42:53].sum(0) 
cba_rs_fp_uk=cba_rs[27,42:53]
cba_rs_fp_rw=cba_rs[28:49,42:53].sum(0) 

lab1=['UK','EU', 'Rest of the World']
lab2= label_s[42:53]

tot= np.zeros((11,3))
tot[:,0]= cba_rs_fp_uk
tot[:,1]= cba_rs_fp_eu
tot[:,2]= cba_rs_fp_rw

prod_cattle=np.zeros((2,3))  # array to compare products of cattle meat before and after the agreement
print()
prod_cattle[0]=np.array([tot[0,0],tot[0,1],tot[0,2]])
#print(prod_cattle)



print("CO2 Plot 2: ")
fig, ax = plt.subplots(figsize=(10,10))


temp= np.zeros(11)
for j in range(3):                                      
    plt.bar([1,2,3,4,5,6,7,8,9,10,11], tot[:,j], bottom = temp, label=lab1[j])
    temp = temp + tot[:,j]

plt.legend(loc = "upper right")                          # Lable the plot (check tutorial link above)                 
plt.title('Consumption-based co2 emissions in UK processed agricutltural products before Treaty (2020, MtCO2)')
plt.xticks(np.arange(1,12))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticklabels(lab2)

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=10) 

plt.savefig('co2emissions_UKEURW_processed2020_before_treaty.svg',bbox_inches='tight',pad_inches = 0)
plt.show()
plt.close()


#%% Estimated changes in final demand with the products of meat cattle  after agreement (index=42)

cba[:,0] = f_co2 @ L @ y_after *conv                                                      # footprint traced to production emissions
cba[:,1] = H_co2[c*ny : (c+1)*ny].sum()*conv                                        # footprint traced to household direct emissions
cba[:,2] = cba[:,:2].sum()
cba[:,3] = cba[:,2]/((F_co2.sum()+H_co2.sum())*conv)*100

cba_s = f_co2 @ L @ np.diag(y_after)*conv

cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right

print()
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba[0,1])                                                   # append household direct emissions to the end
#print(cba_s.sum())
cba_r = cba_rs.sum(1)                                                               # by emitting regions
cba_r[c] = cba_r[c] + cba[0,1]                                                      # add HH direct emissions

cba_r_sorted = np.zeros((nr,3))
rank = np.argsort(cba_r)                                                            # sort from 'low to high' and returns indices
rank = np.flip(rank)                                                                # flip so it goes 'high to low'

cba_r_sorted[:,0] = np.sort(cba_r)                                                  # sort the values from 'low to high'
cba_r_sorted[:,0] = np.flip(cba_r_sorted[:,0])                                      # flip so the values go from high to low
cba_r_sorted[:,1] = cba_r_sorted[:,0]/cba[0,2]*100                                  # percentage contribution
cba_r_sorted[:,2] = np.cumsum(cba_r_sorted[:,1])                                    # cumulative percentage contribution
cba_r_sorted = np.around(cba_r_sorted,3)                                            # evenly round to 3 decimals
# country names of the top 5 displacement locations
if c in rank[0:20] :
    for i in range(0,23) :
        if rank[i] != c :
            print("Top", i, label_r[rank[i]],"(country index =", rank[i], "),","CO2 =", cba_r_sorted[i, 0],"Mt;")
else :
    for i in range(0,23) :
        print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], "),","CO2 =", cba_r_sorted[i, 0],"Mt;")


cba_rs_fp= cba_rs[:,42:53].sum(0) 
print()
print("Total CBA co2 emissions: ", cba_rs.sum())
print("Total CBA co2 emissions of processed food products: ", cba_rs_fp.sum())
print("CBA co2 emissions of products of meat cattle: ", cba_rs_fp[0])

cba_rs_fp_eu=cba_rs[0:27,42:53].sum(0) 
cba_rs_fp_uk=cba_rs[27,42:53]
cba_rs_fp_rw=cba_rs[28:49,42:53].sum(0) 

lab1=['UK','EU', 'Rest of the World']
lab2= label_s[42:53]

tot= np.zeros((11,3))
tot[:,0]= cba_rs_fp_uk
tot[:,1]= cba_rs_fp_eu
tot[:,2]= cba_rs_fp_rw

prod_cattle[1]=np.array([tot[0,0],tot[0,1],tot[0,2]])

#%% Plot processed agricultural products after Treaty

print()
print("CO2 Plot 3: ")
fig, ax = plt.subplots(figsize=(10,10))

temp= np.zeros(11)
for j in range(3):                                      
    plt.bar([1,2,3,4,5,6,7,8,9,10,11], tot[:,j], bottom = temp, label=lab1[j])
    temp = temp + tot[:,j]

plt.legend(loc = "upper left")                          # Lable the plot (check tutorial link above)                 
plt.title('Consumption-based co2 emissions in UK processed agricutltural products after Treaty (2020, Mm3)')
plt.xticks(np.arange(1,12))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticklabels(lab2)

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=10) 

plt.savefig('co2emissions_UKEURW_processed2020_after_treaty.svg',bbox_inches='tight',pad_inches = 0)
plt.show()
plt.close()

#%% Plotting the comparison of CBA CO2 before and after

lab3=["before","after"]
print()
print("CO2 Plot 4: ")

fig, ax = plt.subplots(figsize=(10,10))
temp= np.zeros(2)


for j in range(3):                                      
    plt.bar([1,2], prod_cattle[:,j], bottom = temp, label=lab1[j])
    temp = temp + prod_cattle[:,j]


plt.xticks([1,2])
plt.setp(ax.get_xticklabels(), horizontalalignment='right')
ax.set_xticklabels(lab3)
plt.title('Consumption-based CO2 emissions in the UK for products of meat cattle before and after the Treaty (2020, MtCO2)')
plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=10) 

plt.legend(loc = "upper right")  

for rect in ax.patches:
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    xx = rect.get_x()
    yy= rect.get_y()
    
    label_text = height
    
    # ax.text(x, y, text)
    label_x = xx + width / 2
    label_y = yy + height / 2
    label_text = f'{height:.2f}'   # f'{height:.2f}' to format decimal values
    ax.text(label_x, label_y, label_text, ha='center', va='center',color="white", fontsize=15, fontweight="bold")

plt.savefig('comparison_co2.svg',bbox_inches='tight',pad_inches = 0)
plt.show()

plt.close()

#%% Extension indicator of water consumption in UK's products of meat cattle



# Calculate territorial water consumption for a specific country
# convert units from Mm3 to Mm3 (*1e-6)
conv = 1

# We want to report three cols: production-related, households and total, plus a global contribution percentage

pba = np.zeros((1,4))
pba[:,0] = F_wc[c*ns : (c+1)*ns].sum() *conv                                       # direct emissions by producers
pba[:,1] = H_wc[c*ny : (c+1)*ny].sum() *conv                                       # direct emissions by households
pba[:,2] = pba[:,0:2].sum()
pba[:,3] = pba[:,2]/((F_wc.sum()+H_wc.sum())*conv)*100


# Calculate for a specific country consumption-based wc


# We want to report three cols: production-generated, households and total, plus a global contribution percentage

cba = np.zeros((1,4))
                                                                                   # aggregate final demand to one column
cba[:,0] = f_wc @ L @ y *conv                                                      # footprint traced to production emissions
cba[:,1] = H_wc[c*ny : (c+1)*ny].sum()*conv                                        # footprint traced to household direct emissions
cba[:,2] = cba[:,:2].sum()
cba[:,3] = cba[:,2]/((F_wc.sum()+H_wc.sum())*conv)*100


#%%
# Identify top sectors that contribute to production-based water consumption

# pba_s: we want to report in rows each sector plus household

label_s = list(mrio['label']['product']['Name']) + ['Household direct']# forming a list of 201 rows
pba_s = F_wc[c*ns : (c+1)*ns] *conv
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
# sector names of the top emitters
for i in range(0,5) :
        print("Top", i+1, label_s[rank[i]],"(sector index =", rank[i], "),","WC =", pba_s_sorted[i, 0],"Mm3;")

#%%
# Identify top final demand spending sectors that contribute to water consumption
# Identify top countries where the specific country displaced water consumption to

label_s = list(mrio['label']['product']['Name']) + ['Household direct']
cba_s = f_wc @ L @ np.diag(y)*conv
cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba[0,1])                                                   # append household direct emissions to the end
cba_r = cba_rs.sum(1)                                                               # by emitting regions
cba_r[c] = cba_r[c] + cba[0,1]                                                      # add HH direct emissions
print()
print(cba_rs.sum())
print()
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
        print("Top", i+1, label_s[rank[i]],"(sector index =", rank[i], "),","WC =", cba_s_sorted[i, 0],"Mm3;")


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
if c in rank[0:20] :
    for i in range(0,23) :
        if rank[i] != c :
            print("Top", i, label_r[rank[i]],"(country index =", rank[i], "),","WC =", cba_r_sorted[i, 0],"Mm3;")
else :
    for i in range(0,23) :
        print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], "),","WC =", cba_r_sorted[i, 0],"Mm3;")


#%%  Main water consumption outsouring regions of UK agricultural production sectors (2020, Mm3)

print()
print("Water Plot 1: ")
fig= plt.figure(figsize=(15,15))
sns.set_theme(rc={'ytick.labelsize':12,'xtick.labelsize':10})
cbs_rs_fp = pd.DataFrame(cba_rs[:,42:53], index= label_r, columns= label_s[42:53])
fig = sns.heatmap(cbs_rs_fp,vmin=0, vmax=max(map(max,cba_rs[:,42:53]))/2, cmap="YlGnBu",yticklabels=True, linewidths=.8)
plt.setp(fig.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
fig.set_title("Embodied water consumption per country in UK processed agricutltural products (2020, Mm3)")
plt.savefig('water_consumption_per_country_processed2020.svg',bbox_inches='tight',pad_inches = 0)
plt.show()
plt.close()

#%%  Embodied water consumption in UK agricultural production sectors (2020, Mm3)

cba_rs_fp= cba_rs[:,42:53].sum(0)
print()
print("Total CBA water consumption: ", cba_rs.sum())
print("Total CBA water consumption of processed food products: ", cba_rs_fp.sum())
print("CBA water consumption of products of meat cattle: ", cba_rs_fp[0])

cba_rs_fp_eu=cba_rs[0:27,42:53].sum(0) 
cba_rs_fp_uk=cba_rs[27,42:53]
cba_rs_fp_rw=cba_rs[28:49,42:53].sum(0) 

lab1=['UK','EU', 'Rest of the World']
lab2= label_s[42:53]

tot= np.zeros((11,3))
tot[:,0]= cba_rs_fp_uk
tot[:,1]= cba_rs_fp_eu
tot[:,2]= cba_rs_fp_rw

prod_cattle=np.zeros((2,3)) # array to compare products of cattle meat before and after the agreement
print()
prod_cattle[0]=np.array([tot[0,0],tot[0,1],tot[0,2]])

print("Water Plot 2: ")
fig, ax = plt.subplots(figsize=(10,10))

temp= np.zeros(11)
for j in range(3):                                      
    plt.bar([1,2,3,4,5,6,7,8,9,10,11], tot[:,j], bottom = temp, label=lab1[j])
    temp = temp + tot[:,j]

plt.legend(loc = "upper left")                          # Lable the plot (check tutorial link above)                 
plt.title('Embodied water consumption in UK processed agricutltural products before Treaty (2020, Mm3)')
plt.xticks(np.arange(1,12))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticklabels(lab2)

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=10) 

plt.savefig('water_consumption_UKEURW_processed2020_before_treaty.svg',bbox_inches='tight',pad_inches = 0)
plt.show()
plt.close()


#%% Changes in final demand with the products of meat cattle  (index=42)

cba[:,0] = f_wc @ L @ y_after *conv                                                      # footprint traced to production emissions
cba[:,1] = H_wc[c*ny : (c+1)*ny].sum()*conv                                        # footprint traced to household direct emissions
cba[:,2] = cba[:,:2].sum()
cba[:,3] = cba[:,2]/((F_wc.sum()+H_wc.sum())*conv)*100

cba_s = f_wc @ L @ np.diag(y_after)*conv

cba_rs = np.reshape(cba_s,(nr,ns))                                                  # pay attention to get the new shape dimension right

print()
cba_s = cba_rs.sum(0)                                                               # by final demand sector
cba_s = np.append(cba_s,cba[0,1])                                                   # append household direct emissions to the end
cba_r = cba_rs.sum(1)                                                               # by emitting regions

cba_r[c] = cba_r[c] + cba[0,1]                                                      # add HH direct emissions

cba_r_sorted = np.zeros((nr,3))
rank = np.argsort(cba_r)                                                            # sort from 'low to high' and returns indices
rank = np.flip(rank)                                                                # flip so it goes 'high to low'

cba_r_sorted[:,0] = np.sort(cba_r)                                                  # sort the values from 'low to high'
cba_r_sorted[:,0] = np.flip(cba_r_sorted[:,0])                                      # flip so the values go from high to low
cba_r_sorted[:,1] = cba_r_sorted[:,0]/cba[0,2]*100                                  # percentage contribution
cba_r_sorted[:,2] = np.cumsum(cba_r_sorted[:,1])                                    # cumulative percentage contribution
cba_r_sorted = np.around(cba_r_sorted,3)                                            # evenly round to 3 decimals
# country names of the top 5 displacement locations
if c in rank[0:20] :
    for i in range(0,23) :
        if rank[i] != c :
            print("Top", i, label_r[rank[i]],"(country index =", rank[i], "),","WC =", cba_r_sorted[i, 0],"Mt;")
else :
    for i in range(0,23) :
        print("Top", i+1, label_r[rank[i]],"(country index =", rank[i], "),","WC =", cba_r_sorted[i, 0],"Mt;")



#You can see that adding the three first numbers printed gives you the last one: 
#so no change in the consumption patterns of the world

#%% Plot Embodied water consumption in UK processed agricutltural products after Treaty (2020, Mm3)

cba_rs_fp= cba_rs[:,42:53].sum(0) 
print()
print("Total CBA water consumption: ", cba_rs.sum())
print("Total CBA water consumption of processed food products: ", cba_rs_fp.sum())
print("CBA water consumption of products of meat cattle: ", cba_rs_fp[0])

cba_rs_fp_eu=cba_rs[0:27,42:53].sum(0) 
cba_rs_fp_uk=cba_rs[27,42:53]
cba_rs_fp_rw=cba_rs[28:49,42:53].sum(0) 

lab1=['UK','EU', 'Rest of the World']
lab2= label_s[42:53]

tot= np.zeros((11,3))
tot[:,0]= cba_rs_fp_uk
tot[:,1]= cba_rs_fp_eu
tot[:,2]= cba_rs_fp_rw

prod_cattle[1]=np.array([tot[0,0],tot[0,1],tot[0,2]])

print()
print("Water Plot 3: ")
fig, ax = plt.subplots(figsize=(10,10))


temp= np.zeros(11)
for j in range(3):                                      
    plt.bar([1,2,3,4,5,6,7,8,9,10,11], tot[:,j], bottom = temp, label=lab1[j])
    temp = temp + tot[:,j]

plt.legend(loc = "upper left")                          # Lable the plot (check tutorial link above)                 
plt.title('Embodied water consumption in UK processed agricutltural products after Treaty (2020, Mm3)')
plt.xticks(np.arange(1,12))
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ax.set_xticklabels(lab2)

plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=10) 

plt.savefig('water_consumption_UKEURW_processed2020_after_treaty.svg',bbox_inches='tight',pad_inches = 0)
plt.show()

plt.close()

#%%  Plotting the comparison of CBA water consumption before and after

lab3=["before","after"]


print()
print("Water Plot 4: ")
fig, ax = plt.subplots(figsize=(10,10))
temp= np.zeros(2)

for j in range(3):                                      
    plt.bar([1,2], prod_cattle[:,j], bottom = temp, label=lab1[j])
    temp = temp + prod_cattle[:,j]


plt.xticks([1,2])
plt.setp(ax.get_xticklabels(), horizontalalignment='right')
ax.set_xticklabels(lab3,fontsize=15)
plt.title('Consumption-based water consumption in UK for products of meat cattle before and after the Treaty (2020, Mm3)')
plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=10) 
plt.ylim(0,1200)
plt.legend(loc = "upper left")  

for rect in ax.patches:
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    xx = rect.get_x()
    yy = rect.get_y()
    
    label_text = height
    
    # ax.text(x, y, text)
    label_x = xx+ width / 2
    label_y = yy + height / 2
    label_text = f'{height:.2f}'   # f'{height:.2f}' to format decimal values
    ax.text(label_x, label_y, label_text, ha='center', va='center',color="white", fontsize=15, fontweight="bold")
    plt.savefig('comparison_water.svg',bbox_inches='tight',pad_inches = 0)
plt.show()

plt.close()


