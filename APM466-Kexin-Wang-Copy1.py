#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.linalg import eig
import scipy.optimize as optimize
from sklearn.decomposition import PCA


# In[5]:


#Take a glance at the data
df = pd.read_csv('Anna466.csv') 
df.head(5)


# In[6]:


#Convert data to correct format
df = df.iloc[:,0:-1]
df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
df['Issue Date'] = pd.to_datetime(df['Issue Date'])
df['Coupon'] = df['Coupon'].str.replace(r'%',r'').astype('float')
df['Coupon']=df['Coupon']/100
# Select bonds to analyze
selected_bonds = ['CA135087A610', 'CA135087M276', 'CA135087K528', 'CA135087N423',
                                  'CA135087N340', 'CA135087J967', 'CA135087H490', 'CA135087E679',
                                  'CA135087N266', 'CA135087M847']
df['Time to Maturity'] = df['Months until Maturity ']/12
df_selected = df.loc[df['ISIN'].isin(selected_bonds)]
df_selected= df_selected.sort_values(by = ['Time to Maturity']).reset_index(drop=True)


# In[7]:


df_selected


# In[8]:


days = df_selected.columns[8:-1].tolist()
coupon_rate= df_selected['Coupon'].tolist()
Time_to_maturity = df_selected['Time to Maturity'].tolist()
bonds_prices_perday={}
bonds_prices_perday['2023/1/16'] = df_selected['2023/1/16'].tolist()
bonds_prices_perday['2023/1/17'] = df_selected['2023/1/17'].tolist()
bonds_prices_perday['2023/1/18'] = df_selected['2023/1/18'].tolist()
bonds_prices_perday['2023/1/19'] = df_selected['2023/1/19'].tolist()
bonds_prices_perday['2023/1/20'] = df_selected['2023/1/20'].tolist()
bonds_prices_perday['2023/1/23'] = df_selected['2023/1/23'].tolist()
bonds_prices_perday['2023/1/24'] = df_selected['2023/1/24'].tolist()
bonds_prices_perday['2023/1/25'] = df_selected['2023/1/25'].tolist()
bonds_prices_perday['2023/1/26'] = df_selected['2023/1/26'].tolist()
bonds_prices_perday['2023/1/27'] = df_selected['2023/1/27'].tolist()
bonds_prices_perday['2023/1/30'] = df_selected['2023/1/30'].tolist()


# In[9]:


def bond_spot_rate(price, coupon_rate, T, face_value=100):
    coupon = coupon_rate*100 *0.5
    spot_rate = (coupon / face_value + (face_value / price - 1) / T)
    return spot_rate


# In[13]:


spot_curve_days = {}
for day in days:
    prices_day = df_selected[day].tolist()
    spot_rate = []
    for i in range(len(prices_day)):
        a= bond_spot_rate(prices_day[i], coupon_rate[i], Time_to_maturity[i],face_value=100)
        spot_rate.append(a)
    spot_curve_days[day] = spot_rate


# In[14]:


# Plot spot rate, labelled by day
plt.figure(figsize = (15, 10))
for day in days:
    plt.plot(Time_to_maturity,  spot_curve_days[day], label = day)
plt.title("Spot Rate Curve ")
plt.xlabel("Time until Maturity in Year")
plt.ylabel("Spot rate")
plt.legend()
plt.show()


# In[15]:


# Define function that to determine yield rate
def bond_yield(price, coupon_rate, time_to_maturity, notional=100):
    payment= coupon_rate * notional*0.5
    face_value = notional + coupon_rate * notional
    periodic_rate = coupon_rate * notional / face_value
    num_payments = time_to_maturity * 2
    
    guess = 0.05
    tolerance = 0.0001
    yield_to_maturity = guess
    
    while True:
        price_estimate = 0
        for i in range(int(num_payments)):
            discount_factor = 1 / (1 + yield_to_maturity / 2) ** (2 * (i + 1) / time_to_maturity)
            price_estimate += payment * discount_factor
        price_estimate += notional / (1 + yield_to_maturity / 2) ** (2 * time_to_maturity)
        difference = price - price_estimate
        if abs(difference) < tolerance:
            break
        derivative = 0
        for i in range(int(num_payments)):
            discount_factor = 1 / (1 + yield_to_maturity / 2) ** (2 * (i + 1) / time_to_maturity)
            derivative += -payment * 2 * (i + 1) * discount_factor / time_to_maturity / (1 + yield_to_maturity / 2)
        derivative += -notional * 2 * time_to_maturity / (1 + yield_to_maturity / 2) ** (2 * time_to_maturity)
        yield_to_maturity += difference / derivative
        
    return yield_to_maturity
# Calculate YTM

ytm_curve_bond = {}
yield_maturity = {}

for i in range(10):
    ytm_bond = []
    price_of_bond = df_selected.iloc[i,8:-1].tolist()
    t_to_m = df_selected.iloc[i,-1]
    coupon_rate = df_selected.iloc[i,1]
    
    for price in price_of_bond:
        ytm_bond.append(bond_yield(price, coupon_rate, t_to_m))

    ytm_curve_bond[selected_bonds[i]] = ytm_bond


for i in range(len(days)):
    yield_maturity[days[i]] = []
    for bond in ytm_curve_bond:
        yield_maturity[days[i]].append(ytm_curve_bond[bond][i])
# Plot yield curve
plt.figure(figsize = (15, 10))
for day in days:
    plt.plot(Time_to_maturity, yield_maturity[day], label = day)

plt.title("Yield to Maturity Curve")
plt.xlabel("Year")
plt.ylabel("Yield to Maturity")
plt.legend()
plt.show()


# In[16]:


def forward_rate(spot_rate, t1, t2):
    forward_rate = (1 + spot_rate * t1) / (1 + spot_rate * t2)
    return forward_rate


# In[17]:


# Calculate forward rate
forward_curve_perday = {}
t= [2, 3, 4, 5]
for day in days:
    spot_curve = spot_curve_days[day]
    forward_curve = [0,0,0,0]
    for i in range(4):
        forward_curve[i] = (spot_curve[t[i]*2-1] * t[i] - spot_curve[1]) / (t[i] - 1)
    forward_curve_perday[day] = forward_curve
# Plot forward curve
plt.figure(figsize = (15, 10))
x_label = ['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr']
for day in days:
    plt.plot(x_label, forward_curve_perday[day], label = day)
plt.title("Forward Rate Curve")
plt.xlabel("Forward date")
plt.ylabel("Forward rate")
plt.legend()
plt.show()


# In[28]:


# Covariance matrices
matrix_yield = np.zeros([5, 9])
matrix_forward = np.zeros([4, 9])
#YIELD
for i in range(5):
    for j in range(9):
        de =yield_maturity[days[i]][j+1]
        nu = yield_maturity[days[i]][j]
        matrix_yield[i,j] = np.log(de / nu)
#FORWARD
for i in range(4):
    for j in range(9):
        matrix_forward[i,j] = np.log(forward_curve_perday[days[j+1]][i] / forward_curve_perday[days[j]][i])


# In[29]:


Cov_yield = np.cov(matrix_yield)
Cov_forward = np.cov(matrix_forward)


# In[23]:


Cov_yield


# In[44]:


eig_yield = eig(Cov_yield)
eig_yield


# In[30]:


Cov_forward


# In[31]:


eig_for = eig(Cov_forward)
eig_for


# In[150]:





# In[151]:




