#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import math
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from scipy import stats


# In[7]:

file_nc = {"CM-CH":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/OCEAN-COU/HR/TC-NH_TRACK_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_19500101-20141231.nc",
        "CM-CL":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/OCEAN-COU/LR/TC-NH_TRACK_CMCC-CM2-HR4_hist-1950_r1i1p1f1_gn_19500101-20141231.nc",
        "Had-CH":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/OCEAN-COU/HR/TC-NH_TRACK_HadGEM3-GC31-HM_hist-1950_r1i1p1f1_gn_19500101-20141231.nc",
        "Had-CL":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/OCEAN-COU/LR/TC-NH_TRACK_HadGEM3-GC31-LL_hist-1950_r1i1p1f1_gn_19500101-20141231.nc",
        "CN-CH":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/OCEAN-COU/HR/TC-NH_TRACK_CNRM-CM6-1-HR_hist-1950_r1i1p1f2_gr_19500101-20141231.nc",
        "CN-CL":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/OCEAN-COU/LR/TC-NH_TRACK_CNRM-CM6-1_hist-1950_r1i1p1f2_gr_19500101-20141231.nc"}

path_obs = {"OBS":glob.glob("data/JTWC1/*")}


# 横坐标年 纵坐标LMI LMI就是台风达到最大强度时的平均纬度

# In[8]:


## 寻找每个台风的最高风速气压对应的纬度 
mslp_lat = {}
mslp_wind = {}
mslp_year = {}
mslp_lon = {}
for run in list(file_nc.keys()):
    print(run)
    mslp_lat[run] = []
    mslp_lon[run] = []
    mslp_wind[run] = []
    mslp_year[run] = []
    data = xr.open_dataset(file_nc[run])
    index_pt = data["FIRST_PT"]
    lat_all = data["lat"]
    lon_all = data["lon"]
    mslp_all = data["psl"]
    wind_all = data["sfcWind"]
    year_all = data["time"].dt.year.values
    for i in tqdm(range(len(index_pt)-1)):
        i_s = index_pt[i].values
        i_e = index_pt[i+1].values
        if (year_all[i_s]>1983)&(year_all[i_e-1]<2015)&(lat_all[i_s]>0)&(lat_all[i_e-1]<90)&(lon_all[i_s]>90)&(lon_all[i_e-1]<180):
            lat = lat_all[i_s:i_e].values
            mslp = mslp_all[i_s:i_e].values
            wind = wind_all[i_s:i_e].values
            year = year_all[i_s:i_e]
            i_max = np.argmax(wind)
            mslp_lat[run].append(lat[i_max])
            mslp_wind[run].append(wind[i_max])
            mslp_year[run].append(year[i_max])
    mslp_lat[run] = np.array(mslp_lat[run])
    mslp_wind[run] = np.array(mslp_wind[run])
    mslp_year[run] = np.array(mslp_year[run])


# In[9]:


# 观测值汇总
run = "OBS"
mslp_year[run] = []
mslp_lat[run] = []
mslp_wind[run] = []
for path in path_obs["OBS"]:
    files = glob.glob(path+"/*")
    year = files[0][14:18]
    for file in tqdm(files):
        print(file)
        data = pd.read_table(file,sep=",",usecols =[6,8],header=None)
        lat = data.iloc[:,0].apply(lambda x:int(str(x)[:-1])/10.).values
        wind = data.iloc[:,1].values*0.51444
        i_max = np.argmax(wind)
        mslp_year[run].append(int(year))
        mslp_lat[run].append(lat[i_max])
        mslp_wind[run].append(wind[i_max])
mslp_year[run] = np.array(mslp_year[run])
mslp_lat[run] = np.array(mslp_lat[run])
mslp_wind[run] = np.array(mslp_wind[run])


# In[10]:


# 计算每年的平均值
year_c = np.arange(1984,2015,1)
lat_mean = {}
wnd_mean = {}
dpi = {}
for run in list(mslp_wind.keys()):
    lat_mean[run] = np.full_like(year_c,np.nan,dtype=np.float)
    wnd_mean[run] = np.full_like(year_c,np.nan,dtype=np.float)
    for i,year_con in enumerate(year_c):
        cond  = mslp_year[run]==year_con
        lat_se = mslp_lat[run][cond]
        wnd_se = mslp_wind[run][cond]
        if len(lat_se)>0:
            lat_mean[run][i] = lat_se.mean() 
            wnd_mean[run][i] = wnd_se.mean() 
        else:
            lat_mean[run][i] = np.nan
            wnd_mean[run][i] = np.nan
        #dpi[run][i] = np.sum(wnd_se**3*lat_se)/np.sum(wnd_se**3)
    print(run,lat_mean[run],wnd_mean[run])

# In[13]:


plt.style.use("ggplot")
fig,ax = plt.subplots(1,1,figsize=(7,8))
# 线型
ls = {"CM-CH":"-",
      "Had-CH":"-",
      "CN-CH":"-",
      "Had-CL":"--",
      "CN-CL":"--",
      "CM-CL":"--",
      "OBS":"-"}
# 点型
marker = {"CM-CH":"o",
      "Had-CH":"o",
      "CN-CH":"o",
      "Had-CL":"x",
      "CN-CL":"x",
      "CM-CL":"x",
      "OBS":"p"}

# 线条颜色
lc = {"CM-CH":"r",
      "CM-CL":"r",
      "Had-CH":"b",
      "Had-CL":"b",
      "CN-CH":"g",
      "CN-CL":"g",
      "OBS":"k"}

# 如果要绘制平均的话 y = lat_mean[run],x = wnd_mean[run]
for run in list(mslp_lat.keys()):
    ind = np.argsort(mslp_wind[run])
    x = mslp_wind[run][ind]
    y = mslp_lat[run][ind]
    # 散点绘制
    ax.scatter(x,y,s=5,marker = marker[run],color=lc[run],alpha=0.5)
    # 多项式拟合
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    # 确定范围
    xp = np.linspace(x.min(),x.max(),20)
    # 绘制拟合曲线
    ax.plot(xp,p(xp),label=run,ls = ls[run],color=lc[run],linewidth=2.5,alpha=0.8)
    # 绘制置信区间
    n = len(x)
    yfit = p(x)
    resid = yfit-y
    s_err = np.sqrt(np.sum(resid**2)/(n - 2))  
    t = stats.t.ppf(0.975, n - 2)
    ci = t * s_err * np.sqrt(    1/n + (x - np.mean(x))**2/np.sum((x-np.mean(x))**2))
    ax.fill_between(x, yfit+ci, yfit-ci,color=lc[run], edgecolor='k',alpha=0.3) 

    # 绘制拟合曲线
#    ax.plot(xp,p(xp),label=run,ls = ls[run],color=lc[run],alpha=0.8)
#    ax.fill_between(xp, yfit+ci, yfit-ci, color=lc[run], edgecolor='k',alpha=0.3)

# tick 
xticks = np.arange(0,90,10)
yticks = np.arange(0,50,10)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels(list(map(lambda x:str(x)+"˚N",yticks)))
# label
#ax.set_xlabel(r"Max.lifetime 10m wind speed ($m·s^{-1}$)",fontsize=14)
ax.set_xlabel(r"Max.lifetime 10m wind speed",fontsize=14)
ax.set_ylabel(r"$\varphi_{LMI}$",fontsize=18)
# lim
ax.set_ylim(0,50)
# 题目
#ax.set_title(title,fontsize=15,loc="center")
ax.set_title(f"Coupled",fontsize=15,loc="center")
# grid
ax.grid(True)
ax.legend(fontsize=12)
plt.savefig("Coupled.png",dpi=300)


# In[ ]:




