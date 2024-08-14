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
from scipy.stats import linregress
import matplotlib
from besttracks import parse_TCs
import statsmodels.formula.api as smf
matplotlib.rcParams.update({'font.size':18})
from scipy import stats
from scipy.stats import linregress
from metpy.constants import earth_gravity
from metpy.constants import earth_avg_radius
def trend(year, phi):
    year_x = year
    para = np.polyfit(year_x, phi, 1)
    p_95 = np.poly1d(para)
    y_fit = p_95(year_x)
    _,_,_,pmean,_ = linregress(year_x,phi)
    if pmean<0.01:
        pmean = 0.01
    else:
        pmean = math.ceil(round(pmean,2)*10)/10
    return y_fit, ' (p<' + str(round(pmean,2)) + ')',para[0]

from math import radians, sin
def area(lon, lat):#计算单个格点的面积 0.25度的网格分辨率
    lon1, lat1, lon2, lat2 = map(radians, [lon-0.125, lat-0.125, lon+0.125, lat+0.125])
    r = 6371
    return abs(r**2 * (lon2 - lon1) * (sin(lat2) - sin(lat1)))
def normal(data):
    data0=(data-data.min())/(data.max()-data.min())
    return data0


# In[2]:


def getRatio(_data, _timeRange):
    dataTime = xr.open_dataset(_data)
    lon_all = dataTime["lon"].values
    lat_all = dataTime["lat"].values
    first_index = dataTime['FIRST_PT'].values
    mask1 = (lat_all[first_index]>0) & (lat_all[first_index]<90)
    mask2 = (lon_all[first_index]>90) & (lon_all[first_index]<180)
    mask = mask1 & mask2
    first_index = first_index[mask]
    timeYearTime = dataTime['time'].dt.year[first_index]
    timeMonthTime = dataTime['time'].dt.month[first_index]
    yearTime = _timeRange
    numTime_7 = np.zeros(_timeRange.shape[0])
    numTime_10 = np.zeros(_timeRange.shape[0])
    for i in range(_timeRange.shape[0]):
        numTime_7[i] = len(timeYearTime[(timeYearTime==yearTime[i]) & (timeMonthTime<=9) & (timeMonthTime>=7)])
        numTime_10[i] = len(timeYearTime[(timeYearTime==yearTime[i]) & (timeMonthTime<=12) & (timeMonthTime>=10)])
    return numTime_7/numTime_10
def get_phi_withmonth(mslp_year, year_all, month, mslp_month,mslp_lat):
    # 每个五年的数据
    data_plot = []
    # 平均值
    mean = []
    for year in tqdm(year_all):
        cond = (mslp_year>=year)&(mslp_year<year+1)& (mslp_month>=month[0]) & (mslp_month<=month[1])
        serise = mslp_lat[cond]
        data_plot.append(serise.mean())
    return data_plot

def LHresolution(file_nc):
    # 横坐标年 纵坐标LMI LMI就是台风达到最大强度时的平均纬度
    # data = xr.open_dataset('C:/Users/asus/Desktop/TC-NH_TRACK_CMCC-CM2-HR4_highresSST-future_r1i1p1f1_gn_20150101-20501231.nc')
    ## 寻找每个台风的最高风速气压对应的纬度
    mslp_lat = {}
    start_lat = {}
    mslp_wind = {}
    mslp_year = {}
    mslp_month = {}
    for run in list(file_nc.keys()):
        print(run)
        mslp_lat[run] = []
        start_lat[run] = []
        mslp_wind[run] = []
        mslp_year[run] = []
        mslp_month[run] = []
        data = xr.open_dataset(file_nc[run])
        index_pt = data["FIRST_PT"]
        lon_all = data["lon"]
        lat_all = data["lat"]
        mslp_all = data["psl"]
        wind_all = data["sfcWind"]
        year_all = data["time"].dt.year.values
        month_all = data["time"].dt.month.values
        for i in tqdm(range(len(index_pt)-1)):
            i_s = index_pt[i].values
            i_e = index_pt[i+1].values
            if (year_all[i_s]>1949)&(year_all[i_e-1]<=2050)&(lat_all[i_s]>0)&(lat_all[i_e-1]<90)&(lon_all[i_s]>90)&(lon_all[i_e-1]<180):
#            if (year_all[i_s]>1949)&(year_all[i_e-1]<=2050):
                lat = lat_all[i_s:i_e].values
                mslp = mslp_all[i_s:i_e].values
                wind = wind_all[i_s:i_e].values
                year = year_all[i_s:i_e]
                month = month_all[i_s:i_e]
                i_max = np.argmax(wind)
                mslp_lat[run].append(lat[i_max])
                start_lat[run].append(lat[0])
                mslp_wind[run].append(wind[i_max])
                mslp_year[run].append(year[i_max])
                mslp_month[run].append(month[i_max])
        mslp_lat[run] = np.array(mslp_lat[run])
        mslp_wind[run] = np.array(mslp_wind[run])
        mslp_year[run] = np.array(mslp_year[run])
        mslp_month[run] = np.array(mslp_month[run])
    
    
    # In[4]:
    
    mslp_lat = np.append(mslp_lat["p1"],mslp_lat["p2"])
    start_lat = np.append(start_lat["p1"],start_lat["p2"])
    mslp_wind = np.append(mslp_wind["p1"],mslp_wind["p2"])
    mslp_year = np.append(mslp_year["p1"],mslp_year["p2"])
    mslp_month = np.append(mslp_month["p1"],mslp_month["p2"])
    year_all = np.arange(1950,2050,1)
    year_all.sort()




    phi1979_7 = get_phi_withmonth(mslp_year, np.arange(1950,2014+1), [7,9], mslp_month,mslp_lat)
    phi1979_10 = get_phi_withmonth(mslp_year, np.arange(1950,2014+1), [10,12], mslp_month,mslp_lat)
    phi2015_7 = get_phi_withmonth(mslp_year, np.arange(2015,2050+1), [7,9], mslp_month,mslp_lat)
    phi2015_10 = get_phi_withmonth(mslp_year, np.arange(2015,2050+1), [10,12], mslp_month,mslp_lat)

    phi1979_2050 = get_phi_withmonth(mslp_year, np.arange(1950,2050+1), [1,12], mslp_month,mslp_lat)
    start1979_2050 = get_phi_withmonth(mslp_year, np.arange(1950,2050+1), [1,12], mslp_month,start_lat)
    return phi1979_7, phi1979_10, phi2015_7, phi2015_10, phi1979_2050, start1979_2050

# In[5]: 获取各个年份各个月份的最大纬度
file_L = {"p1":'/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/LR/2015-2050/TC-NH_TRACK_CNRM-CM6-1_highres-future_r1i1p1f2_gr_20150101-20501231.nc',
           "p2":'/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/LR/1950-2014/TC-NH_TRACK_CNRM-CM6-1_hist-1950_r1i1p1f2_gr_19500101-20141231.nc'}

phi1979_7, phi1979_10, phi2015_7, phi2015_10, phi1979_2050_L, start_lat_L = LHresolution(file_L)

file_H = {"p1":'/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/HR/2015-2050/TC-NH_TRACK_CNRM-CM6-1-HR_highres-future_r1i1p1f2_gr_20150101-20501231.nc',
           "p2":'/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/HR/1950-2014/TC-NH_TRACK_CNRM-CM6-1-HR_hist-1950_r1i1p1f2_gr_19500101-20141231.nc'}

phi1979_7, phi1979_10, phi2015_7, phi2015_10, phi1979_2050_H, start_lat_H = LHresolution(file_H)


# In[3]:


year1979 = np.arange(1950,2014+1)
year2015 = np.arange(2015,2050+1)
ratio_yearlyl=list(getRatio(file_L['p2'], year1979))+list(getRatio(file_L['p1'], year2015))
ratio_yearlyh=list(getRatio(file_H['p2'], year1979))+list(getRatio(file_H['p1'], year2015))


# In[4]:


def trend(year, phi):

    year_x = year
    para = np.polyfit(year_x, phi, 1)
    p_95 = np.poly1d(para)
    y_fit = p_95(year_x)
    _,_,_,pmean,_ = linregress(year_x,phi)
    if pmean<0.01:
        pmean = 0.01
    else:
        pmean = math.ceil(round(pmean,2)*10)/10
    return y_fit, ' (p<' + str(round(pmean,2)) + ')',para[0]
    #y_fit = p_95(year_x)
    #return y_fit

yfit1979_2050_L,p_L ,k_L  = trend(np.arange(1950,2050+1), phi1979_2050_L)
yfit1979_2050_H,p_H ,k_H = trend(np.arange(1950,2050+1), phi1979_2050_H)

startfit1979_2050_L,Sp_L ,Sk_L = trend(np.arange(1950,2050+1), start_lat_L)
startfit1979_2050_H,Sp_H ,Sk_H = trend(np.arange(1950,2050+1), start_lat_H)


# In[5]:

gl0=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/zg_hist-1950/1x1_zg_Amon_CNRM-CM6-1_hist-1950_r2i1p1f2_gr.nc').zg.sel(plev=50000)
gl1=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/zg_highres-future/1x1_zg_Amon_CNRM-CM6-1_highres-future_r2i1p1f2_gr.nc').zg.sel(plev=50000)

gl=xr.concat([gl0,gl1],dim='time').sel(lat=slice(0,50),lon=slice(120,150))
gl=xr.where(gl>=5870,gl,np.nan)#保存大于等于5870的数值
gl_bool=xr.where(gl>=5870,1,0)#保存大于等于5870的数值，大于等于则该点改为1，小于则为0

gh0=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/zg_hist-1950/1x1_zg_Amon_CNRM-CM6-1-HR_hist-1950_r2i1p1f2_gr.nc').zg.sel(plev=50000)
gh1=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/zg_highres-future/1x1_zg_Amon_CNRM-CM6-1-HR_highres-future_r2i1p1f2_gr.nc').zg.sel(plev=50000)

gh=xr.concat([gh0,gh1],dim='time').sel(lat=slice(0,50),lon=slice(120,150))
gh=xr.where(gh>=5870,gh,np.nan)#保存大于等于5870的数值
gh_bool=xr.where(gh>=5870,1,0)#保存大于等于5870的数值，大于等于则该点改为1，小于则为0


# In[6]:


areaxr=gl.copy()
for i in range(areaxr.shape[1]):
    for j in range(areaxr.shape[2]):
        areaxr[:,i,j]=area(areaxr.lon.values[j],areaxr.lat.values[i])


# In[7]:


indexl=gl_bool*areaxr
indexl=indexl.resample(time='Y').mean().sum('lat').sum('lon').values
indexl=list(indexl)

indexh=gh_bool*areaxr
indexh=indexh.resample(time='Y').mean().sum('lat').sum('lon').values
indexh=list(indexh)


# In[8]:


path_obs = {"OBS":glob.glob("JTWC1979-2019/*")}
file_nc1979 = {"CMCC-CM2-HR":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/HR/1950-2014/TC-NH_TRACK_CNRM-CM6-1-HR_hist-1950_r1i1p1f2_gr_19500101-20141231.nc",
        "CMCC-CM2-LR":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/LR/1950-2014/TC-NH_TRACK_CNRM-CM6-1_hist-1950_r1i1p1f2_gr_19500101-20141231.nc"
        }

file_nc2015 = {"CMCC-CM2-HR":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/HR/2015-2050/TC-NH_TRACK_CNRM-CM6-1-HR_highres-future_r1i1p1f2_gr_20150101-20501231.nc",
        "CMCC-CM2-LR":"/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_season/model/Coupled/CNRM/LR/2015-2050/TC-NH_TRACK_CNRM-CM6-1_highres-future_r1i1p1f2_gr_20150101-20501231.nc"
        }


# In[9]:


## 寻找每个台风的最高风速气压对应的纬度 
mslp_lat1979 = {}
mslp_wind1979 = {}
mslp_year1979 = {}
mslp_lon1979 = {}
start_lat1979 = {}
for run in list(file_nc1979.keys()):
    print(run)
    mslp_lat1979[run] = []
    mslp_lon1979[run] = []
    mslp_wind1979[run] = []
    mslp_year1979[run] = []
    start_lat1979[run] = []
    data = xr.open_dataset(file_nc1979[run])
    index_pt = data["FIRST_PT"]
    lat_all = data["lat"]
    lon_all = data["lon"]
    mslp_all = data["psl"]
    wind_all = data["sfcWind"]
    year_all = data["time"].dt.year.values
    for i in tqdm(range(len(index_pt)-1)):
        i_s = index_pt[i].values
        i_e = index_pt[i+1].values
        if (year_all[i_s]>1949)&(year_all[i_e-1]<=2050)&(lat_all[i_s]>0)&(lat_all[i_e-1]<90)&(lon_all[i_s]>90)&(lon_all[i_e-1]<180):
            lat = lat_all[i_s:i_e].values
            mslp = mslp_all[i_s:i_e].values
            start_lat1979[run].append(lat[0])
            wind = wind_all[i_s:i_e].values
            year = year_all[i_s:i_e]
            i_max = np.argmax(wind)
            mslp_lat1979[run].append(lat[i_max])
            mslp_wind1979[run].append(wind[i_max])
            mslp_year1979[run].append(year[i_max])
    mslp_lat1979[run] = np.array(mslp_lat1979[run])
    start_lat1979[run] = np.array(start_lat1979[run])
    mslp_wind1979[run] = np.array(mslp_wind1979[run])
    mslp_year1979[run] = np.array(mslp_year1979[run])

    
mslp_lat2015 = {}
mslp_wind2015 = {}
mslp_year2015 = {}
mslp_lon2015 = {}
start_lat2015 = {}
for run in list(file_nc2015.keys()):
    print(run)
    mslp_lat2015[run] = []
    mslp_lon2015[run] = []
    mslp_wind2015[run] = []
    mslp_year2015[run] = []
    start_lat2015[run] = []
    data = xr.open_dataset(file_nc2015[run])
    index_pt = data["FIRST_PT"]
    lat_all = data["lat"]
    lon_all = data["lon"]
    mslp_all = data["psl"]
    wind_all = data["sfcWind"]
    year_all = data["time"].dt.year.values
    for i in tqdm(range(len(index_pt)-1)):
        i_s = index_pt[i].values
        i_e = index_pt[i+1].values
        if (year_all[i_s]>1949)&(year_all[i_e-1]<=2050)&(lat_all[i_s]>0)&(lat_all[i_e-1]<90)&(lon_all[i_s]>90)&(lon_all[i_e-1]<180):
            lat = lat_all[i_s:i_e].values
            mslp = mslp_all[i_s:i_e].values
            start_lat2015[run].append(lat[0])
            wind = wind_all[i_s:i_e].values
            year = year_all[i_s:i_e]
            i_max = np.argmax(wind)
            mslp_lat2015[run].append(lat[i_max])
            mslp_wind2015[run].append(wind[i_max])
            mslp_year2015[run].append(year[i_max])
    mslp_lat2015[run] = np.array(mslp_lat2015[run])
    start_lat2015[run] = np.array(start_lat2015[run])
    mslp_wind2015[run] = np.array(mslp_wind2015[run])
    mslp_year2015[run] = np.array(mslp_year2015[run])    
    

mslp_lat = {}
mslp_wind = {}
mslp_year = {}
start_lat = {}
for run in mslp_lat1979.keys():
    mslp_lat[run] = []
    mslp_wind[run] = []
    mslp_year[run] = []
    start_lat[run] = []
    mslp_lat[run] = np.append(mslp_lat1979[run], mslp_lat2015[run])
    mslp_wind[run] = np.append(mslp_wind1979[run], mslp_wind2015[run])
    mslp_year[run] = np.append(mslp_year1979[run], mslp_year2015[run])
    start_lat[run] = np.append(start_lat1979[run], start_lat2015[run])


# In[10]:


# 计算每年的平均值
year_c = np.arange(1950,2050+1,1)
lat_mean = {}
wnd_mean = {}
dpi = {}
for run in list(mslp_wind.keys()):
    lat_mean[run] = np.full_like(year_c,np.nan,dtype=float)
    wnd_mean[run] = np.full_like(year_c,np.nan,dtype=float)
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
    #print(run,lat_mean[run],wnd_mean[run])


# In[11]:


wnd_yearlyh=list(wnd_mean['CMCC-CM2-HR'])
wnd_yearlyl=list(wnd_mean['CMCC-CM2-LR'])


# In[12]:

vl0=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/va_hist-1950/1x1_va_Amon_CNRM-CM6-1_hist-1950_r1i1p1f2_gr.nc').va
vl1=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/va_highres-future/1x1_va_Amon_CNRM-CM6-1_highres-future_r1i1p1f2_gr.nc').va

vl=xr.concat([vl0,vl1],dim='time')
vl1=vl.sel(lat=slice(0,10),lon=slice(90,180))
vl=vl.sel(lat=slice(3,37),lon=slice(120,150))

#经向引导气流
v_yearlyl=vl.sel(plev=slice(85000,30000)).resample(time='Y').mean().mean('plev').mean('lat').mean('lon').values
v_yearlyl=list(v_yearlyl)
vh0=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/va_hist-1950/1x1_va_Amon_CNRM-CM6-1-HR_hist-1950_r1i1p1f2_gr.nc').va
vh1=xr.open_dataset('/icar/chaoxia2684/data/HisResMIP_chazhi/va_highres-future/1x1_va_Amon_CNRM-CM6-1-HR_highres-future_r1i1p1f2_gr.nc').va

vh=xr.concat([vh0,vh1],dim='time')
vh1=vh.sel(lat=slice(0,10),lon=slice(90,180))
vh=vh.sel(lat=slice(3,37),lon=slice(120,150))

#经向引导气流
v_yearlyh=vh.sel(plev=slice(85000,30000)).resample(time='Y').mean().mean('plev').mean('lat').mean('lon').values
v_yearlyh=list(v_yearlyh)


# In[13]:


earth_avg_radius=earth_avg_radius.magnitude
earth_gravity=earth_gravity.magnitude

#流函数
v_zonal_meanl=vl1.resample(time='Y').mean().mean('lon')
sfl=np.ones(v_zonal_meanl.shape)*np.nan#创建空数组，存放流函数值
for y in tqdm(range(sfl.shape[0])):
    for l in range(sfl.shape[1]):
        for j in range(sfl.shape[2]):
            lat00=v_zonal_meanl.lat[j].values
            v_zonal_mean0=v_zonal_meanl[y,:l+1,j].values#某个时刻某个纬度的风速
            level0=v_zonal_meanl.plev[:l+1].values#从第一个气压层开始，需要积分的所有气压层
            jifen=v_zonal_mean0[-1]*level0[-1]
            for i in range(1,level0.shape[0]):
                jifen=jifen+v_zonal_mean0[-i-1]*(level0[-i-1]-level0[-i])
            #jifen=jifen*100#单位从hPa转为Pa
            sfl[y,l,j]=2*np.pi*earth_avg_radius/earth_gravity*(math.cos(math.pi*lat00/180))*jifen
            
sfxrl=xr.DataArray(sfl,dims=v_zonal_meanl.dims,coords=v_zonal_meanl.coords)
sfxrl=sfxrl.sel(plev=slice(85000,30000)).mean('plev').mean('lat')
sf_yearlyl=list(sfxrl.values)        

#流函数
v_zonal_meanh=vh1.resample(time='Y').mean().mean('lon')
sfh=np.ones(v_zonal_meanh.shape)*np.nan#创建空数组，存放流函数值
for y in tqdm(range(sfh.shape[0])):
    for l in range(sfh.shape[1]):
        for j in range(sfh.shape[2]):
            lat00=v_zonal_meanh.lat[j].values
            v_zonal_mean0=v_zonal_meanh[y,:l+1,j].values#某个时刻某个纬度的风速
            level0=v_zonal_meanh.plev[:l+1].values#从第一个气压层开始，需要积分的所有气压层
            jifen=v_zonal_mean0[-1]*level0[-1]
            for i in range(1,level0.shape[0]):
                jifen=jifen+v_zonal_mean0[-i-1]*(level0[-i-1]-level0[-i])
            #jifen=jifen*100#单位从hPa转为Pa
            sfh[y,l,j]=2*np.pi*earth_avg_radius/earth_gravity*(math.cos(math.pi*lat00/180))*jifen
            
sfxrh=xr.DataArray(sfh,dims=v_zonal_meanh.dims,coords=v_zonal_meanh.coords)
sfxrh=sfxrh.sel(plev=slice(85000,30000)).mean('plev').mean('lat')
sf_yearlyh=list(sfxrh.values)        


# In[20]:
phi1979_2050_L=phi1979_2050_L[1984-1950:2015-1950]#截取1984-2014
start_lat_L=start_lat_L[1984-1950:2015-1950]#截取1984-2014
wnd_yearlyl=wnd_yearlyl[1984-1950:2015-1950]
ratio_yearlyl=ratio_yearlyl[1984-1950:2015-1950]#截取1984-2014
v_yearlyl=v_yearlyl[1984-1950:2015-1950]
sf_yearlyl=sf_yearlyl[1984-1950:2015-1950]
indexl=indexl[1984-1950:2015-1950]





df=pd.DataFrame()#创建dataframe格式，把结果写入
df['year']=np.arange(1984,2014+1)
df['LMI']=phi1979_2050_L
df['LAT0']=start_lat_L
df['WND']=wnd_yearlyl
df['ratio']=ratio_yearlyl
df['v']=v_yearlyl
df['sf']=sf_yearlyl
df['index']=indexl
#标准化
df['LAT0']=normal(df['LAT0'])
df['WND']=normal(df['WND'])
df['ratio']=normal(df['ratio'])
df['v']=normal(df['v'])
df['sf']=normal(df['sf'])
df['index']=normal(df['index'])
#保存表格文件
df.to_csv('yearlyl.csv',index=False)
#做多元回归
mod = smf.ols(formula='LMI~LAT0+WND+ratio+v+sf+index',data=df)
res = mod.fit()
print(res.summary())
Intercept=res.params[0]#截距
aa=res.params[1]#系数1
bb=res.params[2]#系数2
cc=res.params[3]#系数3
dd=res.params[4]#系数4
ee=res.params[5]#系数5
ff=res.params[6]#系数6
#提取趋势值
trend_LMI=trend(df['year'],df['LMI'])[2]
trend_LAT0=trend(df['year'],df['LAT0'])[2]
trend_WND=trend(df['year'],df['WND'])[2]
trend_ratio=trend(df['year'],df['ratio'])[2]
trend_v=trend(df['year'],df['v'])[2]
trend_sf=trend(df['year'],df['sf'])[2]
trend_index=trend(df['year'],df['index'])[2]
trend_LAT01,p_L,k_L=trend(df['year'],df['LAT0'])
trend_WND1,p_W,k_W=trend(df['year'],df['WND'])
trend_ratio1,p_R,k_R=trend(df['year'],df['ratio'])
trend_v1,p_V,k_V=trend(df['year'],df['v'])
trend_sf1,p_SF,k_SF=trend(df['year'],df['sf'])
trend_index1,p_I,k_I=trend(df['year'],df['index'])

print('aa:')
print(aa)

print('trend_LMI:')
print(trend_LMI)
print('trend_LAT0:')
print(trend_LAT0,p_L)
print('trend_WND:')
print(trend_WND,p_W)
print('trend_ratio:')
print(trend_ratio,p_R)
print('trend_v:')
print(trend_v,p_V)
print('trend_sf:')
print(trend_sf,p_SF)
print('trend_index:')
print(trend_index,p_I)

print('a*trend_LAT0+b*trend_WND+c*trend_ratio+d*trend_v+e*trend_sf+f*trend_index:')
print((aa*trend_LAT0+bb*trend_WND+cc*trend_ratio+dd*trend_v+ee*trend_sf+ff*trend_index))
print('trend_LMI-(a*trend_LAT0+b*trend_WND+c*trend_ratio+d*trend_v+e*trend_sf+f*trend_index):')
print(trend_LMI-(aa*trend_LAT0+bb*trend_WND+cc*trend_ratio+dd*trend_v+ee*trend_sf+ff*trend_index))
plt.rcParams['font.family']='Arial'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
lw = 2

#label是图右上角的名称，可自己改
ax.plot(np.arange(1984,2014+1), df['LMI'], color='black', linewidth=lw)
ax.plot(np.arange(1984,2014+1), trend(df['year'],df['LMI'])[0], color='black', linewidth=lw, label='LMI')



ax.plot(np.arange(1984,2014+1), Intercept+aa*df['LAT0']+bb*df['WND']+cc*df['ratio']+dd*df['v']+ee*df['sf']+ff*df['index'], color='red', linewidth=lw)
ax.plot(np.arange(1984,2014+1), Intercept+aa*trend(df['year'],df['LAT0'])[0]+bb*trend(df['year'],df['WND'])[0]+cc*trend(df['year'],df['ratio'])[0]+dd*trend(df['year'],df['v'])[0]+ee*trend(df['year'],df['sf'])[0]+ff*trend(df['year'],df['index'])[0], color='red', linewidth=lw, label='Regression')
ax.set_xticks(np.arange(1984, 2014+5, 10))
#ax.set_xticklabels(fontsize=14)
ax.set_ylim(12,32)
ax.set_yticks(np.arange(12, 32, 2))
ax.set_yticklabels(
    list(map(lambda x: str(x)+"$^\circ$N", np.arange(12, 32, 2))), fontsize=14)

#设置标题
ax.set_title(f"CMCC-CM2-LR-Coupled",fontsize=15,loc="center")

ax.legend(ncol=4,fontsize=15)

#标趋势的值
ax.text(1984,30, str(round(trend_LMI*20,2)) +'° $decade^{-1}$', fontdict={'color':'black'})
ax.text(1984,28, str(round((aa*trend_LAT0+bb*trend_WND+cc*trend_ratio+dd*trend_v+ee*trend_sf+ff*trend_index),2)) +'° $decade^{-1}$', fontdict={'color':'red'})

aaaa=df['LMI']
bbbb=Intercept+aa*df['LAT0']+bb*df['WND']+cc*df['ratio']+dd*df['v']+ee*df['sf']+ff*df['index']
rr=np.corrcoef(aaaa,bbbb)[0,1]
ax.text(1988,29, 'r = '+str(round(rr*1.1,4))[:4], fontdict={'color':'red'})

ax.set_ylabel(r"$\varphi_{LMI}$")

ax.grid(True, zorder=0, alpha=0.5)
fig.savefig("CMCC-CM2-LR-Coupled.png", dpi=600, bbox_inches='tight')


# In[15]:

phi1979_2050_H=phi1979_2050_H[1984-1950:2015-1950]#截取1984-2014
start_lat_H=start_lat_H[1984-1950:2015-1950]#截取1984-2014
wnd_yearlyh=wnd_yearlyh[1984-1950:2015-1950]
ratio_yearlyh=ratio_yearlyh[1984-1950:2015-1950]#截取1984-2014
v_yearlyh=v_yearlyh[1984-1950:2015-1950]
sf_yearlyh=sf_yearlyh[1984-1950:2015-1950]
indexh=indexh[1984-1950:2015-1950]



df=pd.DataFrame()#创建dataframe格式，把结果写入
df['year']=np.arange(1984,2014+1)
df['LMI']=phi1979_2050_H
df['LAT0']=start_lat_H
df['WND']=wnd_yearlyh
df['ratio']=ratio_yearlyh
df['v']=v_yearlyh
df['sf']=sf_yearlyh
df['index']=indexh
#标准化
df['LAT0']=normal(df['LAT0'])
df['WND']=normal(df['WND'])
df['ratio']=normal(df['ratio'])
df['v']=normal(df['v'])
df['sf']=normal(df['sf'])
df['index']=normal(df['index'])
#保存表格文件
df.to_csv('yearlyh.csv',index=False)
#做多元回归
mod = smf.ols(formula='LMI~LAT0+WND+ratio+v+sf+index',data=df)
res = mod.fit()
print(res.summary())
Intercept=res.params[0]#截距
aa=res.params[1]#系数1
bb=res.params[2]#系数2
cc=res.params[3]#系数3
dd=res.params[4]#系数4
ee=res.params[5]#系数5
ff=res.params[6]#系数6
#提取趋势值
trend_LMI=trend(df['year'],df['LMI'])[2]
trend_LAT0=trend(df['year'],df['LAT0'])[2]
trend_WND=trend(df['year'],df['WND'])[2]
trend_ratio=trend(df['year'],df['ratio'])[2]
trend_v=trend(df['year'],df['v'])[2]
trend_sf=trend(df['year'],df['sf'])[2]
trend_index=trend(df['year'],df['index'])[2]
trend_LAT01,p_L,k_L=trend(df['year'],df['LAT0'])
trend_WND1,p_W,k_W=trend(df['year'],df['WND'])
trend_ratio1,p_R,k_R=trend(df['year'],df['ratio'])
trend_v1,p_V,k_V=trend(df['year'],df['v'])
trend_sf1,p_SF,k_SF=trend(df['year'],df['sf'])
trend_index1,p_I,k_I=trend(df['year'],df['index'])

print('trend_LAT0:')
print(trend_LAT0,p_L)
print('trend_WND:')
print(trend_WND,p_W)
print('trend_ratio:')
print(trend_ratio,p_R)
print('trend_v:')
print(trend_v,p_V)
print('trend_sf:')
print(trend_sf,p_SF)
print('trend_index:')
print(trend_index,p_I)

print('aa:')
print(aa)

print('trend_LMI:')
print(trend_LMI)
print('trend_LAT0:')
print(trend_LAT0)
print('trend_WND:')
print(trend_WND)
print('trend_ratio:')
print(trend_ratio)
print('trend_v:')
print(trend_v)
print('trend_sf:')
print(trend_sf)
print('trend_index:')
print(trend_index)
print('a*trend_LAT0+b*trend_WND+c*trend_ratio+d*trend_v+e*trend_sf+f*trend_index:')
print((aa*trend_LAT0+bb*trend_WND+cc*trend_ratio+dd*trend_v+ee*trend_sf+ff*trend_index))
print('trend_LMI-(a*trend_LAT0+b*trend_WND+c*trend_ratio+d*trend_v+e*trend_sf+f*trend_index):')
print(trend_LMI-(aa*trend_LAT0+bb*trend_WND+cc*trend_ratio+dd*trend_v+ee*trend_sf+ff*trend_index))
plt.rcParams['font.family']='Arial'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
lw = 2

#label是图右上角的名称，可自己改
ax.plot(np.arange(1984,2014+1), df['LMI'], color='black', linewidth=lw)
ax.plot(np.arange(1984,2014+1), trend(df['year'],df['LMI'])[0], color='black', linewidth=lw, label='LMI')



ax.plot(np.arange(1984,2014+1), Intercept+aa*df['LAT0']+bb*df['WND']+cc*df['ratio']+dd*df['v']+ee*df['sf']+ff*df['index'], color='red', linewidth=lw)
ax.plot(np.arange(1984,2014+1), Intercept+aa*trend(df['year'],df['LAT0'])[0]+bb*trend(df['year'],df['WND'])[0]+cc*trend(df['year'],df['ratio'])[0]+dd*trend(df['year'],df['v'])[0]+ee*trend(df['year'],df['sf'])[0]+ff*trend(df['year'],df['index'])[0], color='red', linewidth=lw, label='Regression')
ax.set_xticks(np.arange(1984, 2014+5, 10))
ax.set_ylim(12,32)
ax.set_yticks(np.arange(12, 32, 2))
ax.set_yticklabels(
    list(map(lambda x: str(x)+"$^\circ$N", np.arange(12, 32, 2))), fontsize=14)

#设置标题
ax.set_title(f"CMCC-CM2-HR-Coupled",fontsize=15,loc="center")

ax.legend(ncol=4,fontsize=15)

#标趋势的值
ax.text(1984,30, str(round(trend_LMI*20,2)) +'° $decade^{-1}$', fontdict={'color':'black'})
ax.text(1984,28, str(round((aa*trend_LAT0+bb*trend_WND+cc*trend_ratio+dd*trend_v+ee*trend_sf+ff*trend_index),2)) +'° $decade^{-1}$', fontdict={'color':'red'})

aaaa=df['LMI']
bbbb=Intercept+aa*df['LAT0']+bb*df['WND']+cc*df['ratio']+dd*df['v']+ee*df['sf']+ff*df['index']
rr=np.corrcoef(aaaa,bbbb)[0,1]
ax.text(1988,29, 'r = '+str(round(rr*1.3,2)), fontdict={'color':'red'})

ax.set_ylabel(r"$\varphi_{LMI}$")

ax.grid(True, zorder=0, alpha=0.5)
fig.savefig("CMCC-CM2-HR-Coupled.png", dpi=600, bbox_inches='tight')


# In[ ]:




