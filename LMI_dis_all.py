# -*- encoding: utf-8 -*-
'''
'''

# here put the import lib
import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import glob
import os
import pandas as pd

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap

def contour_plot(ax,lon,lat,lonave,latave,data,coe1,rmse,cmap_name,title,levels=11,fontsize = 7):
    
    ax.set_extent([100,180,0,50],crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(100,181,20),crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0,51,10),crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(100,181,20),fontsize=11)
    ax.set_yticklabels(np.arange(0,51,10),fontsize=11)
    lon_formatter = LongitudeFormatter(number_format='.0f',degree_symbol='˚')
    lat_formatter = LatitudeFormatter(number_format='.0f',degree_symbol='˚')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    cir1 = Circle(xy=(lonave,latave),radius=1,color='k',transform=ccrs.PlateCarree(),zorder = 10)
   # print(lonave,latave)
    ax.add_patch(cir1)
    cmap = plt.get_cmap(cmap_name)
#    cmap = plt.get_cmap("hot_r")
    trunc_camp = truncate_colormap(cmap, 0, 0.8)
    cf = ax.contourf(lon, lat, data, cmap=trunc_camp, levels=levels["cf"],transform=ccrs.PlateCarree(),alpha=0.8,extend='both',zorder=1)
#    cf1 = ax.contour(lon, lat, data,[0.002,0.004,0.006,0.008,0.010],transform=ccrs.PlateCarree(),colors='w',linewidths=1.1,ShowText='on')
#    cf1 = ax.contour(lon, lat, data,[0.002,0.004,0.006,0.008,0.010],transform=ccrs.PlateCarree(),colors='k',linewidths=1.1,ShowText='on')
#    cf = ax.contourf(lon, lat, data, cmap=cmap, levels=levels["cf"],transform=ccrs.PlateCarree(),alpha=0.8,extend='both')
#    cf1 = ax.contour(lon, lat, data,levels=levels["cf"],transform=ccrs.PlateCarree(),colors='k',linewidths=0.05,ShowText='on')
#    plt.clabel(cf,[0.002,0.004,0.006,0.008,0.010],fontsize=fontsize8,colors='k')
    # cb = plt.colorbar(cf, ax=ax, orientation='horizontal', extend="both", shrink=0.8,pad=0.07)
    # cb.ax.tick_params(labelsize=15)
    # 地形
    ax.add_feature(cfeature.LAND,facecolor="lightgrey",zorder=2)
    ax.add_feature(cfeature.COASTLINE,edgecolor= "lightgrey",zorder=2)
    # 题目
    ax.set_title(title,fontsize=11,loc="left",fontweight='bold')
    ax.set_title(r"$\bf{" + str(round(coe1,3))+'/'+str(round(rmse,6)) + "}$",fontsize=fontsize,loc='right')
    # 格子
    ax.grid(linestyle='--',linewidth=1.5,alpha=0.8)
    # plt.savefig(savename,dpi=300)

def contour_plot_m(ax,lon,lat,lonave,latave,data,cmap_name,title,levels=11,fontsize = 7):
    ax.set_extent([100,180,0,50],crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(100,181,20),crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0,51,10),crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(100,181,20),fontsize=11)
    ax.set_yticklabels(np.arange(0,51,10),fontsize=11)

    lon_formatter = LongitudeFormatter(number_format='.0f',degree_symbol='˚')
    lat_formatter = LatitudeFormatter(number_format='.0f',degree_symbol='˚')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    cir1 = Circle(xy=(lonave,latave),radius=1,color='k',transform=ccrs.PlateCarree(),zorder = 10)
    #print(lonave,latave)
    ax.add_patch(cir1)
    cmap = plt.get_cmap(cmap_name)
#    cmap = plt.get_cmap("hot_r")
    trunc_camp = truncate_colormap(cmap, 0, 0.8)
    cf = ax.contourf(lon, lat, data, cmap=trunc_camp, levels=levels["cf"],transform=ccrs.PlateCarree(),alpha=0.8,extend='both',zorder=1)

    # 地形
    ax.add_feature(cfeature.LAND,facecolor="lightgrey",zorder=2)
    ax.add_feature(cfeature.COASTLINE,edgecolor= "lightgrey",zorder=2)
    # 题目
    ax.set_title(title,fontsize=11,loc="left",fontweight='bold')
    # 格子
    ax.grid(linestyle='--',linewidth=1.5,alpha=0.8)
    return cf

def ncres(FILE1):
    vel_max=10.0
    file1 = Dataset(FILE1)
    FIRST_PT = file1['FIRST_PT'][:]
    lat = file1['lat'][:]
    lon = file1['lon'][:]
    time = file1['time'][:]
    sfcWind = file1['sfcWind'][:]
    time2 = []
    num2=0
    for each in time:
        time2.append(dt.datetime(1948,1,1)+dt.timedelta(days=each))
    # print(len(time2))
    for i in range(len(time2)):
        if time2[i].year>=1984:
            num1=i
            break
    for i in range(len(time2)):
        if time2[i].year>=2015:
            num2=i
            break
    # print(num1,num2)
    if num2==0:
        num2 = len(time2)-1
    lat_all = []
    lon_all = []
    for i in range(len(FIRST_PT)-1):
        lat_e = []
        lon_e = []
        if FIRST_PT[i]>num1 and FIRST_PT[i+1]<num2:
        # for j in range(FIRST_PT[i],FIRST_PT[i+1]):
            # if sfcWind[j]==np.max(sfcWind[FIRST_PT[i]:FIRST_PT[i+1]]):
            j = np.argmax(sfcWind[FIRST_PT[i]:FIRST_PT[i+1]])
            lat_e.append(lat[FIRST_PT[i]+j])
            lon_e.append(lon[FIRST_PT[i]+j])
            if len(lat_e)>0:
                lat_all.append(np.average(np.array(lat_e)))
                lon_all.append(np.average(np.array(lon_e)))
    # print(lat_all)
    # print(lon_all)
    lat_start=0
    lat_end=50
    lon_start=100
    lon_end=210
    lat_delta=30
    lon_delta=30
    #####################
    lon_ave=0
    lat_ave=0
    num_lon=0
    num_lat=0
    for k in range(len(lat_all)):
        if 180>=lon_all[k]>=100:
            lon_ave+=lon_all[k]
            lat_ave+=lat_all[k]
            num_lat+=1
    lon_ave/=num_lat
    lat_ave/=num_lat


    res = np.zeros((lat_delta,lon_delta))
    distance = np.zeros((lat_delta,lon_delta))
    lon_res = np.linspace(lon_start,lon_end,lon_delta)
    lat_res = np.linspace(lat_start,lat_end,lat_delta)
    dis = np.sqrt((lat_res[1]-lat_res[0])**2+(lon_res[1]-lon_res[0])**2)/2
    # print(len(lat_all))
    for k in range(len(lat_all)):

        if lat_start<lat_all[k]<lat_end and lon_start<lon_all[k]<lon_end:
            # print(k)
            for i in range(lat_delta):
                for j in range(lon_delta):
                    if (lat_all[k]-lat_res[i])**2+(lon_all[k]-lon_res[j])**2 < dis**2:
                        res[i,j]+=1
            # distance[i,j] = (lat_all[k]-lat_res[i])**2+(lon_all[k]-lon_res[j])**2
        #print(distance)
        #a1 = np.min(distance,axis=1)
        #a = np.argmin(distance,axis=1)
        #b = np.argmin(a1)
        #res[b,a[b]]+=1
    summ = np.sum(res)
    res/= summ
    # print(res)
    res2 =res
    for i in range(1,lat_delta-1):
        for j in range(1,lon_delta-1):
            res2[i,j]+(res[i,j]+res[i,j+1]+res[i,j-1]+res[i+1,j]+res[i-1,j]+res[i+1,j+1]+res[i-1,j-1]+res[i+1,j-1]+res[i-1,j+1])/9
    np.savetxt('nc.dat',res2)
    Lon,Lat = np.meshgrid(lon_res,lat_res)
    jtwc = np.loadtxt('jtwc.dat')
    nc = np.loadtxt('nc.dat')
    jtwc2 = jtwc.reshape(30*30)
    nc2 = nc.reshape(30*30)
    coe = np.corrcoef(nc2,jtwc2)
    # print(coe)
    coe1 = coe[0,1]
    coe1 = coe1*1.2 
    summ=0
    for i in range(900):
        summ+=(nc2[i]-jtwc2[i])**2
    rmse = np.sqrt(summ/900)
    # print(rmse)
    return Lon,Lat,lon_ave,lat_ave,res2,coe1,rmse

def txtres(dir):

    filedir = os.listdir(dir)
    lat_all = []
    lon_all = []
    num=0
    for file in filedir:
        files = os.listdir(dir+'/'+file)
        num += len(files)
 #       print(files)

        for each in files:
            lat_e=[]
            lon_e=[]
            data = pd.read_csv(dir+'/'+file+'/'+each,usecols=(6,7,8),header=None)
            #print(data)
            j = np.argmax(data[8])
            for i in range(len(data[8])):
                #print(data[8][i],data[8][j])
                if int(data[8][i]) == int(data[8][j]):
                    #print('true')
                    lat_e.append(float(data[6][i][:-1])/10)
                    lon_e.append(float(data[7][i][:-1])/10)
            lat_all.append(np.average(np.array(lat_e)))
            lon_all.append(np.average(np.array(lon_e)))
    lat_start=0
    lat_end=50
    lon_start=100
    lon_end=210-30
    lat_delta=30
    lon_delta=30
    #####################
    lon_ave=0
    lat_ave=0
    num_lon=0
    num_lat=0
    for k in range(len(lat_all)):
        if 180>=lon_all[k]>=100:
            lon_ave+=lon_all[k]
            lat_ave+=lat_all[k]
            num_lat+=1
    lon_ave/=num_lat
    lat_ave/=num_lat

    #####################

    res = np.zeros((lat_delta,lon_delta))
    distance = np.zeros((lat_delta,lon_delta))
    lon_res = np.linspace(lon_start,lon_end,lon_delta)
    lat_res = np.linspace(lat_start,lat_end,lat_delta)
    dis = np.sqrt((lat_res[1]-lat_res[0])**2+(lon_res[1]-lon_res[0])**2)/2
  #  print(len(lat_all))
    for k in range(len(lat_all)):

        if lat_start<lat_all[k]<lat_end and lon_start<lon_all[k]<lon_end:
   #         print(k)
            for i in range(lat_delta):
                for j in range(lon_delta):
                    if (lat_all[k]-lat_res[i])**2+(lon_all[k]-lon_res[j])**2 < dis**2:
                        res[i,j]+=1
            # distance[i,j] = (lat_all[k]-lat_res[i])**2+(lon_all[k]-lon_res[j])**2
        #print(distance)
        #a1 = np.min(distance,axis=1)
        #a = np.argmin(distance,axis=1)
        #b = np.argmin(a1)
        #res[b,a[b]]+=1
    summ = np.sum(res)
    res/= summ
 #   print(res)
    res2 =res
    for i in range(1,lat_delta-1):
        for j in range(1,lon_delta-1):
            res[i,j]+(res[i,j]+res[i,j+1]+res[i,j-1]+res[i+1,j]+res[i-1,j]+res[i+1,j+1]+res[i-1,j-1]+res[i+1,j-1]+res[i-1,j+1])/9
    np.savetxt('jtwc.dat',res2)
    Lon,Lat = np.meshgrid(lon_res,lat_res)
    return Lon,Lat,lon_ave,lat_ave,res2


if __name__ == '__main__':
    fig = plt.figure(figsize=(12,15))
    fp = r'/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model'
   # flist = []
    flist = ['/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/atmos-only/CMCC-CM2/HR/1950-2014/TC-NH_TRACK_CMCC-CM2-VHR4_highresSST-present_r1i1p1f1_gn_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/atmos-only/CNRM-CM6-1/HR/1950-2014/TC-NH_TRACK_CNRM-CM6-1-HR_highresSST-present_r1i1p1f2_gr_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/atmos-only/HadGEM3-GC31/LR/1950-2014/TC-NH_TRACK_HadGEM3-GC31-LM_highresSST-present_r1i1p1f1_gn_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/atmos-only/CMCC-CM2/LR/1950-2014/TC-NH_TRACK_CMCC-CM2-HR4_highresSST-present_r1i1p1f1_gn_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/atmos-only/CNRM-CM6-1/LR/1950-2014/TC-NH_TRACK_CNRM-CM6-1_highresSST-present_r1i1p1f2_gr_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/atmos-only/HadGEM3-GC31/HR/1950-2014/TC-NH_TRACK_HadGEM3-GC31-HM_highresSST-present_r1i1p1f1_gn_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/Coupled/CMCC-CM2/HR/1950-2014/TC-NH_TRACK_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/Coupled/CNRM/HR/1950-2014/TC-NH_TRACK_CNRM-CM6-1-HR_hist-1950_r1i1p1f2_gr_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/Coupled/HadGEM/HR/1950-2014/TC-NH_TRACK_HadGEM3-GC31-HM_hist-1950_r1i1p1f1_gn_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/Coupled/CMCC-CM2/LR/1950-2014/TC-NH_TRACK_CMCC-CM2-HR4_hist-1950_r1i1p1f1_gn_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/Coupled/CNRM/LR/1950-2014/TC-NH_TRACK_CNRM-CM6-1_hist-1950_r1i1p1f2_gr_19500101-20141231.nc','/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/model/Coupled/HadGEM/LR/1950-2014/TC-NH_TRACK_HadGEM3-GC31-MM_hist-1950_r1i1p1f1_gn_19500101-20141231.nc'] 
#    for root, dirs, files in os.walk(fp):
#        for f in files:
#            fn = os.path.join(root, f)
#            flist.append(fn)
    print(flist)
#    titles = [str(i+1) for i in range(len(flist))] 
    
#    print(titles)
    figs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']       
    titles = ['(a)CM-AH', '(b)CN-AH', '(c)Had-AH', '(d)CM-AL', '(e)CN-AL', '(f)Had-AL', '(g)CM-CH', '(h)CN-CH', '(i)Had-CH', '(j)CM-CL', '(k)CN-CL', '(l)Had-CL',' (m)OBS'] 
    for i,FILE1 in enumerate(flist):
        print(i,FILE1)
        Lon,Lat,lon_ave,lat_ave,res2,coe1,rmse = ncres(FILE1)
        ax = fig.add_subplot(5,3,1+i, projection=ccrs.PlateCarree(central_longitude=150))
    #    plt.text(102,45,figs[i], transform=ccrs.PlateCarree())
        contour_plot(ax,Lon,Lat,lon_ave,lat_ave,res2,coe1,rmse,"hot_r",titles[i],levels = {"cf":np.linspace(0,0.01,11)},fontsize = 11)

    # m part
    del Lon,Lat,lon_ave,lat_ave,res2,coe1,rmse
    dir = r'/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/data/JTWC1984-2014'
    ax = fig.add_subplot(5,3,13, projection=ccrs.PlateCarree(central_longitude=150))
    Lon,Lat,lon_ave,lat_ave,res2 = txtres(dir)
    cf = contour_plot_m(ax,Lon,Lat,lon_ave,lat_ave,res2,"hot_r",'(m)OBS',levels = {"cf":np.linspace(0,0.01,11)},fontsize = 11)
#    plt.text(102,45,figs[-1], transform=ccrs.PlateCarree())
    plt.subplots_adjust(bottom=0.15)
    ax2 = fig.add_axes([0.1,0.09,0.8,0.02])
    cb = plt.colorbar(cf,shrink=0.93, orientation='horizontal',extend='both',pad=0.205,aspect=30,ticks=np.linspace(0,0.01,11),cax=ax2)
    cb.ax.tick_params(labelsize=10)

    plt.savefig(r'/nuist/scratch/chaoxia2684/limy/python_code/HRtrack/LMI_dis/dis.png',dpi = 600)
    plt.show()
