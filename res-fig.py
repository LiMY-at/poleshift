import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = ['Times New Roman','Simsun']  # 新罗马字体Times New Roman'Times New Roman',
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['mathtext.fontset'] ='stix'

ALL_Model = pd.read_excel('./data/ALL Model.xls')
AGCM_CGCM = pd.read_excel('./data/AGCM-CGCM.xls')
HR_LR = pd.read_excel('./data/HR-LR.xls')
HR = pd.read_excel('./data/HR.xls')
LR = pd.read_excel('./data/LR.xls')
AGCM = pd.read_excel('./data/AGCM.xls')
CGCM = pd.read_excel('./data/CGCM.xls')

HR['label'] = 'HR'
LR['label'] = 'LR'
AGCM['label'] = 'AGCM'
CGCM['label'] = 'CGCM'

def add_label(ax=None,ticks=None,labels=None,colors =None,wids=0.3):
    for i,tick in enumerate(ticks):
        wids = wids
        p1 = plt.Rectangle([tick-0.5*wids,1],wids,0.2,transform=ax.transAxes,clip_on=False,fc=colors[i])
        ax.add_patch(p1)
        ax.text(tick,1.1,labels[i],va='center',ha='center',fontsize=8,transform=ax.transAxes)
        
        
fig,axs = plt.subplots(3,1,figsize=(2,6),dpi=300)

#定义标记点哪些加哪些不加

mks = [[False,False,False,False,True,False,False,False],
        [False,False,False,False,False,False,True,False],
        [False,False,False,False,False,False,False,False]]
#1行第1个图
ax = axs[0]
wids = 0.2 #箱线图宽度

colors = ['#74a29f','#7ca7d3','#eaa6a7'] #颜色
dfs  = [ALL_Model,AGCM_CGCM,HR_LR]
x = np.arange(2)
for i,df in enumerate(dfs):
    for j in x:
        if i==0:
            plot_data = df.iloc[:-1,2+j]
            ax.scatter(j+(i-1)*wids,df.iloc[-1,2+j],marker='*',fc='r',ec='k',zorder=200,lw=0.6)
        else:
            plot_data = df.iloc[:,2+j] 
        pa = ax.boxplot(plot_data,positions=[j+(i-1)*wids],  widths=0.8*wids,patch_artist=True,
                   boxprops=dict(fc=colors[i],ec='none'),medianprops=dict(color='w'))
        if mks[i][j]:
#             ax.scatter(j+(i-1)*wids,-1.45,marker='*',color='#1e90ff') #每个下面增加标记点
            ax.text(j+(i-1)*wids,-1.65,'Y',va='center',ha='center',fontsize=7,color='r',weight="bold")
        else:
            ax.text(j+(i-1)*wids,-1.65,'N',va='center',ha='center',fontsize=7,color='r',weight="bold")
            
        
ax.spines['right'].set_visible(False) #右边和上边框去掉
ax.spines['top'].set_visible(False)
ax.set_xticks([]) #横轴刻度不显示
ax.set_xlim(x[0]-0.6,x[-1]+0.6) #横轴显示范围
add_label(ax=ax,ticks=np.linspace(0.3,0.73,2),labels=[r"$\varphi_{LMI}$",r"fitted $\varphi_{LMI}$"],colors=plt.cm.RdPu(np.linspace(0.2,0.4,2)),wids=0.3)#图例
ax.set_ylim(-1.5,1.5)#纵轴显示范围
ax.set_yticks(np.arange(-1.5,1.6,0.5))#纵轴刻度
ax.plot([0.55,0.95],[1.25,1.25],transform=ax.transAxes,color='k',lw=0.8,clip_on=False)


#1行第2个图
ax = ax.inset_axes([1.2,0,3,1])
colors = ['#74a29f','#7ca7d3','#eaa6a7']
dfs  = [ALL_Model,AGCM_CGCM,HR_LR]
x = np.arange(6)
for i,df in enumerate(dfs):
    for j in x:
        if i==0:
            plot_data = df.iloc[:-1,4+j]
            ax.scatter(j+(i-1)*wids,df.iloc[-1,4+j],marker='*',fc='r',ec='k',zorder=200,lw=0.6)
        else:
            plot_data = df.iloc[:,4+j]
        ax.boxplot(plot_data,positions=[j+(i-1)*wids],  widths=0.8*wids,patch_artist=True,showfliers=False,
                   boxprops=dict(fc=colors[i],ec='none'),medianprops=dict(color='w'))
        if mks[i][j+2]:
#             ax.scatter(j+(i-1)*wids,-7.45,marker='*',color='#1e90ff') #每个下面增加标记点
            ax.text(j+(i-1)*wids,-8.95,'Y',va='center',ha='center',fontsize=7,color='r',weight="bold")
        else:
            ax.text(j+(i-1)*wids,-8.95,'N',va='center',ha='center',fontsize=7,color='r',weight="bold")
   




ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)    
add_label(ax=ax,ticks=np.linspace(0.1,0.9,6),labels=['Genesis', 'LMI', 'Seasonality', 'Vmean', 'TS',
       'SH'],colors=plt.cm.RdPu(np.linspace(0.2,0.8,8)),wids=0.15)

recs = []
for i in range(3):
    recs.append(plt.Rectangle([0,0],1,1,fc=colors[i]))
ax.legend(recs,['All Model','AGCM-CGCM','HR-LR'],ncol=1,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False,
         fontsize=10)

ax.set_ylim(-8,8)
ax.set_yticks(np.arange(-8,9,4))

ax.plot([0.05,0.95],[1.25,1.25],transform=ax.transAxes,color='k',lw=0.8,clip_on=False)
ax.annotate("",
 #           xytext=(0.5, 1.22), textcoords='axes fraction',
            xy=(0.5, 1.22),xycoords='axes fraction',
            xytext=(-0.15, 1.22), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", color="k",
                            shrinkA=5, shrinkB=5,
                            patchA=None, patchB=None,
                            connectionstyle="bar,fraction=-0.07",
                            ),transform=ax.transAxes,clip_on=False
            )



#2行第1个图
ax = axs[1]
#colors = ['#BDBADB','#F2CDCF']
colors = ["#FFC0CB",'limegreen']
dfs  = [AGCM,CGCM]
x = np.arange(2)
df_ = []
for i,df in enumerate(dfs):
    df_.append(df.iloc[:,[2,3,-1]])
df_ = pd.concat(df_,axis=0)
df_ = pd.melt(df_,id_vars=['label'], value_vars=df_.columns[:2])

sns.violinplot(data=df_,x='variable',y='value',hue='label',ax=ax,  width=0.4,palette=sns.color_palette(colors),alpha=0.5,inner=None,
               split=True,bw_method=0.2,zorder=100)
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)  
ax.legend_.remove()
ax.set_xlabel('')
ax.set_ylabel('')
add_label(ax=ax,ticks=np.linspace(0.3,0.73,2),labels=[r"$\varphi_{LMI}$",r"fitted $\varphi_{LMI}$"],colors=plt.cm.RdPu(np.linspace(0.2,0.4,2)),wids=0.3)
ax.set_ylim(-1.5,1.5)
ax.set_yticks(np.arange(-1.5,1.6,0.5))
ax.plot([0.55,0.95],[1.25,1.25],transform=ax.transAxes,color='k',lw=0.8,clip_on=False)

#2行第2个图

ax = ax.inset_axes([1.2,0,3,1])
colors = ["#FFC0CB",'limegreen']
dfs  = [AGCM,CGCM]

x = np.arange(6)
df_ = []
for i,df in enumerate(dfs):
    df_.append(df.iloc[:,[*range(4,10),-1]])
df_ = pd.concat(df_,axis=0)
df_ = pd.melt(df_,id_vars=['label'], value_vars=df_.columns[:6])

sns.violinplot(data=df_,x='variable',y='value',hue='label',ax=ax, width=0.4, density_norm='width',palette=sns.color_palette(colors),alpha=0.5,inner=None,
               split=True,bw_method=0.2,zorder=100)
       
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
ax.legend_.remove()
ax.set_xlabel('')
ax.set_ylabel('')
add_label(ax=ax,ticks=np.linspace(0.1,0.9,6),labels=['Genesis', 'LMI', 'Seasonality', 'Vmean', 'TS',
       'SH'],colors=plt.cm.RdPu(np.linspace(0.2,0.8,8)),wids=0.15)

plt.subplots_adjust(hspace=0.5)
recs = []
for i in range(2):
    recs.append(plt.Rectangle([0,0],1,1,fc=colors[i],alpha=0.5))
ax.legend(recs,['AGCM','CGCM'],ncol=1,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False,
         fontsize=10)
#ax.set_ylim(-8,8)
#ax.set_yticks(np.arange(-10,11,4))

#ax.set_ylim(-8,8)
#ax.set_yticks(np.arange(-8,9,4))
ax.set_ylim(-9,9)
ax.set_yticks(np.arange(-9,10,3))
ax.plot([0.05,0.95],[1.25,1.25],transform=ax.transAxes,color='k',lw=0.8,clip_on=False)
ax.annotate("",
 #           xytext=(0.5, 1.22), textcoords='axes fraction',
            xy=(0.5, 1.22),xycoords='axes fraction',
            xytext=(-0.15, 1.22), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", color="k",
                            shrinkA=5, shrinkB=5,
                            patchA=None, patchB=None,
                            connectionstyle="bar,fraction=-0.07",
                            ),transform=ax.transAxes,clip_on=False
            )


#3行第1个图

ax = axs[2]
colors = ["#FFC0CB",'limegreen']
dfs  = [HR,LR]
x = np.arange(2)
df_ = []
for i,df in enumerate(dfs):
    df_.append(df.iloc[:,[2,3,-1]])
df_ = pd.concat(df_,axis=0)
df_ = pd.melt(df_,id_vars=['label'], value_vars=df_.columns[:2])

sns.violinplot(data=df_,x='variable',y='value',hue='label',ax=ax,  width=0.4,palette=sns.color_palette(colors),alpha=0.5,inner=None,
               split=True,bw_method=0.2,zorder=100)
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)  
ax.legend_.remove()
ax.set_xlabel('')
ax.set_ylabel('')
add_label(ax=ax,ticks=np.linspace(0.3,0.73,2),labels=[r"$\varphi_{LMI}$",r"fitted $\varphi_{LMI}$"],colors=plt.cm.RdPu(np.linspace(0.2,0.4,2)),wids=0.3)
ax.set_ylim(-1.5,1.5)
ax.set_yticks(np.arange(-1.5,1.6,0.5))
ax.plot([0.55,0.95],[1.25,1.25],transform=ax.transAxes,color='k',lw=0.8,clip_on=False)
#3行第2个图
ax = ax.inset_axes([1.2,0,3,1])
colors = ["#FFC0CB",'limegreen']
dfs  = [HR,LR]
x = np.arange(6)
df_ = []
for i,df in enumerate(dfs):
    df_.append(df.iloc[:,[*range(4,10),-1]])
df_ = pd.concat(df_,axis=0)
df_ = pd.melt(df_,id_vars=['label'], value_vars=df_.columns[:6])

sns.violinplot(data=df_,x='variable',y='value',hue='label',ax=ax, width=0.4, density_norm='width',palette=sns.color_palette(colors),alpha=0.5,inner=None,
               split=True,bw_method=0.2,zorder=100)
        
ax.set_xticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
ax.legend_.remove()
ax.set_xlabel('')
ax.set_ylabel('')
add_label(ax=ax,ticks=np.linspace(0.1,0.9,6),labels=['Genesis', 'LMI', 'Seasonality', 'Vmean', 'TS',
       'SH'],colors=plt.cm.RdPu(np.linspace(0.2,0.8,8)),wids=0.15)

recs = []
for i in range(2):
    recs.append(plt.Rectangle([0,0],1,1,fc=colors[i],alpha=0.5))
ax.legend(recs,['HR','LR'],ncol=1,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False,
         fontsize=10)
#ax.set_ylim(-8,8)
#ax.set_yticks(np.arange(-10,11,4))

ax.set_ylim(-9,9)
ax.set_yticks(np.arange(-9,10,3))
ax.plot([0.05,0.95],[1.25,1.25],transform=ax.transAxes,color='k',lw=0.8,clip_on=False)
ax.annotate("",
            xy=(0.5, 1.22),xycoords='axes fraction',
            xytext=(-0.15, 1.22), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", color="k",
                            shrinkA=5, shrinkB=5,
                            patchA=None, patchB=None,
                            connectionstyle="bar,fraction=-0.07",
                            ),transform=ax.transAxes,clip_on=False
            )


plt.savefig('fig.png',dpi=300,bbox_inches='tight')