# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:50:38 2019

@author: Jie Zhang，微信公众号【EasyShu】，本代码源自《Python数据可视化之美》
"""
# %%
import matplotlib
import pandas as pd
import numpy as np
from plotnine import *
#from plotnine.data import *
import matplotlib.pyplot as plt
from plotnine import labels
from seaborn.palettes import color_palette 

#---------------------------(a) 单数剧系列条形图----------------------------------------------------
filepath = 'F:\Github项目\Beautiful-Visualization-with-python-master\第4章_类别比较型图表\Stackedbar_Data.csv'
df=pd.read_csv('F:\Github项目\Beautiful-Visualization-with-python-master\第4章_类别比较型图表\Stackedbar_Data.csv')

df=df.sort_values(by='Pensions', ascending=True)
# %%
df['Country']=df['Country'].astype(pd.CategoricalDtype(categories=df['Country'], ordered=True))
base_plot=(ggplot(df,aes('Country','Pensions'))+
  geom_bar(stat="identity", color="black", width=0.6,fill="#FC4E07",size=0.25) +#"#00AFBB"
  #scale_fill_manual(values=brewer.pal(9,"YlOrRd")[c(6:2)])+
  coord_flip()+
  theme(
    axis_title=element_text(size=15,face="plain",color="black"),
    axis_text = element_text(size=12,face="plain",color="black"),
    legend_title=element_text(size=13,face="plain",color="black"),
    legend_position = "right",
    aspect_ratio =1.15,
    figure_size = (6.5, 6.5),
     dpi = 100
  ))
  
print(base_plot)
# %%
#---------------------------(b)双数剧系列条形图----------------------------------------------------

df=pd.read_csv('F:\Github项目\Beautiful-Visualization-with-python-master\第4章_类别比较型图表\Stackedbar_Data.csv')

df=df.iloc[:,[0,2,1]]
df=df.sort_values(by='Pensions', ascending=True)
mydata=pd.melt(df,id_vars='Country')

# mydata['Country']=mydata['Country'].astype("category",categories= df['Country'],ordered=True)
cat_dtype = pd.CategoricalDtype(categories=df['Country'], ordered=True)
mydata['Country']=mydata['Country'].astype(cat_dtype)

base_plot=(ggplot(mydata,aes('Country','value',fill='variable'))+
  geom_bar(stat="identity", color="black", position=position_dodge(),width=0.7,size=0.25)+
  scale_fill_manual(values=("#00AFBB", "#FC4E07", "#E7B800"))+
  coord_flip()+
  theme(
    axis_title=element_text(size=15,face="plain",color="black"),
    axis_text = element_text(size=12,face="plain",color="black"),
    legend_title=element_text(size=14,face="plain",color="black"),
    legend_background  =element_blank(),
    legend_position = (0.8,0.2),
    aspect_ratio =1.15,
    figure_size = (6.5, 6.5),
     dpi = 100
  ))
print(base_plot)

#-------------------------------(c)堆积条形图-------------------------------------------------------
df=pd.read_csv(filepath)
Sum_df=df.iloc[:,1:].apply(lambda x: x.sum(), axis=0).sort_values(ascending=True)
meanRow_df=df.iloc[:,1:].apply(lambda x: x.mean(), axis=1)
Sing_df=df['Country'][meanRow_df.sort_values(ascending=True).index]
mydata=pd.melt(df,id_vars='Country')
cat_dtype2 = pd.CategoricalDtype(categories= Sum_df.index,ordered=True)

# mydata['variable']=mydata['variable'].astype("category",categories= Sum_df.index,ordered=True)
mydata['varialbel'] = mydata['variable'].astype(cat_dtype2)
cat_dtype3 = pd.CategoricalDtype(categories= Sing_df,ordered=True)
mydata['Country']=mydata['Country'].astype(cat_dtype3)


base_plot=(ggplot(mydata,aes('Country','value',fill='variable'))+
  geom_bar(stat="identity", color="black", position='stack',width=0.65,size=0.25)+
  scale_fill_brewer(palette="YlOrRd")+
  coord_flip()+
  theme(
    axis_title=element_text(size=18,face="plain",color="black"),
       axis_text = element_text(size=16,face="plain",color="black"),
       legend_title=element_text(size=18,face="plain",color="black"),
       legend_text = element_text(size=16,face="plain",color="black"),
    legend_background  =element_blank(),
    legend_position = 'right',
    aspect_ratio =1.15,
    figure_size = (6.5, 6.5),
     dpi = 50
  ))
print(base_plot)
#base_plot.save('堆积条形图.pdf')
# %% 窗口比例堆积图
#------------------------------(d) 百分比堆积柱形图-------------------------------------------------------
df=pd.read_excel('G:\动态结果图表\GCotherStats.xlsx',sheet_name='state_fraction_50',header=0)
SumCol_df=df.iloc[:,1:].apply(lambda x: x.sum(), axis=1)
df.iloc[:,1:]=df.iloc[:,1:].apply(lambda x: x/SumCol_df, axis=0)

meanRow_df=df.iloc[:,1:].apply(lambda x: x.mean(), axis=0).sort_values(ascending=True)
Per_df=df.loc[:,meanRow_df.idxmax()].sort_values(ascending=True)
Sing_df=df['group'][Per_df.index]

mydata=pd.melt(df,id_vars='group')
cat_dtype4 = pd.CategoricalDtype(categories=Sing_df,ordered=True)
cat_dtype5 = pd.CategoricalDtype(categories=meanRow_df.index,ordered=True)
mydata['group']=mydata['group'].astype(cat_dtype4)
mydata['variable']=mydata['variable'].astype(cat_dtype5)
print(mydata)


base_plot=(ggplot(mydata,aes(x='group',y='value',fill='variable'))
+geom_bar(stat="identity", color="black", position='fill',width=0.7,size=0.25)
+scale_fill_discrete()
+theme(
       #text=element_text(size=15,face="plain",color="black"),
       axis_title=element_text(size=18,face="plain",color="black"),
       axis_text = element_text(size=16,face="plain",color="black"),
       legend_title=element_text(size=18,face="plain",color="black"),
       legend_text = element_text(size=16,face="plain",color="black"),
       aspect_ratio =1.15,
       figure_size = (6.5, 6.5),
       dpi = 75
       )
)
print(base_plot)
#base_plot.save('百分比堆积柱形图.pdf')
# %% matplotlib画图
# 这里需要将数据分为两列，第一列是ncNT，第二列是depNT
df = pd.read_excel('G:\动态结果图表\GCstats.xlsx',sheet_name='NT')
nc = np.array(df['ncNT'])
dep = np.array(df['depNT'][:-1]) # 去掉最后一个nan值

ncAvg = nc.mean()
depAvg = dep.mean()

# print(ncAvg, depAvg)

# print(data)
mydata = pd.DataFrame({'group':['NC', 'DEP'], 'value':[ncAvg, depAvg]})
X = pd.Series(['NC', 'DEP'])
X =  X.astype(pd.CategoricalDtype(categories=['NC', 'DEP'],ordered=True))
X = pd.factorize(X)[0]
fig = plt.figure(figsize=(5,5), dpi=100)

plt.bar(mydata['group'], mydata['value'], width = 0.3, align ='center', label='group',color=['#DB5F57','#57D3DB'])
ax = plt.gca()
# plt.title(label='Number of state transitions',y=-0.2)
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')
# ax.set_xlabel("group")
ax.set_ylabel('Number of state transitions')
ax.scatter(x=0.5, y=7, marker="*",c="k")
p_value_cov = 0.015
ax.plot([0, 0, 1, 1], [6.5, 6.8, 6.8, 6.5], lw=1, c="k") 
ax.text(0.5, 7.2, "permutation-test: "+ str(p_value_cov), ha='center', va='bottom', color="k")
# ax.tick_params(which='major',direction='in',length=3,width=1.,labelsize=14,bottom=False)
# ax.grid(axis='y',ls='--',c='gray')
plt.show()

# %% 用plotnine画图
# 这里需要将数据分为两列，第一列是ncNT，第二列是depNT
df = pd.read_excel('G:\动态结果图表\GCstats.xlsx',sheet_name='NT')
nc = np.array(df['ncNT'])
dep = np.array(df['depNT'][:-1]) # 去掉最后一个nan值

ncAvg = nc.mean()
depAvg = dep.mean()

# print(ncAvg, depAvg)

# print(data)
mydata = pd.DataFrame({'group':['NC', 'DEP'], 'value':[ncAvg, depAvg]})

mydata = mydata.sort_index()
# print(mydata)
# 有颜色填充必然存在图例legend
mydata['group'] = pd.Categorical(mydata['group'],categories=mydata['group'],ordered=True)
base_plot=(ggplot(mydata,aes('group','value',fill='group', color_palette=['#00BFFF','#DC143C']))+
  geom_bar(stat="identity", color="black",position='dodge',width=0.5,size=0.25)+
  scale_fill_discrete()+
  # geom_text(aes(label='value'),position='dodge',width=0.5,
  # size=8,va='bottom', format_string='{}')+
  theme(
    # panel_background=element_rect(fill='white'),
    panel_grid=element_blank(),
    axis_title=element_text(size=15,face="plain",color="black"),
    axis_text = element_text(size=12,face="plain",color="black"),
    # legend_
    legend_title=element_text(size=12,face="plain",color="white"),
    # legend_background  =element_blank(),
    # legend_position = (0.8,0.8),
    # aspect_ratio =1.15,
    figure_size = (6.5, 6.5),
     dpi = 75
  ))  
print(base_plot)
# %%
#---------------------------(b)双数剧系列条形图----------------------------------------------------
import seaborn as sns

df = pd.read_excel('G:\动态结果图表\GCstats.xlsx',sheet_name='avgMDT')

# 计算出每个状态的平均驻留时间和标准差



sns.set_context("notebook", font_scale=1.5,
                rc={'font.size': 12, 'axes.labelsize': 15, 'legend.fontsize':12, 
                    'xtick.labelsize': 12,'ytick.labelsize': 12})
fig = plt.figure(figsize=(10,8),dpi=120)
# sns.set_palette("Set2")
ax = sns.barplot(x='group', y="value",hue="state",
              data = df,capsize=0.1,linewidth=0.3,errwidth=0.8,palette=['#00BFFF','#DC143C'])
# mydata['Country']=mydata['Country'].astype("category",categories= df['Country'],ordered=True)
# cat_dtype = pd.CategoricalDtype(categories=df['Country'], ordered=True)
# mydata['Country']=mydata['Country'].astype(cat_dtype)
ax = plt.gca()
# plt.title(label='Number of state transitions',y=-0.2)
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')
ax.set_ylabel('average state dwell time')
ax.set_ylim([0,40])
ax.set_xlim([-0.8, 1.8])

ax.scatter(x=0.3, y=38.5, marker="*",c="k")
p_value_cov =0.012
ax.plot([-0.2, -0.2, 0.8, 0.8], [37.5, 38, 38, 37.5], lw=1, c="k") 
ax.text(0.3, 39, "permutation-test: "+ str(p_value_cov), ha='center', va='bottom', color="k")

plt.show()

# base_plot=(ggplot(df,aes('group','state 1',fill='group'))+
#   geom_bar(stat="identity", color="black", position=position_dodge(),width=0.7,size=0.25)+
#   # scale_fill_manual(values=("#00AFBB", "#FC4E07", "#E7B800"))+
#   scale_fill_discrete()+

#   theme(
#     axis_title=element_text(size=15,face="plain",color="black"),
#     axis_text = element_text(size=12,face="plain",color="black"),
#     legend_title=element_text(size=14,face="plain",color="black"),
#     legend_background  =element_blank(),
#     legend_position = (0.8,0.2),
#     aspect_ratio =1.15,
#     figure_size = (6.5, 6.5),
#      dpi = 100
#   ))
# print(base_plot)
# %% 小提琴图
# 画出四种状态转换比例之间的差异
import seaborn as sns

df = pd.read_excel('G:\动态结果图表\GCstats.xlsx',sheet_name='state_convert')

sns.set_style("ticks")

fig = plt.figure(figsize=(12,8))
violinplot=sns.violinplot(x="state", y="value", hue="group",
                         data=df, inner="quartile", split=True,orient="v",
                         linewidth=1,colors=["#F7746A", "#36ACAE"]) 
#violinplot.despine(left=True)
plt.legend(loc="center right",
          bbox_to_anchor=(1.5, 0, 0, 1))  
ax = plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')  
Means = df.groupby(['state', 'group']).agg({'value':np.mean})

# control = Means['value'][0::2]
# depression = Means['value'][1::2]
# X = pd.Series(['InsulaL','InsulaR','LOFL', 'LOFR', 'MOFL', 'MOFR','PPL','PPR', 'rACGL', 'rACGR', 'rMFL', 'rMFR'])
# X = X.astype(CategoricalDtype(categories=['InsulaL','InsulaR','LOFL', 'LOFR', 'MOFL', 'MOFR','PPL','PPR', 'rACGL', 'rACGR', 'rMFL', 'rMFR'],ordered=True))
# X = pd.factorize(X)[0] -0.2
# ax.scatter(x=X, y=control,c='k')
# ax.scatter(x=X + 0.4, y=depression,c='k')
# plt.ylim(0,14)                            
#violinplot.set_axis_labels("day", "total bill")       
#inner：控制violinplot内部数据点的表示，
#有“box”, “quartile”, “point”, “stick”四种方式。
#fig.savefig('violinplot_split2.pdf')
# %%
sns.set_context("notebook", font_scale=1.5,
                rc={'font.size': 12, 'axes.labelsize': 20, 'legend.fontsize':15, 
                    'xtick.labelsize': 15,'ytick.labelsize': 15})

fig = plt.figure(figsize=(12,7), dpi=75)

violinplot=sns.violinplot(x="state", y="value", hue="group",
                         data=df, inner="box", split=False,
                         linewidth=2,palette=["#F7746A", "#36ACAE"]) 
ax = plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')
ax.set_ylabel('State transition ratio')
ax.set_xlabel(None)
ax.scatter(x=1, y=1.52, marker="*",c="k")
ax.scatter(x=3, y=1.52, marker="*",c="k")
p_value_cov = 0.016

ax.plot([0.8, 0.8, 1.2, 1.2], [1.47, 1.5,1.5, 1.47], lw=1, c="k") 
ax.text(1, 1.54, "p = "+ str(p_value_cov), ha='center', va='bottom', color="k")
ax.plot([2.8, 2.8, 3.2, 3.2], [1.47, 1.5,1.5, 1.47], lw=1, c="k") 
ax.text(3, 1.54, "p = "+ str(0.018), ha='center', va='bottom', color="k")
plt.legend(loc="center right",
          bbox_to_anchor=(1.2, 0,0, 2)) 
              
# %%
import seaborn as sns

df = pd.read_excel('G:\动态结果图表\GCstats.xlsx',sheet_name='state_convert')

# 计算出每个状态的平均驻留时间和标准差



sns.set_context("notebook", font_scale=1.5,
                rc={'font.size': 12, 'axes.labelsize': 15, 'legend.fontsize':12, 
                    'xtick.labelsize': 12,'ytick.labelsize': 12})
fig = plt.figure(figsize=(10,8),dpi=120)
# sns.set_palette("Set2")
ax = sns.barplot(x='state', y="value",hue="group",
              data = df,capsize=0.1,linewidth=0.3,errwidth=0.8,palette=['#00BFFF','#DC143C'])
# mydata['Country']=mydata['Country'].astype("category",categories= df['Country'],ordered=True)
# cat_dtype = pd.CategoricalDtype(categories=df['Country'], ordered=True)
# mydata['Country']=mydata['Country'].astype(cat_dtype)
ax = plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')
ax.set_ylabel('State transition ratio')
ax.set_xlabel(None)
ax.scatter(x=1, y=.82, marker="*",c="k")
ax.scatter(x=3, y=0.97, marker="*",c="k")
p_value_cov = 0.016

ax.plot([0.8, 0.8, 1.2, 1.2], [0.75, 0.78,0.78, 0.75], lw=1, c="k") 
ax.text(1, 0.83, "p = "+ str(p_value_cov), ha='center', va='bottom', color="k")
ax.plot([2.8, 2.8, 3.2, 3.2], [0.94, 0.95,0.95, 0.94], lw=1, c="k") 
ax.text(3, 0.99, "p = "+ str(0.018), ha='center', va='bottom', color="k")
plt.legend(loc="center right",
          bbox_to_anchor=(1.2, 0,0, 2)) 
# %% 条形图，画出转换次数之间的差异
import seaborn as sns
df = pd.read_excel('G:\动态结果图表\GCotherStats.xlsx',sheet_name='NT_50')



sns.set_context("notebook", font_scale=1.5,
                rc={'font.size': 12, 'axes.labelsize': 15, 'legend.fontsize':12, 
                    'xtick.labelsize': 12,'ytick.labelsize': 12})
fig = plt.figure(figsize=(10,8),dpi=120)
# sns.set_palette("Set2")
ax = sns.barplot(x='group', y="value",
              data = df,capsize=0.1,linewidth=0.8,errwidth=0.6,palette=['#00BFFF','#DC143C'])

ax = plt.gca()
# plt.title(label='Number of state transitions',y=-0.2)
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')
ax.set_ylabel('average state dwell time')

ax.set_ylim([0,10])
# 设置条形图的宽度
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .35)
ax.set_xlim([-0.5, 1.5])
ax.set_ylabel('Number of state transitions')
ax.scatter(x=0.5, y=9.4, marker="*",c="k")
p_value_cov = 0.023
ax.plot([0, 0, 1, 1], [9.2, 9.3, 9.3, 9.2], lw=1, c="k") 
ax.text(0.5, 9.5, "p = "+ str(p_value_cov), ha='center', va='bottom', color="k")
plt.show()
# %%
import seaborn as sns
df = pd.read_excel('G:\动态结果图表\GCotherStats.xlsx',sheet_name='NT_all')

# 计算0.2s、0.4s，0.5s，2sGC状态
# 条形图，画出转换次数之间的差异
df = df.sort_index(ascending=False)


sns.set_context("notebook", font_scale=1.5,
                rc={'font.size': 12, 'axes.labelsize': 15, 'legend.fontsize':12, 
                    'xtick.labelsize': 12,'ytick.labelsize': 12})
fig = plt.figure(figsize=(10,8),dpi=120)
# sns.set_palette("Set2")
ax = sns.barplot(x='winlen', y="value",hue='group',
              data = df,capsize=0.1,linewidth=0.8,errwidth=0.6,palette=['#00BFFF','#DC143C'])

ax = plt.gca()
# plt.title(label='Number of state transitions',y=-0.2)
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')
ax.set_ylabel('average state dwell time')

# ax.set_ylim([0,10])
# 设置条形图的宽度
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

# change_width(ax, .35) # 原始值
# ax.set_xlim([-0.5, 1.5])
ax.set_ylabel('Number of state transitions')
# ax.scatter(x=0.5, y=9.4, marker="*",c="k")
# p_value_cov = 0.023
# ax.plot([0, 0, 1, 1], [9.2, 9.3, 9.3, 9.2], lw=1, c="k") 
# ax.text(0.5, 9.5, "p = "+ str(p_value_cov), ha='center', va='bottom', color="k")
plt.show()
# %%
# %% 
import seaborn as sns

df = pd.read_excel('G:\动态结果图表\GCotherStats.xlsx',sheet_name='MDT_50')
# 计算0.2s，0.4s，0.5s，1.5s，2s窗的平均停留时间

# 计算出每个状态的平均驻留时间和标准差


# df = df.sort_index(ascending=False)
df=df.sort_values(by='state')
sns.set_context("notebook", font_scale=1.5,
                rc={'font.size': 12, 'axes.labelsize': 15, 'legend.fontsize':12, 
                    'xtick.labelsize': 12,'ytick.labelsize': 12})
fig = plt.figure(figsize=(10,8),dpi=120)
# sns.set_palette("Set2")
ax = sns.barplot(x='group', y="value",hue="state",
              data = df,capsize=0.1,linewidth=0.3,errwidth=0.8,palette=['#00BFFF','#DC143C'])

ax = plt.gca()

ax.spines['right'].set_color('None')
ax.spines['top'].set_color('none')
ax.set_ylabel('average state dwell time')
ax.set_ylim([0,65])
ax.set_xlim([-0.8, 1.8])

ax.scatter(x=0.3, y=62.5, marker="*",c="k")
p_value_cov =0.012
ax.plot([-0.2, -0.2, 0.8, 0.8], [61, 62, 62, 61], lw=1, c="k") 
ax.text(0.3, 63, "permutation-test: "+ str(p_value_cov), ha='center', va='bottom', color="k")

plt.show()

# %%画出不同窗长度之间的转换次数差异

# %%
