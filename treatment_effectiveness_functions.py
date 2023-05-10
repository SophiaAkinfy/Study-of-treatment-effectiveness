#!/usr/bin/env python
# coding: utf-8

# In[39]:

# import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import math as m

# visualisation
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pylab as py

# libraries for statistics 
#import re
#import statistics
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from scipy.stats import pearsonr
from scipy.stats import kurtosis, skew

# Survival analysis
from lifelines.statistics import multivariate_logrank_test
from lifelines import NelsonAalenFitter
from lifelines import KaplanMeierFitter

# ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

# Post-hocs 
import scikit_posthocs as sp

def pie_chart(data, group, title, colors = None, explode = None, figsize = (8, 8)):
    ''' Visualizes proportions (pie chart) of a categorical variable with text '''
    
    data_table = (data
                    .groupby(group, as_index = False)
                    .agg({'ID': 'count'})
                    .sort_values(by='ID', ascending=False)
                    .reset_index(drop=True))
    data_table_lable = data_table['ID'].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    labels = data_table[group].tolist()

    for i in range(len(data_table['ID'].tolist())):
        data_table_lable[i] = str(data_table[group].tolist()[i]) + ' (n='+ data_table['ID'][i].astype('str') + ')'

    x = data_table['ID'].tolist()
    ax.pie(x, labels=data_table_lable, autopct='%.1f%%',
         wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
         colors = colors,
         textprops={'size': '14'},
         explode = explode)
    ax.set_title(title, fontsize=18)
    plt.tight_layout()


# In[87]:


def pie_chart_icd(data, group, title, colors = None, explode = None, figsize = (8, 8)):
    ''' Pie chart of patient's diagnosis following ICD-10 '''
    data_table = (data
                    .groupby(group, as_index = False)
                    .agg({'ID': 'count'})
                    .sort_values(by='ID', ascending=False)
                    .reset_index(drop=True))
    data_table_lable = data_table['ID'].tolist()
    fig = px.pie(data_table, values='ID', names=group, title=title)
    fig.show()


# In[91]:


def compar_distrib_in_groups(data, param, group, title):
    
    '''Distribution of a variable in groups. Visualisation and Tukey's test'''
    
    # graf
    fig = px.box(data, x=group, y=param, color = group, points="all", 
                template="gridon",
                title = title)
    fig.show('png')

    # perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=data[param],
    groups=data[group],
    alpha= 0.05)
    
    #display results
    print(tukey)
    
    data_stat = data.groupby(group, as_index=False)[param].agg({'mean', 'sem'})
    data_stat['95% CI low'] = data_stat['mean'] - data_stat['sem'] * 1.96
    data_stat['95% CI up'] = data_stat['mean'] + data_stat['sem'] * 1.96
    print(data_stat)


# In[99]:


# Bar plot %
def bar_plot_proc(data, group1, group2, labels, title, w = 1000, h=700, colors = None, category_orders = None):
    
    ''' Bar plot with proportions by groups '''
    
    data_group = data.pivot_table(index=[group1, group2], values='ID', 
                                      aggfunc='count', fill_value = 0).reset_index()
    data_group_total = data.pivot_table(index=[group2], values='ID', 
                                      aggfunc='count', fill_value = 0).reset_index()
    data_groups_total = data_group.merge(data_group_total, how='left', on=group2)
    data_groups_total['proc'] = data_groups_total['ID_x'] / data_groups_total['ID_y'] * 100
    data_groups_total['title'] = (round(data_groups_total['proc'], 1).astype('str') + '% ' 
                                  + '(n='+ data_groups_total['ID_x'].astype('str') +')')
    
    fig = px.bar(data_groups_total, x=group2, y="proc", color=group1, 
              labels=labels,
              title = title,
              color_discrete_map=colors,
              text = 'title', 
              category_orders=category_orders,
                                  template="gridon",
                                  width=w, height=h)
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
    fig.show('png')


# In[121]:


def numerical_var_analisys_per_visits(data, group1, group2, title, y_axis):#, y_lim):
    
    '''The function displays descriptive stats of the variable at control points (including sem, skew and kurt), 
       plot qq plot by control points, 
       plot bar chart by control points, 
       Plots the mean and 95% of CI in control points'''
    
    print(group2)
   
    # descriptive statistics
    print(data[[group1, group2]].dropna().groupby(group1, as_index='False')[group2].agg({'describe', 'sem', 'skew'}))
    print('kurt')
    print(data[[group1, group2]].dropna().groupby(group1).apply(pd.DataFrame.kurt))

    
    # hist and qq plot      
    for item in data['number_of_visit'].unique().tolist():
        plt.rcParams['figure.figsize'] = [12, 6]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(str(group2) + ' QQ Plot Visit ' + str(item), fontsize=18)
        
        
        # Q-Q Plot graph
        stats.probplot(data.query('number_of_visit == @item')[group2].dropna(), plot=ax1)
        ax1.set_title("Normal Q-Q Plot")

        # normal distribution histogram + distribution
        sns.histplot(data.query('number_of_visit == @item')[group2].dropna(), kde=True)
        plt.show('png')
    
    # distribution by visits
    fig = px.box(data, x=group1, y=group2, color = group1, points="all", 
            template="gridon",
            labels = {group2: y_axis, 'number_of_visit': 'Number of visit'},
            title = title)
    fig.show('png')

    
    # mean changing
    data_mean = (data.groupby(group1)[group2].agg({'mean', 'median', 'sem'}).reset_index())   
    print(data_mean)
    
    fig = go.Figure(data=go.Scatter(
        x=data_mean['number_of_visit'].astype('str').tolist(),
        y=data_mean['mean'].tolist(),
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=(data_mean['sem']*1.96).tolist(),
            visible=True)
    ))

    fig.update_layout(title=title)
    fig.update_layout(template='gridon')
    fig.update_yaxes(title='Mean (95% CI)')
    fig.show('png')


# In[122]:


def parametric_test(data, param, group):
    '''Parametric test. Visualisation, ANOVA and Tukey's test'''
    
    # graf
    fig = px.box(data, x=group, y=param, color = group, points="all", 
                template="gridon",
                title = param)
    fig.show('png')
    
    data_stat = data.groupby(group, as_index=False)[param].agg({'mean', 'sem'})
    data_stat['95% CI low'] = data_stat['mean'] - data_stat['sem'] * 1.96
    data_stat['95% CI up'] = data_stat['mean'] + data_stat['sem'] * 1.96
    print(data_stat)
    print()
    
    text_ANOVA = param + ' ~ C(' + group + ')'
    
    #two-way ANOVA
    model = ols(text_ANOVA, data=data).fit()
    print(sm.stats.anova_lm(model, typ=2))

    # perform Tukey's test
    tukey = pairwise_tukeyhsd(endog=data[param],
    groups=data[group],
    alpha= 0.05)
    
    #display results
    if True in tukey.reject:
        print('There is a sig difference')
        print("Tukey's test")
        print(tukey)
          
        # mean changing
        data_mean = (data.groupby(group)[param].agg({'mean', 'median', 'sem'}).reset_index())   

        fig = go.Figure(data=go.Scatter(
            x=data_mean['number_of_visit'].astype('str').tolist(),
            y=data_mean['mean'].tolist(),
            error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=(data_mean['sem']*1.96).tolist(),
                visible=True)
        ))

        fig.update_layout(title=param)
        fig.update_layout(template='gridon')
        fig.update_yaxes(title='Mean (95% CI)')
        fig.show('png')


# In[123]:


def nonparametric_test(data, param, group):
    '''Nonparametric test. Visualisation, Kruskal Wallis test, Mann-Whitneyu test Bonferroni adjusting'''
    # descriptive statistics
    print(data[[group, param]].dropna().groupby(group, as_index='False')[param].agg({'describe'}))
    
    # graf
    fig = px.box(data, x=group, y=param, color = group, points="all", 
                template="gridon",
                title = param)
    fig.show('png')

    # Kruskal-Wallis test
    a = data.query('number_of_visit == 1')[param].dropna().tolist()
    b = data.query('number_of_visit == 2')[param].dropna().tolist()
    c = data.query('number_of_visit == 3')[param].dropna().tolist()
    d = data.query('number_of_visit == 4')[param].dropna().tolist()
    st, p_value = stats.kruskal(a, b, c, d, nan_policy='omit')
    
    print(param)
    print('H(3) =', round(st, 2), 'p =', round(p_value, 2))    

    # Mann-Whitneyu test. Method for adjusting p values - Bonferroni
    if p_value < 0.05:
        print('Mann-Whitneyu test. Method for adjusting p values - Bonferroni')
        print(sp.posthoc_mannwhitney(data, val_col=param, group_col=group, alternative='two-sided', p_adjust='bonferroni'))


# In[124]:


def param_per_protocol_analysis(data, number_of_par, title): 
    
    '''Visualisation, ANOVA for repeated measures, Tukey's test'''
    
    
    data_visit = ({'ID': range(1, len(data.query('number_of_visit == 1')[number_of_par]) + 1, 1),
    'col1': data.query('number_of_visit == 1')[number_of_par].tolist(), 
    'col2': data.query('number_of_visit == 2')[number_of_par].tolist(),
    'col3': data.query('number_of_visit == 3')[number_of_par].tolist(),
    'col4': data.query('number_of_visit == 4')[number_of_par].tolist()})
        
    data_drop = pd.DataFrame(data=data_visit)
    data_drop = data_drop.dropna()

    print("Number of the participants: ", len(data_drop['col1'].tolist()))
    
    d = {'number_of_visit': [1] * len(data_drop['col1'].tolist()) + [2] * len(data_drop['col1'].tolist()) + 
         [3] * len(data_drop['col1'].tolist()) + [4] * len(data_drop['col1'].tolist()), 
         "ID": data_drop['ID'].tolist() * 4,
         number_of_par: data_drop['col1'].tolist() + data_drop['col2'].tolist() + data_drop['col3'].tolist() 
         + data_drop['col4'].tolist()}
    
    df = pd.DataFrame(data=d) 
    
    fig = px.box(df, x='number_of_visit', 
             y=number_of_par, color = 'number_of_visit', points="all", 
             template="gridon",
             title = title)    
    fig.show('png')  
    
    # mean changing
    data_mean = (df.groupby('number_of_visit', as_index=False)[number_of_par].agg({'mean', 'sem'}))

    fig = go.Figure(data=go.Scatter(
    x=['col1', 'col2', 'col3', 'col4'],
    y=data_mean['mean'].tolist(),
    error_y=dict(
                 type='data', # value of error bar given in data coordinates
                 array=(data_mean['sem']*1.96).tolist(),
                 visible=True)
    ))

    fig.update_layout(title=title)
    fig.update_layout(template='gridon')
    fig.update_yaxes(title='Mean (95% CI)')
    fig.show('png')
    
    result = AnovaRM(data=df, depvar=number_of_par, subject='ID', within=['number_of_visit']).fit()
    print(result)
    
    #display results
    if result.anova_table['Pr > F'][0] < 0.05:
        print('There is a sig difference')
        
        # perform Tukey's test
        tukey = pairwise_tukeyhsd(endog=df[number_of_par],
        groups=df['number_of_visit'],
        alpha= 0.05)
        print(tukey)         


# In[125]:


def nonparam_per_protocol_analysis(data, number_of_par, title): 
    
    '''Visualisation, The Friedman test for repeated samples, Conover test'''
    
    data_visit = ({'ID': range(1, len(data.query('number_of_visit == 1')[number_of_par]) + 1, 1),
    'col1': data.query('number_of_visit == 1')[number_of_par].tolist(), 
    'col2': data.query('number_of_visit == 2')[number_of_par].tolist(),
    'col3': data.query('number_of_visit == 3')[number_of_par].tolist(),
    'col4': data.query('number_of_visit == 4')[number_of_par].tolist()})
        
    data_drop = pd.DataFrame(data=data_visit)
    data_drop = data_drop.dropna()

    print("Number of the participants: ", len(data_drop['col1'].tolist()))

    d = {'number_of_visit': [1] * len(data_drop['col1'].tolist()) + [2] * len(data_drop['col1'].tolist()) + 
         [3] * len(data_drop['col1'].tolist()) + [4] * len(data_drop['col1'].tolist()), 
         "ID": data_drop['ID'].tolist() * 4,
         number_of_par: data_drop['col1'].tolist() + data_drop['col2'].tolist() + data_drop['col3'].tolist() 
         + data_drop['col4'].tolist()}
    
    df = pd.DataFrame(data=d) 
    
    fig = px.box(df, x='number_of_visit', 
              y=number_of_par, color = 'number_of_visit', points="all", 
              template="gridon",
              title = title)    
    fig.show('png')    
    
    # mean changing
    data_mean = (df.groupby('number_of_visit', as_index=False)[number_of_par].agg({'mean', 'sem'}))
    
    fig = go.Figure(data=go.Scatter(
    x=['col1', 'col2', 'col3', 'col4'],
    y=data_mean['mean'].tolist(),
    error_y=dict(
              type='data', # value of error bar given in data coordinates
              array=(data_mean['sem']*1.96).tolist(),
              visible=True)
     ))

    fig.update_layout(title=title)
    fig.update_layout(template='gridon')
    fig.update_yaxes(title='Mean (95% CI)')
    fig.show('png')
    
    # the Friedman test for repeated samples
    friedman_result = stats.friedmanchisquare(data_drop['col1'], data_drop['col2'], data_drop['col3'], data_drop['col4'])
    print(friedman_result)
 
  
    if friedman_result.pvalue < 0.05:
        # Conover test
        print('Conover test')
        print(sp.posthoc_conover_friedman(data_drop[['col1', 'col2', 'col3', 'col4']]))   