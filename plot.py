import pandas as pd
from os import chdir
from researchpy import corr_pair
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib
from matplotlib.transforms import TransformedBbox
from sklearn.preprocessing import normalize
import scipy.io
import statsmodels.formula.api as smf
from math import e

chdir("/path/to/your/wd/")

df_all = pd.read_csv("all_measures.csv") # output of get_LM_surprisal.py

############
# PLOTTING #
############

rating_log = -np.log(df_all["rating_mean"])
df_all.insert(loc=9, column='rating_s', value=rating_log)

# exclude punctuation!

# From Frank: "Words attached to a comma, clitics, sentence-initial, and sentence-final words were discarded from further analysis [...]."
# note that tokens at the beginning of sentences are already excluded (no cloze/rating data)

is_start_end = []
for index, row in df_all.iterrows():
    if "." in row.word:
        is_start_end.append(1)
    elif "," in row.word:
        is_start_end.append(1)
    elif "'" in row.word:
        is_start_end.append(1) # clitics
    else:
        is_start_end.append(0)

df_all["is_start_end"] = is_start_end

cols = ['rating_mean', 'cloze_p_smoothed','rating_s','cloze_s','s_GPTNeo_2.7B', 's_GPTNeo', 's_GPTNeo_125M', "s_GPT2_xl", "s_GPT2_large","s_GPT2_medium", 's_GPT2', 'rnn', 'psg', 'bigram', 'trigram', 'tetragram', #'rnn_pos', 'psg_pos', 'bigram_pos', 'trigram_pos', 'tetragram_pos', 
        'RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time','ELAN', 'LAN', 'N400', 'P600', 'EPNP', 'PNP']

labels = ['rating', 'cloze$_{p}$','rating$_{H}$', 'cloze$_{H}$','GPT-Neo$_{2.7B}$', 'GPT-Neo$_{1.3B}$', 'GPT-Neo$_{125M}$', 'GPT-2$_{1.5B}$', 'GPT-2$_{774M}$', 'GPT-2$_{355M}$', 'GPT-2$_{124M}$', 'RNN', 'PSG', 'bigram', 'trigram', 'tetragram', #'RNN$_{PoS}$', 'PSG$_{PoS}$', 'bigram$_{PoS}$', 'trigram$_{PoS}$', 'tetragram$_{PoS}$', 
          'FFix', 'FPass', 'GoPast', 'RightBound','SPR',  'ELAN', 'LAN', 'N400', 'P600', 'EPNP', 'PNP'] # LaTeX-like labels for plotting correlation

d_vars = {key:value for key, value in zip(cols, labels)}

df_corr= df_all[df_all.is_start_end == 0][cols].corr(method='pearson')

# some parameters for nice plotting
font = {'weight' : 'normal',
        'size'   : 18}
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(18, 14.4), dpi=300) # (15, 12)
ax = fig.add_subplot(111)
sns.heatmap(df_corr, #cmap="viridis",
            cmap=sns.diverging_palette(22, 220, s=80, n=200),
            annot=False,vmin=-1, vmax=1, center=0,
        xticklabels=labels,
        yticklabels=labels)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

# print correlation table (for the appendix)
corr_table = corr_pair(df_all[df_all.is_start_end == 0][cols])
np.mean([abs(float(w)) for w in corr_table["r value"]]) # avg pairwise corr

make_table = []
for index, row in corr_table.iterrows():
    a, b = index.split(" & ")
    make_table.append(" & ".join([d_vars[a], d_vars[b], row["r value"], row["p-value"]]))
make_table.append("Average & Average & 0.4201 & NA")

num1 = [n1 for n1 in range(1, 352, 2)]
num2 = [n2 for n2 in range(0, 353, 2)]

for n1, n2 in zip(num1, num2):
    s = " & ".join([make_table[n1], make_table[n2]])+" \\\\"
    print(s)

###########################
# HIERARCHICAL CLUSTERING #
###########################

dissimilarity = 1 - np.abs(df_corr) # distance as negative absolute correlation
distArray = ssd.squareform(dissimilarity)
Z = linkage(distArray, method='ward', metric='euclidean', optimal_ordering=False)

fig = plt.figure(figsize=(15, 5), dpi=300)
ax = fig.add_subplot(111)
with plt.rc_context({'lines.linewidth': 2}):
    dn = dendrogram(Z, labels=labels, color_threshold=0, above_threshold_color='k')
for axis in ['left']:
    ax.spines[axis].set_linewidth(2)
for axis in ['top', 'right', 'bottom']:
    ax.spines[axis].set_linewidth(2)
plt.xticks(fontsize=15, horizontalalignment='right', rotation=45)
plt.yticks(fontsize=15)
plt.show()

#######################
# PREDICTIVE ANALYSES #
#######################

# SUBTLEX FREQUENCY
subt = pd.read_excel('/path/to/your/subtlex.xlsx')
s = {}; cd = {}
for index, row in subt.iterrows():
    s[row.Word] = np.log(row.FREQcount)
    cd[row.Word] = np.log(row.CDcount)

punct_words = {'Ill':"I'll", # correct mistakes in punctuation/spacing/etc
 'cant':"can't",
 'couldnt':"couldn't",
 'didnt':"didn't",
 'doesnt':"doesn't",
 'dont':"don't",
 'friends':"friend's",
 'mans':"man's",
 'shouldnt':"shouldn't",
 'wasnt':"wasn't",
 'wouldnt':"wouldn't",
 'youll':"you'll"}

# fix problem with punctuation
for k, v in punct_words.items():
    try:
        s[v] = s[k.lower()]
        cd[v] = cd[k.lower()]
    except KeyError:
        s[v] = np.nan
        cd[v] = np.nan
    
df_all["Subtlex_log10"] = df_all['word'].str.replace('[^\w\s]','').map(s)
df_all["length"] = [len(row.word) for index, row in df_all.iterrows()]

df = df_all[df_all.is_start_end == 0][['cloze_p_smoothed', 'competition','entropy', 'rating_mean', 'rating_s', "rating_sd", 'cloze_s',"s_GPT2",'s_GPT2_medium', 's_GPT2_large', 's_GPT2_xl', 's_GPTNeo_125M', "s_GPTNeo", 's_GPTNeo_2.7B', 'rnn', 'psg', 'bigram', 'trigram', 'tetragram', 'rnn_pos', 'psg_pos', 'bigram_pos', 'trigram_pos', 'tetragram_pos', 'RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time','ELAN', 'LAN', 'N400', 'EPNP', 'P600', 'PNP', "Subtlex_log10", "length", "context_length"]]

df["s_GPTNeo_large"] = df["s_GPTNeo_2.7B"] # renaming column; the dot messes up with statsmodels formula

# formatting variable names for plots
dv_nice_dict = {'RTfirstfix':'FFIx',
                'RTfirstpass':'Fpass',
                'RTgopast':'GoPast', 
                'RTrightbound':'RightBound',
                'self_paced_reading_time':"SPR",
                'ELAN':"ELAN",
                "LAN":'LAN', 
                "N400":'N400',
                "EPNP":'EPNP', 
                "P600":'P600', 
                "PNP":'PNP',
                "average":"average"}

iv_nice_dict = {'rating_mean':'rating',
                'rating_s':'rating$_{H}$',
                'rating_sd':'rating',
                'cloze_s':'cloze$_{H}$',
                'cloze_p_smoothed':'cloze$_{p}$',
                "competition": "competition",
                "entropy":"entropy",
                "number of lemmas": "n° lemmas",
                "psg":"PSG",
                's_GPT2': 'GPT-2$_{124M}$', 
                's_GPT2_medium':'GPT-2$_{355M}$',
                's_GPT2_large':'GPT-2$_{774M}$', 
                's_GPT2_xl':'GPT-2$_{1.5B}$', 
                's_GPTNeo':"GPT-Neo$_{1.3B}$", 
                's_GPTNeo_2.7B':"GPT-Neo$_{2.7B}$",
                "s_GPTNeo_large":"GPT-Neo$_{2.7B}$",
                's_GPTNeo_125M':"GPT-Neo$_{125M}$",
                's_BLOOM':"BLOOM",
                'rnn':"RNN", 
                "PSG":'psg', 
                'bigram':"bigram", 
                "trigram":'trigram', 
                "tetragram":'tetragram'}

##############################################
# compare IVs in predicting cognitive effort #
##############################################

IVs = ['bigram', 'trigram', 'tetragram', "psg", "rnn", "s_GPT2",'s_GPT2_medium', 's_GPT2_large', 's_GPT2_xl', 's_GPTNeo_125M', "s_GPTNeo", 's_GPTNeo_large', "cloze_p_smoothed", "cloze_s", "rating_mean", "rating_s"]#, "competition", "entropy", "number of lemmas"] # removed BLOOM

DVs = ['RTfirstfix', 'RTfirstpass', 'RTrightbound', 'RTgopast', 'self_paced_reading_time','ELAN', 'LAN', 'N400', 'EPNP', 'P600', 'PNP']

compare_iv = []
for dv in DVs:
    # baseline model
    temp = df[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
    # first fit baseline model without predictability
    OLS_baseline = smf.ols(formula= dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data = temp).fit()
    R2_baseline = OLS_baseline.rsquared
    aic_baseline = OLS_baseline.aic
    for iv in IVs:
        temp = df[[iv, dv, "Subtlex_log10", "length", "context_length"]]
        # experimental model with iv
        OLS_model = smf.ols(formula= dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + '+iv, data = temp).fit()
        is_sig = OLS_model.tvalues[iv]
        the_p = OLS_model.pvalues[iv]
        the_B = OLS_model.params[iv]
        R2_model = OLS_model.rsquared
        aic_model = OLS_model.aic
        compare_iv.append([dv, iv, the_B, is_sig, the_p, R2_model, R2_baseline, R2_model-R2_baseline, aic_model-aic_baseline])
compare_iv = pd.DataFrame(compare_iv, columns = ["dv", "iv","b","t","p","R2","R2_baseline", "D_R2", "D_AIC"])


# make table for paper
for index, row in compare_iv.iterrows():
    print(" & ".join([iv_nice_dict[row.iv], dv_nice_dict[row.dv], str(round(row.b, 4)), str(round(row.t, 4)), str(round(row.p, 4)), str(round(row.R2, 4)), str(round(row.R2_baseline, 4)), str(round(row.D_R2, 4)), str(round(row.R2_baseline, 4))])+" \\\\")

# get average by predictor
average = []
for iv in IVs:
    temp = compare_iv[(compare_iv.iv == iv) & (compare_iv.dv.isin(['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time','LAN', 'N400','P600']))]
    #print(temp)
    average.append(["average", iv, np.mean([abs(t) for t in temp.t]), np.mean(temp.D_R2)])
average = pd.DataFrame(average, columns=["dv", "iv","t", "D_R2"])

compare_iv = pd.concat([compare_iv, average])
DVs.append("average")

compare_colors = []  # to color-code the bars in barplot as a function of significance
for index, row in compare_iv.iterrows():
    if 2.581 > abs(row.t) > 1.96:
        compare_colors.append("yellow") # p < .05
    elif 3.300 > abs(row.t) > 2.581:
        compare_colors.append("orange") # p < .01
    elif abs(row.t) > 3.300:
        compare_colors.append("tomato") # p < .001
    else:
        compare_colors.append("darkgrey")
compare_iv["colors"] =  compare_colors

# font formatting
font = {'weight' : 'normal',
        'size'   : 20}
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
matplotlib.rc('font', **font)

fig, axs = plt.subplots(4, 3, figsize=(20, 26), dpi=250)
c = 0
for colvalue in range(4):
    for rowvalue in range(3):
        dv = DVs[c]; c+=1
        y = compare_iv[compare_iv.dv == dv]["D_R2"]
        x = np.arange(1,17)*1.2
        colors = compare_iv[compare_iv.dv == dv]["colors"]
        axs[colvalue, rowvalue].bar(x,y,color = colors)
        varname = dv_nice_dict[dv]
        axs[colvalue, rowvalue].set_title(varname, fontsize="xx-large")
        axs[colvalue, rowvalue].set_ylim(0,0.045)
        axs[colvalue, rowvalue].set_xticks(x)
        axs[colvalue, rowvalue].set_yticks([0, 0.01, 0.02, 0.03, 0.04], labelsize=10)
        xticks = [iv_nice_dict[label] for label in compare_iv[compare_iv.dv == dv]["iv"]]
        [x.set_linewidth(2.5) for x in axs[colvalue, rowvalue].spines.values()]
        axs[colvalue, rowvalue].set_xticklabels(
            xticks,
            rotation=45,
            horizontalalignment='right'
        );
colors = {'$p < .001$':'tomato', 
          '$p < .01$':'orange', 
          '$p < .05$':'yellow', 
          '$p > .05$':'darkgrey'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
fig.legend(handles, labels, bbox_to_anchor=(1.103, 0.96))
fig.text(-0.01, 0.5, '$\mathrm{\Delta R^2}$', ha='center', va='center', rotation="vertical", fontsize="xx-large")
fig.tight_layout(pad=1.4)
