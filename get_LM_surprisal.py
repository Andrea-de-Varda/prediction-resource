import pandas as pd
from os import chdir
from researchpy import corr_pair
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from scipy.special import softmax
from tqdm import tqdm
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
from math import e

chdir("/path/to/your/wd/")

df_all = pd.read_csv("all_measures.csv") # created in merge_with_behavioural_data.py

##################
# GPT-2 VARIANTS #
##################

def get_p(sent, w): # gets p of word (w) given context. Relies on model and toker.
    inpts = toker(sent, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inpts).logits[:, -1, :]
    target_id = toker(" "+w)["input_ids"][0]
    p = softmax(logits[0])[target_id].item()
    return p

def get_p2(sent, w): # as get_p if len(toker(w)) == 1; else, sums logP of subword tokens
    inpts = toker(sent, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inpts).logits[:, -1, :]
    target = toker(" "+w)["input_ids"]
    if len(target) == 1:
        target_id = target[0]
        p = softmax(logits[0])[target_id].item()
        return p, 0
    else:
        out_p = []
        target_id = target[0]
        p = softmax(logits[0])[target_id].item()
        out_p.append(p)
        sent = sent+toker.decode(target_id)
        for token in target[1:]:
            t = toker.decode(token)
            p = get_p(sent, t)
            out_p.append(p)
            #print(sent, "--"+t, p)
            sent = sent+t
        p_multi = np.prod(out_p)
        return p_multi, 1

# GPT2 SMALL (124M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("gpt2")

out = []
for index, row in tqdm(df_all.iterrows(), total=len(df_all)):
    p, ismulti = get_p2(row["item"], row["word"])
    out.append([p, ismulti])

out = pd.DataFrame(out, columns=["p", "is_multitoken"])
df_all["is_multitoken_GPT2"] = out["is_multitoken"]
df_all["s_GPT2"] = -np.log(out["p"]) # e as base of log

# GPT2 MEDIUM (355M parameter version of GPT-2; GPT-2 had 124M)
model = AutoModelForCausalLM.from_pretrained('gpt2-medium', return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained('gpt2-medium')

out = []
for index, row in tqdm(df_all.iterrows(), total=1726):
    p, ismulti = get_p2(row["item"], row["word"])
    out.append([p, ismulti])
    
out = pd.DataFrame(out, columns=["p", "is_multitoken"])
df_all["is_multitoken_GPT2_medium"] = out["is_multitoken"]
df_all["s_GPT2_medium"] = -np.log(out["p"])

# GPT2 LARGE (774M parameter)
model = AutoModelForCausalLM.from_pretrained('gpt2-large', return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained('gpt2-large')

out = []
for index, row in tqdm(df_all.iterrows(), total=1725):
    p, ismulti = get_p2(row["item"], row["word"])
    out.append([p, ismulti])
    
out = pd.DataFrame(out, columns=["p", "is_multitoken"])
df_all["is_multitoken_GPT2_large"] = out["is_multitoken"]
df_all["s_GPT2_large"] = -np.log(out["p"])

# GPT2 XL (1.5B parameter)
model = AutoModelForCausalLM.from_pretrained('gpt2-xl', return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained('gpt2-xl')

out = []
for index, row in tqdm(df_all.iterrows(), total=1725):
    p, ismulti = get_p2(row["item"], row["word"])
    out.append([p, ismulti])
    
out = pd.DataFrame(out, columns=["p", "is_multitoken"])
df_all["is_multitoken_GPT2_xl"] = out["is_multitoken"]
df_all["s_GPT2_xl"] = -np.log(out["p"])

###########
# GPT-Neo #
###########

# 125M
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

out = []
for index, row in tqdm(df_all.iterrows(), total=1725):
    p, ismulti = get_p2(row["item"], row["word"])
    out.append([p, ismulti])

out = pd.DataFrame(out, columns=["p", "is_multitoken"])
df_all["is_multitoken_GPTNeo_125M"] = out["is_multitoken"]
df_all["s_GPTNeo_125M"] = -np.log(out["p"])

# 1.3 billion
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

out = []
for index, row in tqdm(df_all.iterrows(), total=1725):
    p, ismulti = get_p2(row["item"], row["word"])
    out.append([p, ismulti])

out = pd.DataFrame(out, columns=["p", "is_multitoken"])
df_all["is_multitoken_GPTNeo"] = out["is_multitoken"]
df_all["s_GPTNeo"] = -np.log(out["p"])

# 2.7 billion
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", return_dict_in_generate=True)
toker = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

out = []
for index, row in tqdm(df_all.iterrows(), total=1725):
    p, ismulti = get_p2(row["item"], row["word"])
    out.append([p, ismulti])

out = pd.DataFrame(out, columns=["p", "is_multitoken"])
df_all["is_multitoken_GPTNeo_2.7B"] = out["is_multitoken"]
df_all["s_GPTNeo_2.7B"] = -np.log(out["p"])

SAVING 

df_all.to_csv("all_measures.csv", index=False) # overwrites previous df
