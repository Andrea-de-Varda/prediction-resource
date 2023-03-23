import pandas as pd
from os import chdir
from researchpy import corr_pair
import numpy as np
import torch
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.preprocessing import normalize
import scipy.io
from math import e

chdir("/path/to/your/wd")

df = pd.read_csv("ratings_and_cloze.csv") # output of preprocessing.py
df.columns = ['item', 'word', 'word2', 'sent_id', 'item_id', 'list',
       'rating_mean', 'rating_sd', 'cloze_p_smoothed', 'cloze_s',
       'competition', 'entropy']

#######################
# add data from Frank #
#######################

mat = scipy.io.loadmat('/your/path/to/Frank/EEG/data/stimuli_erp.mat')
mat.keys()

# Frank has probabilities from 10 increasingly large corpora in RNN
# N-gram size: 2, 3, 4

out = []
for idx, sentence in enumerate(mat["sentences"]):
    tokens = sentence[0][0]
    sentence = [t[0] for t in tokens]
    ERP = mat["ERP"][idx][0]
    s_rnn = mat["surp_rnn"][idx][0] # 10 models
    s_rnn_pos = mat["surp_pos_rnn"][idx][0]
    s_psg = mat["surp_psg"][idx][0] # 9 models
    s_psg_pos = mat["surp_pos_psg"][idx][0]
    s_ngram = mat["surp_ngram"][idx][0] # 3 models (*n*-grams)
    s_ngram_pos = mat["surp_pos_ngram"][idx][0]
    artefact = mat["artefact"][idx][0]
    ERP[np.where(artefact == 1)] = np.nan # filter artefacts following Frank cleaning
    print(ERP.shape[0] == len(sentence))
    for index in range(len(sentence)):
        word = sentence[index]
        erp = ERP[index]
        ELAN, LAN, N400, EPNP, P600, PNP = np.nanmean(erp, axis=0) # average across participants; excluded NaN (artifacts)
        rnn = s_rnn[index][9]
        rnn_pos = s_rnn_pos[index][9]
        psg = s_psg[index][8]
        psg_pos = s_psg_pos[index][8]
        bigram, trigram, tetragram = s_ngram[index]
        bigram_pos, trigram_pos, tetragram_pos = s_ngram_pos[index]
        item = sentence[:index]
        out.append([" ".join(sentence).strip(), index, " ".join(item), word, ELAN, LAN, N400, EPNP, P600, PNP, rnn, rnn_pos, psg, psg_pos, bigram, trigram, tetragram, bigram_pos, trigram_pos, tetragram_pos])
        
frank = pd.DataFrame(out, columns = ["sentence", "context_length", "item", "word", "ELAN", "LAN", "N400", "EPNP", "P600", "PNP", "rnn", "rnn_pos", "psg", "psg_pos", "bigram", "trigram", "tetragram", "bigram_pos", "trigram_pos", "tetragram_pos"])
corr_pair(frank[["N400", "rnn", "rnn_pos", "psg", "psg_pos", "bigram", "trigram", "tetragram"]])

# to merge the dfs, find complete sentence
sent_d = {}
for index, row in df.iterrows():
    if "." in row.word:
        sent_d[row.sent_id] = " ".join([row["item"], row.word]).strip()
df["sentence"] = df["sent_id"].map(sent_d)

# there is a mismatch to correct (punctuation - dont vs don't). These datapoints are excluded in the analyses, but must be fixed to merge the dataframes.
# first correcting in sentence
len(set.intersection(set(frank["sentence"]), set(df.sentence)))

d_punct = {sent_df:sent_df for sent_df in sent_d.values()} # initialize with sent_df

a = sorted(list(set(df["sentence"]).difference(set(frank["sentence"])))) 
b = sorted(list(set(frank["sentence"]).difference(set(df["sentence"]))))

for idx, (sent_df, sent_frank) in enumerate(zip(a, b)):
    d_punct[sent_df] = sent_frank
    c = set(sent_df.split()).difference(set(sent_frank.split()))
    if len(c) > 1:
        print(idx, sent_df, sent_frank)

# manual fix of punct
d_punct["If I have time at the end Ill fill you in on what happened."] = "If I have time at the end I'll fill you in on what happened."
d_punct["Ill break your neck if you dont fix that water."] = "I'll break your neck if you don't fix that water."
d_punct["Ill give him more than a slap."] = "I'll give him more than a slap."
df["sentence"] = df["sentence"].map(d_punct)

len(set.intersection(set(frank["sentence"]), set(df.sentence))) # now it's complete

# then correcting words
d_punct_words = {word_df:word_df for word_df in set(df.word)}
punct_words = {'Ill':"I'll",
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
 'youll':"you'll",
 'Youre':"You're",
 "Dont":"Don't",
 "Hes":"He's",
 "Youll":"You'll",
 "Theyre":"They're",
 "Lets":"Let's",
 "Weve":"We've"}

d_punct_words.update(punct_words)
df["word"] = df["word"].map(d_punct_words)

# then correcting items
items_punct = []
for index, row in df.iterrows():
    item = row["item"]
    temp = []
    x = item.split(" ")
    for w in x:
        try:
            temp.append(d_punct_words[w])
        except KeyError:
            temp.append(w)
    newitem = " ".join(temp).strip()
    items_punct.append(newitem)
df["item"] = items_punct

# merge dataframes
df_frank = pd.merge(df, frank, left_on=["sentence", "word", "item"], right_on = ["sentence", "word", "item"], how="left")

#df_frank["context_length"] = [len(row["item"].split(" ")) for index, row in df_frank.iterrows()]

######################
# EYE-TRACKING & SPR #
######################

eye = pd.read_csv('/your/path/to/UCL_ET_Corpus/eyetracking.RT.txt', sep="\t")
sentences = pd.read_csv('/your/path/to/UCL_ET_Corpus/stimuli.txt', sep="\t", encoding='cp1252')
eye = eye.groupby(["sent_nr", "word_pos"], as_index=False, sort=False).agg({'RTfirstfix':"mean", 'RTfirstpass':"mean", 'RTrightbound':"mean", 'RTgopast':"mean", "word":"max"})

sent_map = {row.sent_nr:row.sentence.strip() for index, row in sentences.iterrows()}
eye["sentence"] = eye["sent_nr"].map(sent_map)
eye["word"] = [w.strip() for w in eye.word]
eye["context_length"] = eye.word_pos-1
df_eye_erp = pd.merge(df_frank, eye, on=["sentence", "word", "context_length"], how="left")

spr =  pd.read_csv('/your/path/to/UCL_ET_Corpus/selfpacedreading.RT.txt', sep="\t")
spr = spr.groupby(["sent_nr", "word_pos"], as_index=False, sort=False).agg({'RT':"mean", "word":"max"})
spr["sentence"] = spr["sent_nr"].map(sent_map)
spr["word"] = [w.strip() for w in spr.word]
spr["context_length"] = spr.word_pos-1
df_eye_erp_spr = pd.merge(df_eye_erp, spr, on=["sentence", "word", "context_length"], how="left")

df_eye_erp_spr["self_paced_reading_time"] = df_eye_erp_spr["RT"]
df_eye_erp_spr.columns

df_all = df_eye_erp_spr[['item', 'word', 'word2', 'sentence', 'context_length', 'sent_id', 'item_id', 'list', 'rating_mean','rating_sd', 'cloze_p_smoothed', 'cloze_s', 'competition', 'entropy','ELAN', 'LAN', 'N400', 'EPNP', 'P600','PNP', 'RTfirstfix', 'RTfirstpass', 'RTrightbound', 'RTgopast','self_paced_reading_time', 'rnn', 'rnn_pos', 'psg', 'psg_pos', 'bigram', 'trigram','tetragram', 'bigram_pos', 'trigram_pos', 'tetragram_pos']]

df_all.to_csv("all_measures.csv", index=False)
