# re-doing pre-processing from scratch
from os import chdir
import pandas as pd
import numpy as np
import collections
import re
from spellchecker import SpellChecker
from scipy.stats import pearsonr
from tqdm import tqdm
from heapq import nlargest
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

chdir("/path/to/your/wd")

item_set = pd.read_csv("item-set.csv")
item_set["sentence"] = item_set["sentence"].str.strip()

item_1 = item_set[item_set["list"] == 1] # dividing by list
item_2 = item_set[item_set["list"] == 2]
item_3 = item_set[item_set["list"] == 3]
item_4 = item_set[item_set["list"] == 4]
item_5 = item_set[item_set["list"] == 5]
item_6 = item_set[item_set["list"] == 6]
item_7 = item_set[item_set["list"] == 7]
item_8 = item_set[item_set["list"] == 8]

trgt_compregension = ['get off the ground suddenly', 'will not wait happily', 'child that is determined to do what it wants','small dogs with long ears', 'stuck through with a sharp instrument', 'an integrated human-machine system', 'at its peak of success', 'the faint color of her skin', 'sliding box', 'worried and puzzled']

###########
# RATINGS #
###########

def clean_rating_questions(series): # remove text of the question
    qs = []
    for i, x in enumerate(series):
        q = re.sub(' - How much would you expect to read the word  "\[Field-3\]"  as the next word of this sentence fragment:\n\n\n\[Field-1\]...', "", x).strip()
        qs.append(q)
    print("N° items =", len(qs))
    return pd.Series(qs)

def clean_ratings_responses(df): # remove "Very much" and "Not at all" from rating responses; convert to integer
    cleaned = df.copy(deep=True)
    cols = df.columns
    for index, row in df.iterrows():
        for col in cols:
            r = re.sub("- Very much", "", row[col])
            r = re.sub("- Not at all", "", r)
            r = int(r)
            cleaned.loc[index, col] = r
    return cleaned

# IMPORTANT -- there was a mistake in the encoding ("2 - very much" instead of "5 - very much") in list 5. Correlation with cloze probability data
# shows that it is not a problem (also, it was positioned at the end of the likert scale).

def clean_ratings_responses_list5(df):
    cleaned = df.copy(deep=True)
    cols = df.columns
    for index, row in df.iterrows():
        for col in cols:
            r = re.sub("2 - Very much", "5", row[col])
            r = re.sub("- Not at all", "", r)
            r = int(r)
            cleaned.loc[index, col] = r
    return cleaned

def load_clean_ratings(listnum, items):
    ratings = pd.read_excel("ratings/list"+str(listnum)+".xlsx")
    part_comprehension = ratings.iloc[1:, 17:28]
    correctness = part_comprehension.iloc[:, 1:] == trgt_compregension
    part_comprehension["score"] = correctness.sum(axis=1)
    to_exclude = set(part_comprehension[part_comprehension.score < 8].iloc[:, 0])
    print(len(to_exclude), "participants excluded")
    ratings = ratings[~ratings.iloc[:, 17].isin(to_exclude)]
    qs_ratings = clean_rating_questions(ratings.iloc[0,28:])
    resp_ratings = ratings.iloc[1:, 28:]
    if listnum == 5:
        ratings_clean = clean_ratings_responses_list5(resp_ratings)
    else:
        ratings_clean = clean_ratings_responses(resp_ratings)
    # print(ratings_clean)
    rating_mean = list(ratings_clean.mean(axis=0))
    rating_std = list(ratings_clean.std(axis=0))
    print("Check =", list(qs_ratings) == list(items["sentence"]))
    return rating_mean, rating_std

rating_mean_1, rating_std_1 = load_clean_ratings(1, item_1) # 0 participants excluded
rating_mean_2, rating_std_2 = load_clean_ratings(2, item_2) # 1 participants excluded
rating_mean_3, rating_std_3 = load_clean_ratings(3, item_3) # 4 participants excluded
rating_mean_4, rating_std_4 = load_clean_ratings(4, item_4) # 2 participants excluded
rating_mean_5, rating_std_5 = load_clean_ratings(5, item_5) # 3 participants excluded
rating_mean_6, rating_std_6 = load_clean_ratings(6, item_6) # 3 participants excluded
rating_mean_7, rating_std_7 = load_clean_ratings(7, item_7) # 0 participants excluded
rating_mean_8, rating_std_8 = load_clean_ratings(8, item_8) # 1 participants excluded

#####################
# CLOZE PROBABILITY #
#####################

# spellcheck
spell = SpellChecker()

def clean_cloze_questions(series):
    qs = []
    for i, x in enumerate(series):
        q = re.sub("\[Field-1\]\.\.\.", "", x) # remove unnecessary string stuff
        q = re.sub("- Write the next word of the sentence:", "", q).strip()
        qs.append(q)
    print("N° items =", len(qs))
    return pd.Series(qs)
    
def clean_cloze_responses(df):
    cols = df.columns
    out = []
    corrected = df.copy(deep=True)
    logs = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        for col in cols:
            try:
                w = row[col].split(" ")[0].lower() # some participants wrote more than 1 word
                w = re.sub("’", "'", w)
                w_spell = spell.correction(w)
                if w_spell:
                    w_spell = w_spell
                else:
                    w_spell = w
                if w_spell != w:
                    logs.append([index, col, w, w_spell])
                corrected.loc[index, col] = w_spell
            except AttributeError: # empty string, only space
                pass
    logs = pd.DataFrame(logs, columns = ["idx", "Q_num", "word", "corrected"])
    #print("\n\nPercentage corrected =", round((logs.size/corrected.size)*100, 4), "%")
    print("\n\nCorrected ", logs.size, "out of", corrected.size, "words") 
    return corrected, logs

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

# Laplacian smoothing -- ( O + a ) / ( N + aK )
a = 1
def get_p_smoothed(candidates, target, return_all = False):
    target = re.sub("[\.\,\?\!\:\;]", "", target)
    if target in punct_words.keys():
        target = punct_words[target]
    d_temp = {}
    K = len(set(candidates))+1
    N = len(candidates)
    p_unknown = 1/(N+(a*K))
    d_temp["unk"] = p_unknown
    for word in candidates:
        O = candidates.count(word)
        p = (O+a)/N+a*K
        d_temp[word] = (O+a)/(N+(a*K))
    if return_all:
        return d_temp
    else:
        if target in d_temp.keys():
            cloze_p = d_temp[target]
        else:
            cloze_p = d_temp["unk"]
        top_word = nlargest(2, d_temp, key=d_temp.get)
        if top_word[0] == target:  # if the top word is equal to the tword, ignore it...
            top_freq = d_temp[top_word[1]]  # ... and pick the second largest
        else:  # otherwise pick the top word
            top_freq = d_temp[top_word[0]]
        compet = cloze_p - top_freq
        entropy = -sum([p*np.log(p) for p in d_temp.values()])
        cloze_s = -np.log(cloze_p)
        return cloze_p, cloze_s, compet, entropy

def load_clean_cp(listnum, items):
    cloze = pd.read_excel("cp/list"+str(listnum)+".xls")
    part_comprehension = cloze.iloc[1:, 17:28]
    correctness = part_comprehension.iloc[:, 1:] == trgt_compregension
    part_comprehension["score"] = correctness.sum(axis=1)
    to_exclude = set(part_comprehension[part_comprehension.score < 8].iloc[:, 0])
    cloze = cloze[~cloze.iloc[:, 17].isin(to_exclude)]
    qs_cloze = clean_cloze_questions(cloze.iloc[0,28:])
    print("Check =", list(qs_cloze) == list(items["sentence"]))
    resp_cloze = cloze.iloc[1:, 28:]
    spelled, logs = clean_cloze_responses(resp_cloze)
    spell_transposed = spelled.transpose()
    cloze_probabilities = []; cloze_surprisal = []; competition = []; entropy = []
    for n in range(len(spell_transposed)):
        candidates = list(spell_transposed.iloc[n,:])
        target = list(items["word"])[n].strip()
        p_smoothed, cloze_s, compet, H = get_p_smoothed(candidates, target)
        cloze_probabilities.append(p_smoothed)
        cloze_surprisal.append(cloze_s)
        competition.append(compet)
        entropy.append(H)
    print(len(to_exclude), "participants excluded")
    return logs, cloze_probabilities, cloze_surprisal, competition, entropy
    
logs1, cloze_probabilities_1, clz_h1, comp_1, h1 = load_clean_cp(1, item_1) # corrected = 2.5    %, 1 participants excluded
logs2, cloze_probabilities_2, clz_h2, comp_2, h2 = load_clean_cp(2, item_2) # corrected = 2.8752 %, 4 participants excluded
logs3, cloze_probabilities_3, clz_h3, comp_3, h3 = load_clean_cp(3, item_3) # corrected = 5.1852 %, 1 participants excluded
logs4, cloze_probabilities_4, clz_h4, comp_4, h4 = load_clean_cp(4, item_4) # corrected = 4.4304 %, 2 participants excluded
logs5, cloze_probabilities_5, clz_h5, comp_5, h5 = load_clean_cp(5, item_5) # corrected = 3.9886 %, 2 participants excluded
logs6, cloze_probabilities_6, clz_h6, comp_6, h6 = load_clean_cp(6, item_6) # corrected = 3.8249 %, 0 participants excluded
logs7, cloze_probabilities_7, clz_h7, comp_7, h7 = load_clean_cp(7, item_7) # corrected = 4.321  %, 2 participants excluded
logs8, cloze_probabilities_8, clz_h8, comp_8, h8 = load_clean_cp(8, item_8) # corrected = 4.469  %, 1 participants excluded

logs1.to_csv("logs/logs1.csv") # save to csv the list of corrections
logs2.to_csv("logs/logs2.csv")
logs3.to_csv("logs/logs3.csv")
logs4.to_csv("logs/logs4.csv")
logs5.to_csv("logs/logs5.csv")
logs6.to_csv("logs/logs6.csv")
logs7.to_csv("logs/logs7.csv")
logs8.to_csv("logs/logs8.csv")

# correlations ratings - cloze
pearsonr(rating_mean_1, cloze_probabilities_1) # 0.6787567441749423
pearsonr(rating_mean_2, cloze_probabilities_2) # 0.6426763410673153
pearsonr(rating_mean_3, cloze_probabilities_3) # 0.6676436009576135
pearsonr(rating_mean_4, cloze_probabilities_4) # 0.5698478118865321
pearsonr(rating_mean_5, cloze_probabilities_5) # 0.6723058125994421
pearsonr(rating_mean_6, cloze_probabilities_6) # 0.7035257030715293
pearsonr(rating_mean_7, cloze_probabilities_7) # 0.6247808761704174
pearsonr(rating_mean_8, cloze_probabilities_8) # 0.6096548812893455

# create whole df
def make_df(items, rating_m, rating_sd, clozep, clozeh, comp, H):
    items["rating_mean"] = rating_m
    items["rating_sd"] = rating_sd
    items["cloze_p_smoothed"] = clozep
    items["cloze_s"] = clozeh
    items["competition"] = comp
    items["entropy"] = H
    return items

list1 = make_df(item_1, rating_mean_1, rating_std_1, cloze_probabilities_1, clz_h1, comp_1, h1)
list2 = make_df(item_2, rating_mean_2, rating_std_2, cloze_probabilities_2, clz_h2, comp_2, h2)
list3 = make_df(item_3, rating_mean_3, rating_std_3, cloze_probabilities_3, clz_h3, comp_3, h3)
list4 = make_df(item_4, rating_mean_4, rating_std_4, cloze_probabilities_4, clz_h4, comp_4, h4)
list5 = make_df(item_5, rating_mean_5, rating_std_5, cloze_probabilities_5, clz_h5, comp_5, h5)
list6 = make_df(item_6, rating_mean_6, rating_std_6, cloze_probabilities_6, clz_h6, comp_6, h6)
list7 = make_df(item_7, rating_mean_7, rating_std_7, cloze_probabilities_7, clz_h7, comp_7, h7)
list8 = make_df(item_8, rating_mean_8, rating_std_8, cloze_probabilities_8, clz_h8, comp_8, h8)

pdList = [list1, list2, list3, list4, list5, list6, list7, list8]
final = pd.concat(pdList, ignore_index =True)

final.to_csv("ratings_and_cloze.csv", index=False)
