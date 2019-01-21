#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
import random
import re
import requests
from sklearn import metrics
from sklearn.model_selection import KFold


# In[ ]:


# Impute missing values

# fare => lowest : missing fare was 3rd class old Mr. w/out cabin, alone, ticket 4
impute_fare = lambda df: df.fare.replace('missing', 9, inplace=True)

# age => mode age for group with same pclass honor sisp pach cabin
def impute_age(df):
    bin_age = bins([13, 21, 26, 32, 40, 50])
    features = ['pclass', 'honor', 'sisp', 'pach', 'cabin', 'age']
    base = df[df.age != 'missing'][features]
    missing = df[df.age == 'missing'][features]
    for pid, obs in missing.iterrows():
        pclass, honor, sisp, pach, cabin, _ = obs
        matches = base.query('pclass == @pclass and honor == @honor and sisp == @sisp and pach == @pach and cabin == @cabin')
        if matches.empty:
            matches = base[(base.pclass == pclass) & (base.honor == honor)]
        mean_age = int(round(matches.age.mean()))
        mean_age = bin_age(mean_age)
        df.loc[pid, 'age'] = mean_age

# Derived features

# Traveling with any family?
def add_alone(df):
    df['alone'] = (df.sisp + df.pach) == 0
    return df

# Give female children separate honorific
# honor == 'miss' and age == '<12'  ==> honor = 'lass'
def add_girls_honor(df):
    girls = df.query('honor == "miss" and age == 13')
    for pid in girls.index:
        df.loc[pid, 'honor'] = 'lass'

# Better class: combine pclass and cabin
def add_clare(df):
    a = df.query('fare == 9')
    b = df.query('pclass == 3 and fare > 9')
    c = df.query('pclass < 3 and fare > 9 and fare < 40')
    d = df.query('pclass < 3 and fare >= 40')
    a.insert(1, 'clare', 1)
    b.insert(1, 'clare', 2)
    c.insert(1, 'clare', 3)
    d.insert(1, 'clare', 4)
    res = pd.concat([a, b, c, d], axis=0)
    if res.shape[0] != df.shape[0]:
        print('Clare error no!!!')
    return res

def adjust_features(old_df):
    df = old_df.copy(deep=True)
    impute_fare(df)
    impute_age(df)
    add_girls_honor(df)
    df = add_alone(df)
    df = add_clare(df)
    return df


# In[ ]:


# Conversion maps for cleaning data

def bins(uppers):
    def conversion(v):
        last = uppers[-1] + 1
        if v == 'missing':
            return v
        for i, upper in enumerate(uppers):
            if v < upper:
                return upper
        return last
    return conversion

honorifics = {
    'Mr' : 'mr',
    'Don': 'mr', 'Sir': 'mr', 'Rev': 'mr', 'Dr': 'mr', 'Major': 'mr', 'Col': 'mr', 'Capt': 'mr', 'Jonkheer': 'mr',
    'Master': 'master',
    'Mlle': 'miss', 'Ms': 'miss', 'Miss': 'miss',
    'Mrs': 'mrs', 'Mme': 'mrs', 'Dona': 'mrs', 'Lady': 'mrs', 'the Countess': 'mrs'
}

with open('name_origins.pickle', 'rb') as file:
    first_name_origins, last_name_origins = pickle.load(file)

eng_origin = {'English', 'Irish', 'Cornish', 'Welsh', 'Scottish'}

def split_name(name):
    match = re.search(r'(\w+), ([\w+ ]+)\. \(?(\w+)', name)
    if not match:
        print('Regex Error: ', name)
        return None
    honor = honorifics[match.group(2)]
    lorigin = last_name_origins[match.group(1)]
    forigin = first_name_origins[match.group(3)]
    last = 'eng' if lorigin in eng_origin else 'other'
    first = 'eng' if forigin in eng_origin else 'other'
    if first == last:
        origin = first
    elif first == 'unknown':
        origin = last
    elif last == 'unknown':
        origin = first
    else:
        origin = first
    return (honor, origin)


# In[ ]:


# Pickle results and don't repeat unless necessary!
# Make conversion maps for name origins
def make_name_maps():
    names = list(input_values['name'])
    fnames = set()
    lnames = set()
    for name in names:
        match = re.search(r'(\w+),[\w+ ]+\. \(?(\w+)', name)
        if match:
            lnames.add(match.group(1))
            fnames.add(match.group(2))
        else:
            print('Regex Error: ', name)
    fnames = list(fnames)
    lnames = list(lnames)
    forigins = defaultdict(dict)
    lorigins = defaultdict(dict)
    with requests.session() as s:
        for name in fnames:
            forigins[name] = name_origin(name, s, surname=0)
        for name in lnames:
            lorigins[name] = name_origin(name, s, surname=1)
    return (forigins, lorigins)

def name_origin(name, session, surname=0):
    url = ('https://www.behindthename.com/name/', 'https://surnames.behindthename.com/name/')[surname]
    response = session.get(url + name.strip(), stream=True)
    if response.status_code != 200:
        return 'unknown'
    raw_html = response.content
    html = BeautifulSoup(raw_html, 'html.parser')
    usage = html.find('a', attrs={'class': 'usg'})
    if not usage:
        return 'unknown'
    usage = re.sub(r' \(.*\)', '', usage.string)
    split = usage.split()
    if len(split) > 1:
        usage = split[-1]
    return usage

#forig, lorig = make_name_maps()
#to_save = [forig, lorig]
#with open('name_origins.pickle', 'wb') as dump:
#    pickle.dump(to_save, dump)


# In[ ]:


counts = lambda df: df.apply(pd.Series.value_counts, axis=0)

def get_errors(X, y, model):
    errors = []
    for i in range(X.shape[0]):
        obs = X.iloc[i:i+1]
        real = y.iloc[i]
        y_pred = model.predict(obs)
        if y_pred != [real]:
            errors.append(i)
    errs = pd.concat([X.iloc[errors], y.iloc[errors]], axis=1, join='outer')
    print('Errors:', errs.shape[0])
    errs.sort_values('survived', inplace=True)
    return errs

def test_data(X, y, title, clf, sampler=None, splits=3):
    print('\n', '='*10, title, '='*10)
    kfold = KFold(n_splits=splits)
    for train_i, test_i in kfold.split(X):
        X_train, X_test = X.iloc[train_i], X.iloc[test_i]
        y_train, y_test = y.iloc[train_i], y.iloc[test_i]
        if sampler:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        model = clf.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(' '.join('{0: <7.7}{1}'.format(k, v) for v, k in sorted(zip(model.feature_importances_, X.columns), reverse=True)))
        print(metrics.accuracy_score(y_test, preds))
        test_survived, test_size = sum(y_test), y_test.shape[0]
        print(test_survived, test_size, test_survived//test_size)
        print(metrics.classification_report(y_test, preds))

def sample_features(X, y, clf, population, repeat, fixed=None, size=6):
    for _ in range(repeat):
        sample = random.sample(population, size)
        if fixed is not None:
            sample.append(fixed)
        X_sample = X.iloc[:, sample]
        test_data(X_sample, y, clf)

