#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


from typing import Generator, Tuple


# ### Read

# In[2]:


import pandas as pd
import numpy as np
import re
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# In[3]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
list(stop_words)[:4]


# ### ML

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns


# ### Rendering

# In[5]:



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# ## Configs

# In[6]:


START = "<?xml version='1.0' encoding='utf-8'?><BODY>"
END = "</BODY>"

DATADIR = "big_data_project/data"


# ## Reading

# In[7]:

print("loading CSVs...")

train_files = pd.read_csv("big_data_project/train.csv", names=["id", "file", "pos"], skiprows=1)


# In[8]:


test_files = pd.read_csv("big_data_project/test.csv", names=["id", "file", "pos"], skiprows=1)


# In[9]:


train_files.head()


# In[10]:


test_files.head()


# In[11]:


def get_files(files: pd.DataFrame) -> Generator[Tuple[str,str], None, None]:
    for file, pos in zip(files["file"], files["pos"]):
        with open(f"{DATADIR}/{file}", "r") as f:
            content = " ".join(f.readlines())[len(START):-len(END)]
        
        yield content, pos


# In[12]:


def get_tests(files: pd.DataFrame) -> Generator[Tuple[str,str,str], None, None]:
    for fid, file in zip(files["id"], files["file"]):
        with open(f"{DATADIR}/{file}", "r") as f:
            content = " ".join(f.readlines())[len(START):-len(END)]
        
        yield fid, file, content


# In[13]:

print("loading files...")


files = [e for e in get_files(train_files)]
files[1]


# In[14]:


X = [x for x,y in files]
y = [y for x,y in files]

X = np.array(X)
y = np.array(y)


# ## Training

# ### Clean Data

# In[15]:


from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()


# In[16]:


def clean_txt(txt: str) -> str:
    # Remove all the special characters
    txt = re.sub(r'\W', ' ', txt)
    
    # remove all single characters
    txt = re.sub(r'\s+[a-zA-Z]\s+', ' ', txt)
    
    # Remove single characters from the start
    txt = re.sub(r'\^[a-zA-Z]\s+', ' ', txt) 
    
    # Substituting multiple spaces with single space
    txt = re.sub(r'\s+', ' ', txt, flags=re.I)
    
    # Removing prefixed 'b'
    txt = re.sub(r'^b\s+', '', txt)
    
    # Converting to Lowercase
    txt = txt.lower()
    
    # Lemmatization
    document = txt.split()

    document = [stemmer.lemmatize(w) for w in document]
    return " ".join(document)


# In[17]:


print("cleaning text...")

Xc = np.array([clean_txt(x) for x in X])


# ### Split data

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=0.2, random_state=0)


# ### Train data

# 

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer()


# In[20]:


print("fitting vectorizer...")

X_train = tfidfconverter.fit_transform(X_train)
X_test = tfidfconverter.transform(X_test)


# In[21]:

print("fitting classifier...")

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)


# In[40]:

print("predicting test values...")

y_pred = classifier.predict_proba(X_test)
y_pred_bool = classifier.predict(X_test)


# In[43]:



print("Confusion matrix:")
print(confusion_matrix(y_test,y_pred_bool))
print(classification_report(y_test,y_pred_bool))
print("Accuracy:", accuracy_score(y_test, y_pred_bool))


# In[45]:


print("producting images...")

cf_matrix = confusion_matrix(y_test, y_pred_bool)
categories = ['Negative', 'Positive']
sns.heatmap(cf_matrix, annot = True, cmap = 'Blues',fmt = '', xticklabels = categories, yticklabels = categories)
plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
plt.savefig("big_data_project/cmatrix.png")


# In[47]:


fpr, tpr, thresholds = roc_curve(y_test, [pos for neg, pos in y_pred])

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.savefig("big_data_project/roc.png")


# ## Usage

# In[26]:


print("predicting test file...")

tfiles = [e for e in get_tests(test_files)]
tfiles[1]


# In[34]:


prediction = classifier.predict(tfidfconverter.transform([content for fid, file, content in tfiles]))
end_test = [(fid, file, pred) for (fid, file, content), pred in zip(tfiles, prediction)]


# In[28]:


end_test_df = pd.DataFrame(end_test, columns=["id","file","earnings: 0 no/ 1 yes"])


# In[29]:


end_test_df.head()


# In[30]:

print("producting output file...")

end_test_df.to_csv("big_data_project/output.csv", index=False)

print("done.")
