#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn import pipeline
sklearn.pipeline.Pipeline
import numpy as np


# In[2]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
#engine = create_engine('sqlite:///InsertDatabaseName.db')
#df[['message']].shape
#pd.read_sql_table('InsertTableName','sqlite:///InsertDatabaseName.db')


# In[3]:


# load data from database
#engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName','sqlite:///InsertDatabaseName.db')
X = df.drop('genre', axis =1)
Y = df[['genre']]


# ### 2. Write a tokenization function to process your text data

# In[4]:


X = (X['message'].tolist())


# In[5]:


def tokenize(data):
    for i in range(0,len(data)):
        tokens = word_tokenize(data[i])
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
        return clean_tokens


# In[ ]:





# In[ ]:





# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, Y)

vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
clf = RandomForestClassifier()

X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)

X_test_counts = vect.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)


# In[7]:


#display_results(y_test, y_pred)
#print(len(y_test))
#print(len(y_pred))

labels = np.unique(y_pred)
confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
(y_pred == y_test.values).mean()
#print(type(y_pred))
#print(type(y_test))
#y_test.values


# In[ ]:





# In[8]:


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test.values).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


def main():
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf = RandomForestClassifier()

    # train classifier
    X_train_counts = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    clf.fit(X_train_tfidf, y_train)

    # predict on test data
    X_test_counts = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)

    # display results
    display_results(y_test, y_pred)


main()


# In[9]:


import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, Y)
print(np.shape(X_train)[0])
np.shape(y_train)[0]


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[ ]:





# In[10]:


def main():
    #X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    # build pipeline
    pipeline = sklearn.pipeline.Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

      
        
    # train classifier
    pipeline.fit(X_train,y_train)
    # predict on test data

    # display results
    display_results(y_test, y_pred)
main()  


# In[91]:


'''labels = np.unique(y_pred)
confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
(y_pred == y_test.values).mean()


# In[ ]:





# In[ ]:





# In[92]:


'''X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,random_state=20)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

#X.shape[0] == Y.shape[0]
len(X)==len(Y)'''


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[104]:


len(y_pred)


# In[105]:


len(y_test)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[11]:


sklearn.metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None)


# In[12]:


nltk.sent_tokenize(X[0])


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[115]:


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        '''sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False'''
        for i in range(0,len(X)):
            try:
                sentence_list = nltk.sent_tokenize(X[i])
                if (sentence_list == []):
                    pass
                pos_tags = nltk.pos_tag(tokenize(sentence_list[0]))  
                if (len(pos_tags)==0):
                    pass
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
                else:
                    return False        
            except:
                ""
      

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# In[106]:


for i in range(0,len(X)):
    try:
        sentence_list = nltk.sent_tokenize(X[i])
        if (sentence_list == []):
            pass
        pos_tags = nltk.pos_tag(tokenize(sentence_list[0]))  
        if (len(pos_tags)==0):
            pass
        first_word, first_tag = pos_tags[0]
        if first_tag in ['VB', 'VBP'] or first_word == 'RT':
            print('t')
        else:
            print(first)        
    except:
        ""
        


# In[108]:


first_tag


# In[ ]:





# In[ ]:





# In[33]:


#sentence_list = nltk.sent_tokenize(X[0])

pos_tags = nltk.pos_tag(tokenize(sentence_list[0]))
pos_tags
first_word, first_tag = pos_tags[0]

#sentence = 0
#pos_tags = nltk.pos_tag(tokenize(sentence_list[sentence]))
#print(sentence_list)


# In[116]:


def build_model():
    pipeline = sklearn.pipeline.Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def display_results(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test.values).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)


def main():
    #X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    #model = build_model()
    #model.fit(X_train, y_train)
    build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    display_results(model, y_test, y_pred)

main()


# In[153]:


#(y_train.astype(str).values.tolist()[0])
#y_train.astype(str)
#build_model().fit(X_train, y_train.astype(str).values.tolist())
#build_model().fit(X_train, y_train.astype(str))
#str(y_train.values[0])
#type(X_train[0])


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


X_train


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




