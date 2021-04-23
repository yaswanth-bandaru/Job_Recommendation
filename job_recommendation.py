## Connect to gdrive to get the datasets

"""

from google.colab import drive
drive.mount('/content/gdrive')

"""

## Importing libraries"""

import pandas as pd
import numpy as np
import nltk

"""## Load the data

Loading the jobs dataset
"""

df_job = pd.read_csv("Combined_Jobs_Final.csv")

"""## Exploratory Data Analysis

#### Check NA's
"""

df_job.isnull().sum()

"""From the above list we see that there are lot of NaN values, perform data cleaning for each and every column

#### Selecting the columns for the jobs corpus

For this example we only consider the columns: 'Job.ID', 'Title', 'Position', 'Company','City', 'Empl_type','Edu_req','Job_Description'
"""

cols = ['Job.ID', 'Title', 'Position', 'Company', 'City', 'Employment.Type', 'Job.Description']
df_job =df_job[cols]
df_job.columns = ['Job.ID', 'Title', 'Position', 'Company','City', 'Empl_type','Job_Description']
df_job.head()

# checking for the null values again.
df_job.isnull().sum()

df_nan_city = df_job[pd.isnull(df_job['City'])]
print(df_nan_city.shape)
df_nan_city.head()

df_nan_city.groupby(['Company'])['City'].count()

"""#### We see that there are only 9 companies cities that are having NaN values so it must be manually adding their head quarters (by simply searching at google)

"""

#replacing nan with thier headquarters location
df_job['Company'] = df_job['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')
df_job.loc[df_job['Company'] == 'CHI Payment Systems', 'City'] = 'Illinois'
df_job.loc[df_job['Company'] == 'Academic Year In America', 'City'] = 'Stamford'
df_job.loc[df_job['Company'] == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
df_job.loc[df_job['Company'] == 'Driveline Retail', 'City'] = 'Coppell'
df_job.loc[df_job['Company'] == 'Educational Testing Services', 'City'] = 'New Jersey'
df_job.loc[df_job['Company'] == 'Genesis Health System', 'City'] = 'Davennport'
df_job.loc[df_job['Company'] == 'Home Instead Senior Care', 'City'] = 'Nebraska'
df_job.loc[df_job['Company'] == 'St. Francis Hospital', 'City'] = 'New York'
df_job.loc[df_job['Company'] == 'Volvo Group', 'City'] = 'Washington'
df_job.loc[df_job['Company'] == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'

df_job.isnull().sum()

#The employement type NA are from Uber so I assume as part-time and full time
df_nan_emp = df_job[pd.isnull(df_job['Empl_type'])]
df_nan_emp.head()

#replacing na values with part time/full time
df_job['Empl_type']=df_job['Empl_type'].fillna('Full-Time/Part-Time')
df_job.groupby(['Empl_type'])['Company'].count()
df_job.head()

"""##  Creating the jobs corpus

#### combining the columns of position, company, city, emp_type and position
"""

df_job["text"] = df_job["Position"].map(str) + " " + df_job["Company"] +" "+ df_job["City"]+ " "+df_job['Empl_type']+" "+df_job['Job_Description'] +" "+df_job['Title']
df_job.head(2)

df_all = df_job[['Job.ID', 'text', 'Title']]

df_all = df_all.fillna(" ")

df_all.head()

from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2   
  
def clean_txt(text):
  clean_text = []
  clean_text2 = []
  text = re.sub("'", "",text)
  text=re.sub("(\\d|\\W)+"," ",text) 
  text = text.replace("nbsp", "")
  clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
  clean_text2 = [word for word in clean_text if black_txt(word)]
  return " ".join(clean_text2)

"""#### Cleaning the jobs corpus """

df_all['text'] = df_all['text'].apply(clean_txt)

df_all.head()

"""####TF-IDF 

"""

#initializing tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

tfidf_jobid = tfidf_vectorizer.fit_transform((df_all['text'])) #fitting and transforming the vector
tfidf_jobid

"""# Cretating the User Corpus"""

df_job_view = pd.read_csv("Job_Views.csv")
df_job_view.head(2)

"""In this case we will use only the columns 'Applicant.ID', 'Job.ID', 'Position', 'Company','City'"""

df_job_view = df_job_view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]
df_job_view["select_pos_com_city"] = df_job_view["Position"].map(str) + "  " + df_job_view["Company"] +"  "+ df_job_view["City"]
df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].map(str).apply(clean_txt)
df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].str.lower()
df_job_view = df_job_view[['Applicant.ID','select_pos_com_city']]
df_job_view.head()

"""### Experience Dataset

"""

#Experience
df_experience = pd.read_csv("Experience.csv")
df_experience.head(2)

#taking only Position
df_experience= df_experience[['Applicant.ID','Position.Name']]
#cleaning the text
df_experience['Position.Name'] = df_experience['Position.Name'].map(str).apply(clean_txt)
df_experience.head()

df_experience =  df_experience.sort_values(by='Applicant.ID')
df_experience = df_experience.fillna(" ")
df_experience.head()

"""same applicant has 3 applications 100001 in sigle line so we need to join them"""

#adding same rows to a single row
df_experience = df_experience.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()
df_experience.head(5)

"""### Position of Interest dataset"""

#Position of interest
df_poi =  pd.read_csv("Positions_Of_Interest.csv", sep=',')
df_poi = df_poi.sort_values(by='Applicant.ID')
df_poi.head()

df_poi = df_poi.drop('Updated.At', 1)
df_poi = df_poi.drop('Created.At', 1)

#cleaning the text
df_poi['Position.Of.Interest']=df_poi['Position.Of.Interest'].map(str).apply(clean_txt)
df_poi = df_poi.fillna(" ")
df_poi.head(10)

df_poi = df_poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()
df_poi.head()

"""## Creating the final user dataset by merging all the users datasets

Merging jobs and experience dataframes
"""

df_job_exp = df_job_view.merge(df_experience, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
df_job_exp = df_job_exp.fillna(' ')
df_job_exp = df_job_exp.sort_values(by='Applicant.ID')
df_job_exp.head()

"""Merging position of interest with existing dataframe"""

df_job_exp_poi = df_job_exp.merge(df_poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
df_job_exp_poi = df_job_exp_poi.fillna(' ')
df_job_exp_poi = df_job_exp_poi.sort_values(by='Applicant.ID')
df_job_exp_poi.head()

"""combining all the columns"""

df_job_exp_poi["text"] = df_job_exp_poi["select_pos_com_city"].map(str) + df_job_exp_poi["Position.Name"] +" "+ df_job_exp_poi["Position.Of.Interest"]

df_job_exp_poi.head()

"""Select only "Applicant.ID" and "text" columns:"""

df_final_person= df_job_exp_poi[['Applicant.ID','text']]
df_final_person.head()

df_final_person.columns = ['Applicant_id','text']
df_final_person.head()

df_final_person['text'] = df_final_person['text'].apply(clean_txt)

"""#### Computing cosine similarity using tfidf"""

from sklearn.metrics.pairwise import cosine_similarity
user_tfidf = tfidf_vectorizer.transform(user_q['text'])
cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

output2 = list(cos_similarity_tfidf)

"""###  Function to get the top-N recomendations order by score"""

def get_recommendation(top, df_all, scores):
  recommendation = pd.DataFrame(columns = ['ApplicantID', 'JobID',  'title', 'score'])
  count = 0
  for i in top:
      recommendation.at[count, 'ApplicantID'] = u
      recommendation.at[count, 'JobID'] = df_all['Job.ID'][i]
      recommendation.at[count, 'title'] = df_all['Title'][i]
      recommendation.at[count, 'score'] =  scores[count]
      count += 1
  return recommendation

"""## The top recommendations using TF-IDF"""

top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
list_scores = [output2[i][0][0] for i in top]
get_recommendation(top,df_all, list_scores)

"""### Using Count Vectorizer"""

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()

count_jobid = count_vectorizer.fit_transform((df_all['text'])) #fitting and transforming the vector
count_jobid

from sklearn.metrics.pairwise import cosine_similarity
user_count = count_vectorizer.transform(user_q['text'])
cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)

output2 = list(cos_similarity_countv)

"""## The top recommendations using CountVectorizer"""

top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
list_scores = [output2[i][0][0] for i in top]
get_recommendation(top, df_all, list_scores)

"""##Recomendation  using Spacy"""

import spacy

"""#### Transform the copurs text to the *spacy's documents* """

# Commented out IPython magic to ensure Python compatibility.
# %%time
# list_docs = []
# for i in range(len(df_all)):
#   doc = nlp("u'" + df_all['text'][i] + "'")
#   list_docs.append((doc,i))
# print(len(list_docs))

def calculateSimWithSpaCy(nlp, df, user_text, n=6):
    # Calculate similarity using spaCy
    list_sim =[]
    doc1 = nlp("u'" + user_text + "'")
    for i in df.index:
      try:
            doc2 = list_docs[i][0]
            score = doc1.similarity(doc2)
            list_sim.append((doc1, doc2, list_docs[i][1],score))
      except:
        continue

    return  list_sim

user_q.text[186]

# Commented out IPython magic to ensure Python compatibility.
# %%time
#  df3 = calculateSimWithSpaCy(nlp, df_all, user_q.text[186], n=15)

df_recom_spacy = pd.DataFrame(df3).sort_values([3], ascending=False).head(10)

df_recom_spacy.reset_index(inplace=True)

index_spacy = df_recom_spacy[2]
list_scores = df_recom_spacy[3]

"""## The Top recommendations using Spacy"""

get_recommendation(index_spacy, df_all, list_scores)