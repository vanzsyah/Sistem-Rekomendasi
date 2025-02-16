import re
import nltk
import warnings
import numpy as np
import contractions
import pandas as pd
from nltk.corpus import stopwords
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, sep='|', index_col=0)
    return df

def preprocess_reviews(df):
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda x: contractions.fix(x))
    df['review'] = df['review'].str.replace(r'[;()!?:&\-="]', ' ', regex=True)
    df['review'] = df['review'].str.replace(' +', ' ', regex=True)
    df['review'] = df['review'].str.lower()
    
    grouped_df = df.groupby("item").agg({'rating': list, 'review': lambda x: ' '.join(x)}).reset_index()
    grouped_df['review'] = grouped_df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return grouped_df

# Feature extraction and clustering
def vectorize_reviews(df):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(df['review']).todense()
    df['review_vector'] = vectors.sum(axis=1)
    
    df['total_rating'] = df['rating'].apply(lambda x: sum(x) if isinstance(x, list) else x)
    
    return df[['item', 'total_rating', 'review_vector']]

def perform_clustering(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['total_rating', 'review_vector']])
    
    kmeans = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
    kmeans.fit(scaled_data)
    
    dbi = davies_bouldin_score(scaled_data, kmeans.labels_)
    df['cluster'] = kmeans.labels_
    
    cluster_counts = df['cluster'].value_counts().to_dict()
    
    return df, dbi, cluster_counts

# Recommendation system
def build_recommendation_models_by_cluster(df, df_clustered):
    cluster_eval_results = {}
    
    for cluster in df_clustered['cluster'].unique():
        cluster_items = df_clustered[df_clustered['cluster'] == cluster]['item']
        cluster_data = df[df['item'].isin(cluster_items)]
        
        reader = Reader(rating_scale=(1.0, 10.0))
        data = Dataset.load_from_df(cluster_data[['user', 'item', 'rating']], reader)
        trainset = data.build_full_trainset()

        svd_model = SVD(random_state=42)
        svd_model.fit(trainset)
        
        cv_results = cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        
        cluster_eval_results[cluster] = {
            'model': svd_model,
            'rmse': np.mean(cv_results['test_rmse']),
            'mae': np.mean(cv_results['test_mae'])
        }
        
    return cluster_eval_results

import os
import streamlit as st

def list_files(startpath):
    file_structure = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = "│   " * level + "├── "
        file_structure.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = "│   " * (level + 1) + "├── "
        for f in files:
            file_structure.append(f"{sub_indent}{f}")
    return file_structure



# Main training process
def main():
    df_raw = load_data('../data/clean_data_manual.csv')
    df_processed = preprocess_reviews(df_raw)
    df_vectorized = vectorize_reviews(df_processed)
    df_clustered, dbi_score, cluster_counts = perform_clustering(df_vectorized)

    cluster_eval_results = build_recommendation_models_by_cluster(df_raw, df_clustered)

    df_new = pd.DataFrame.from_dict(cluster_eval_results, orient='index')
    df_new = df_new.drop(columns=['model'])
    print(df_new, '\n')

st.title("📂 Struktur Folder di Streamlit Cloud")

# Tentukan path root dari proyek
root_dir = os.getcwd()
files = list_files(root_dir)

# Tampilkan hasil di Streamlit
st.text("\n".join(files))

"""
    try:
        with open("model_data.pkl", "rb") as f:
            df_clustered, dbi_score, cluster_counts, cluster_eval_results = joblib.load(f)
        print("Model berhasil disimpan sebagai model_data.pkl")
    except Exception as e:
        print("Model gagal disimpan sebagai model_data.pkl")

""" 

if __name__ == "__main__":
    main()
