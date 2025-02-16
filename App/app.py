import os
import joblib
import streamlit as st
import pandas as pd

# Load saved models and data

try:
    model_path = os.path.join("src", "model_data.pkl")
    
    with open(model_path, "rb") as f:
        df_clustered, dbi_score, cluster_counts, cluster_eval_results = joblib.load(f)
    
    print("File loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")

# Generate recommendations
def generate_recommendations(model, user_id, items_to_predict):
    recommendations = [(item, model.predict(user_id, item).est) for item in items_to_predict]
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:2]

# Streamlit app
def main():
    st.title("Product Recommendation System")

    product_list = df_clustered['item'].tolist()
    selected_product = st.selectbox("Select a Product", product_list)

    if selected_product:
        product_cluster = df_clustered[df_clustered['item'] == selected_product]['cluster'].values[0]
        st.write(f"This product belongs to Cluster: {product_cluster}")

        cluster_items = df_clustered[df_clustered['cluster'] == product_cluster]['item'].tolist()
        user_id = 'temp_user'

        items_to_predict = [item for item in cluster_items if item != selected_product]
        recommendations = generate_recommendations(cluster_eval_results[product_cluster]['model'], user_id, items_to_predict)

        st.write("### Recommended Products:")
        for rec in recommendations:
            st.write(f"Product: {rec[0]} | Predicted Rating: {rec[1]:.2f}")

        if st.button("View All Products in This Cluster"):
            st.write(cluster_items)

    # Evaluation metrics section
    st.sidebar.title("Model Evaluation")
    st.sidebar.write(f"**Clustering Davies-Bouldin Index:** {dbi_score:.4f}")

    st.sidebar.write(f"**Recommendation System Evaluation by Cluster:**")
    for cluster, metrics in cluster_eval_results.items():
        st.sidebar.write(f"Cluster {cluster} : {cluster_counts[cluster]} items, \nRMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    main()
