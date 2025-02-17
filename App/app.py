import os
import joblib
import streamlit as st
import pandas as pd

# Load saved models and data
model_path = os.path.join(os.path.dirname(__file__), "src", "model_data.pkl")

if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            df_clustered, dbi_score, cluster_counts, cluster_eval_results = joblib.load(f)
        st.success("Model data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model data: {e}")
        df_clustered = pd.DataFrame()  # Empty DataFrame to prevent further crashes
else:
    st.error(f"File 'model_data.pkl' tidak ditemukan di path: {model_path}")
    df_clustered = pd.DataFrame()

# Generate recommendations
def generate_recommendations(model, user_id, items_to_predict):
    try:
        recommendations = [(item, model.predict(user_id, item).est) for item in items_to_predict]
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:2]
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []

# Streamlit app
def main():
    st.title("Product Recommendation System")

    if not df_clustered.empty:
        product_list = df_clustered['item'].tolist()
        selected_product = st.selectbox("Select a Product", product_list)

        if selected_product:
            product_cluster = df_clustered[df_clustered['item'] == selected_product]['cluster'].values[0]
            st.write(f"This product belongs to Cluster: {product_cluster}")

            cluster_items = df_clustered[df_clustered['cluster'] == product_cluster]['item'].tolist()
            user_id = 'temp_user'

            items_to_predict = [item for item in cluster_items if item != selected_product]

            if product_cluster in cluster_eval_results:
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
    else:
        st.error("Data model tidak tersedia. Pastikan file 'model_data.pkl' telah diunggah.")

if __name__ == "__main__":
    main()
