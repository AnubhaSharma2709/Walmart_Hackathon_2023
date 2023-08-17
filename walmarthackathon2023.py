# -*- coding: utf-8 -*-
"""walmarthackathon2023.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12lIsvnbT-OJdFCVR8-drETCB9cBg32CC
"""

import streamlit as st

# Front Page
st.title("Walmart Hackathon")
st.header("Problem Statement: Supply Chain and Analysis Approach")
st.write("Write a brief description of the problem statement.")

# Inventory Optimization Section
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ("Home", "Inventory Optimization", "Store Optimization", "Warehouse Optimization"))

if section == "Inventory Optimization":
    st.header("Inventory Optimization")

    # Add your inventory optimization code here

    # Display image
    #inventory_image = "path_to_inventory_image.png"  # Replace with your image path
    #st.image(inventory_image, use_column_width=True)

    # Explain the model and results
    st.subheader("Inventory Optimization Model")
    st.write("Explain your inventory optimization model and its approach.")
    st.write("Accuracy of the Model: XX%")  # Replace with your accuracy

    # Add more content as needed

# Store Optimization Section
elif section == "Store Optimization":
    st.header("Store Optimization")

    # Add your store optimization code here

    # Display image
    #store_image = "path_to_store_image.png"  # Replace with your image path
    #st.image(store_image, use_column_width=True)

    # Tabs for different models
    store_tabs = st.sidebar.radio("Select Model", ("Model 1", "Model 2", "Model 3"))

    if store_tabs == "Model 1":
        st.subheader("Store Optimization Model 1")
        st.write("Explain your first store optimization model and its approach.")

    elif store_tabs == "Model 2":
        st.subheader("Store Optimization Model 2")
        st.write("Explain your second store optimization model and its approach.")

    elif store_tabs == "Model 3":
        st.subheader("Store Optimization Model 3")
        st.write("Explain your third store optimization model and its approach.")

    # Add more content as needed

# Warehouse Optimization Section
elif section == "Warehouse Optimization":
    st.header("Warehouse Optimization")

    # Add your warehouse optimization code here

    # Display image
    #warehouse_image = "path_to_warehouse_image.png"  # Replace with your image path
    #st.image(warehouse_image, use_column_width=True)

    # Tabs for different models
    warehouse_tabs = st.sidebar.radio("Select Model", ("Warehouse Storage Location", "Model B"))
    warehouse_tabs = st.sidebar.radio("Select Model", ("Warehouse Storage Location", "Model B"))
    
    if warehouse_tabs == "Warehouse Storage Location":
        st.subheader("Warehouse Storage Location")
        st.write("Explain your first warehouse optimization model and its approach.")
        selected_warehouse = st.selectbox("Select a Warehouse:", warehouses)
        selected_category = st.selectbox("Select a Category:", categories)
        selected_subcategory = st.selectbox("Select a Subcategory:", subcategories)
        selected_data = df[
                (df['Warehouse'] == selected_warehouse) &
                (df['Category'] == selected_category) &
                (df['Subcategory'] == selected_subcategory)]
        if not selected_data.empty:
            st.write(f"Storage Location: **{selected_data['StorageLocation'].values[0]}**")
            plt.figure(figsize=(10, 6))
            plt.scatter(selected_data['Latitude'], selected_data['Longitude'], c=selected_data['StorageLocation'], cmap='viridis', marker='o')
            plt.title(f'Scatter Plot of Warehouse: {selected_warehouse}, Category: {selected_category}, Subcategory: {selected_subcategory}')
            plt.xlabel('Latitude')
            plt.ylabel('Longitude')
            plt.colorbar(label='Storage Location')
            for index, row in selected_data.iterrows():
                plt.text(row['Latitude'], row['Longitude'], f"Product ID: {row['ProductID']}", fontsize=8, ha='left', va='bottom', color='black')
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.write("No data available for the selected criteria.")


elif warehouse_tabs == "Model B":
    st.subheader("Warehouse Optimization Model B")
    st.write("Explain your second warehouse optimization model and its approach.")

    # Add more content as needed

# Home Section
else:
    st.write("Welcome to the Walmart Hackathon website!")
    st.write("Use the sidebar to navigate to different sections.")

# Display any additional content or images as needed
