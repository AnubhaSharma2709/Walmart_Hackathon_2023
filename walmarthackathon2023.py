import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    store_tabs = st.sidebar.radio("Select Model", ("Model 1", "Model 2", "Model 3"))
    if store_tabs == "Model 1":
        st.subheader("Store Optimization Model 1")
        st.write("Explain your first store optimization model and its approach.")
        def generate_random_profitability_data(num_locations=100):
            np.random.seed(42)
            locations = [f'Location_{i+1}' for i in range(num_locations)]
            data = {
                'Location': np.random.randint(10, 100, num_locations),
                'Population': np.random.randint(10000, 500000, num_locations),
                'CompetitionStrength': np.random.uniform(0.1, 0.9, num_locations),
                'IncomeLevel': np.random.randint(20000, 80000, num_locations),
                'RentCost': np.random.randint(5000, 20000, num_locations)}
            profitability_data = pd.DataFrame(data)
            profitability_data['Profitability'] = 1000 + 5 * profitability_data['Population'] + \
                                         1000 * profitability_data['CompetitionStrength'] + \
                                         10 * profitability_data['IncomeLevel'] - \
                                         2 * profitability_data['RentCost'] + \
                                         np.random.normal(0, 5000, num_locations)
            profitability_data.to_csv('profitability_data.csv', index=False)
                def train_profitability_model():
                    profitability_data = pd.read_csv('profitability_data.csv')
                    X = profitability_data.drop(['Location', 'Profitability'], axis=1)
                    y = profitability_data['Profitability']
                    model = LinearRegression()
                    model.fit(X, y)
                    joblib.dump(model, 'profitability_model.pkl')
                def predict_profitability(location_data):
                    loaded_model = joblib.load('profitability_model.pkl')
                    X = location_data.drop(['Location'], axis=1)
                    predicted_profitability = loaded_model.predict(X)[0]
                    return predicted_profitability
                def simulate_profit_over_5_years(initial_profit, growth_rate=0.05):
                    years = np.arange(1, 6)
                    profits = initial_profit * (1 + growth_rate) ** years
                    return years, profits
                def predict_warehouse_and_inventory(location_data):
                    nearest_warehouse = np.random.choice(['Warehouse_A', 'Warehouse_B', 'Warehouse_C'])
                    distance_to_warehouse = np.random.uniform(1, 10)
                    recommended_inventory = location_data['Population'] * 0.02 + distance_to_warehouse * 50
                    return nearest_warehouse, distance_to_warehouse, recommended_inventory.item()
                def recommend_products_for_inventory(location_data, num_products=5):
                    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
                    demand = np.random.randint(100, 1000, len(products))
                    unit_cost = np.random.randint(10, 100, len(products))
                    selling_price = np.random.randint(50, 200, len(products))
                    profitability = (selling_price - unit_cost) * demand
                    product_data = pd.DataFrame({
                        'Product': products,
                        'Demand': demand,
                        'UnitCost': unit_cost,
                        'SellingPrice': selling_price,
                        'Profitability': profitability})
                    recommended_products = product_data.sort_values(by='Profitability', ascending=False).head(num_products):
                    return recommended_products
                    def main():
                        st.title("Warehouse Location and Inventory Recommendation")
                        generate_random_profitability_data()
                        train_profitability_model()
                        selected_location = st.text_input("Enter the Location", "Sample_Location")
                        population = st.number_input("Enter the Population", 10000, 500000, 300000)
                        competition_strength = st.number_input("Enter the Competition Strength", 0.1, 0.9, 0.6)
                        income_level = st.number_input("Enter the Income Level", 20000, 80000, 50000)
                        rent_cost = st.number_input("Enter the Rent Cost", 5000, 20000, 15000)
                        location_data = pd.DataFrame({
                            'Location': [selected_location],
                            'Population': [population],
                            'CompetitionStrength': [competition_strength],
                            'IncomeLevel': [income_level],
                            'RentCost': [rent_cost]})
                        predicted_profitability = predict_profitability(location_data)
                        st.write(f"Predicted Profitability for {selected_location}: {predicted_profitability:.2f}")
                        initial_profit = predicted_profitability
                        years, profits = simulate_profit_over_5_years(initial_profit)
                        st.plotly_chart(plt.plot(years, profits, marker='o'))
                        nearest_warehouse, distance_to_warehouse, recommended_inventory = predict_warehouse_and_inventory(location_data)
                        st.write(f"Nearest Warehouse: {nearest_warehouse}")
                        st.write(f"Distance to Warehouse: {distance_to_warehouse:.2f} miles")
                        st.write(f"Recommended Inventory: {recommended_inventory:.0f}")
                        recommended_products = recommend_products_for_inventory(location_data)
                        st.write("Recommended Products for Inventory:")
                        st.write(recommended_products)
                        plt.figure(figsize=(10, 6))
                        plt.bar(recommended_products['Product'], recommended_products['Profitability'])
                        plt.xlabel('Product')
                        plt.ylabel('Profitability')
                        plt.title('Recommended Products for Inventory')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(plt)

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
    
    if warehouse_tabs == "Warehouse Storage Location":
        from sklearn.cluster import KMeans
        np.random.seed(42)
        num_products = 100
        num_warehouses = 5
        product_ids = np.arange(num_products)
        demand_patterns = np.random.randint(1, 11, size=(num_products, num_warehouses))  # Generate demand patterns for each product and warehouse
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(demand_patterns)
        products = pd.DataFrame({'ProductID': product_ids,'ClusterLabel': cluster_labels})
        warehouses = pd.DataFrame({'WarehouseID': np.arange(num_warehouses),'Latitude': np.random.uniform(37.0, 40.0, num_warehouses),'Longitude': np.random.uniform(-125.0, -121.0, num_warehouses)})
        product_warehouse = pd.merge(products, warehouses, how='cross')
        st.title("Warehouse Layout Optimization")
        selected_warehouse = st.selectbox("Select a Warehouse:", warehouses['WarehouseID'])
        selected_data = product_warehouse[product_warehouse['WarehouseID'] == selected_warehouse]
        st.write(f"Warehouse Layout Map for Warehouse {selected_warehouse}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(warehouses['Longitude'], warehouses['Latitude'], c='red', marker='s', label='Warehouses')
        for warehouse in warehouses.itertuples():
            ax.annotate(warehouse.WarehouseID, (warehouse.Longitude, warehouse.Latitude), textcoords="offset points", xytext=(0, 10), ha='center')
        ax.scatter(selected_data['Longitude'], selected_data['Latitude'], c=selected_data['ClusterLabel'], cmap='viridis', marker='o', label='Products')
        for product in selected_data.itertuples():
            ax.annotate(product.ProductID, (product.Longitude, product.Latitude), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Optimized Warehouse Layout for Warehouse {selected_warehouse}')
        ax.legend()
        st.pyplot(fig)

    elif warehouse_tabs == "Model B":
        st.subheader("Warehouse Optimization Model B")
        st.write("Explain your second warehouse optimization model and its approach.")

    # Add more content as needed

# Home Section
else:
    st.write("Welcome to the Walmart Hackathon website!")
    st.write("Use the sidebar to navigate to different sections.")

# Display any additional content or images as needed
