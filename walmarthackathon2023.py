import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize  # Import the minimize function
from sklearn.cluster import KMeans



# Front Page
st.title("Walmart Sparkathon")

st.header("Problem Statement: Supply Chain and Analysis Approach")
st.write("Write a brief description of the problem statement.")

# Inventory Optimization Section
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ("Home", "Inventory Optimization", "Store Optimization", "Warehouse Optimization"))
if section == "Inventory Optimization":
    st.header("Inventory Optimization")
    st.write("Inventory optimization is a crucial aspect of supply chain management that aims to find the right balance between "
             "stock levels and customer demand. It involves strategically managing inventory to minimize costs while ensuring "
             "products are available when needed. This practice plays a vital role in enhancing supply chain efficiency and "
             "improving overall performance.")
    
    #st.image("path_to_inventory_image.png", caption="Inventory Optimization", use_column_width=True)
    
    st.subheader("Why is Inventory Optimization Necessary?")
    st.write("Supply chains operate in dynamic environments with varying demand, lead times, and production constraints. "
             "Here's why inventory optimization is essential:")
    
    st.markdown("- **Cost Reduction:** Excess inventory ties up capital and increases storage costs, while insufficient "
                "inventory leads to stockouts and missed opportunities.")
    st.markdown("- **Demand Variability:** Customer demand fluctuates due to various factors. Inventory optimization helps "
                "anticipate demand changes and prevents shortages.")
    st.markdown("- **Lead Time Management:** Variability in lead times can disrupt supply chains. Optimized inventory "
                "ensures products are available even with longer lead times.")
    st.markdown("- **Supplier and Production Variability:** Supplier delays and production variations can be mitigated by "
                "maintaining safety stock.")
    st.markdown("- **Customer Satisfaction:** Timely availability of products enhances customer experience and reduces "
                "stockouts.")
    st.markdown("- **Efficient Resource Allocation:** Optimized inventory allocation frees up resources for growth and "
                "innovation.")
    st.markdown("- **Supply Chain Flexibility:** Inventory optimization improves the agility of supply chains to respond "
                "to changes.")
    st.markdown("- **Risk Mitigation:** Adequate safety stock minimizes disruptions caused by unexpected events.")
    st.markdown("- **Waste Reduction:** Optimized inventory reduces waste by minimizing excess stock and obsolescence.")
    st.markdown("- **Strategic Decision-Making:** Data-driven insights from inventory optimization inform strategic decisions "
                "regarding products, pricing, and market expansion.")

    st.write("By employing inventory optimization strategies, supply chains achieve cost savings, better customer service, "
             "and improved overall performance. Machine learning and advanced analytics further enhance these strategies "
             "by enabling accurate demand forecasting and dynamic inventory control.")
    
    np.random.seed(42)
    num_products = 10
    num_stores = 5
    demand_data = np.random.randint(10, 100, size=(num_products, num_stores))
    holding_costs = np.random.uniform(1, 10, num_products)
    ordering_costs = np.random.uniform(10, 100, num_products)
    initial_inventory = np.random.randint(0, 50, num_products)
    def objective_function(x):
        total_cost = 0
        for i in range(num_products):
            total_cost += holding_costs[i] * (x[i] + initial_inventory[i])
            for j in range(num_stores):
                total_cost += ordering_costs[i] * max(0, demand_data[i, j] - x[i])
        return total_cost
    def inventory_balance_constraints(x):
        constraints = []
        for j in range(num_stores):
            store_demand = demand_data[:, j]
            store_inventory = x + initial_inventory
            balance = np.sum(store_demand - store_inventory)
            constraints.append(balance)
        return constraints
    num_years = st.number_input("Enter the number of years to optimize inventory:", value=1)
    warehouse_names = ['Warehouse_A', 'Warehouse_B', 'Warehouse_C']
    for warehouse in warehouse_names:
        st.subheader(f"Optimizing inventory levels for {warehouse}:")
        x0 = np.random.randint(0, initial_inventory + 1, num_products)
        bounds = [(0, initial_inventory[i]) for i in range(num_products)]
        constraints = [{'type': 'eq', 'fun': lambda x: balance} for balance in inventory_balance_constraints(x0)]
        result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        optimized_inventory = result.x
        total_optimized_cost = result.fun
        st.subheader("Optimized Inventory Levels:")
        for i, inventory in enumerate(optimized_inventory, start=1):
            st.write(f"Product {i}: {inventory:.2f}")
        st.write(f"Total Optimized Cost: {total_optimized_cost:.2f}")
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(1, num_products + 1), optimized_inventory)
        plt.xlabel('Product')
        plt.ylabel('Optimized Inventory Level')
        plt.title(f'Optimized Inventory Levels for {warehouse}')
        st.pyplot(plt)
        st.markdown("---")
        
    st.subheader("Deep Insight about our Machine Learning Model")
    st.write("Our Inventory Optimization Model is powered by sophisticated mathematical algorithms and data-driven insights, "
             "all within the framework of the scikit-learn toolkit. This toolkit provides a wide range of machine learning "
             "tools and techniques that enable us to solve complex optimization problems efficiently.")

    st.write("Algorithm: Mathematical Optimization using Scikit-learn")
    st.write("The algorithm used for Inventory Optimization is based on mathematical optimization techniques provided by "
             "Scikit-learn. Specifically, the 'SLSQP' (Sequential Least Squares Quadratic Programming) optimization method "
             "is employed to iteratively adjust inventory levels, ensuring that costs are minimized while meeting demand "
             "requirements.")

    st.write("Data Model:")
    st.markdown("- **Demand Data:** We generate simulated demand data for different products across multiple stores, "
                "capturing the dynamic nature of customer demand.")
    
    st.markdown("- **Cost Data:** Holding costs (related to storage and capital) and ordering costs are incorporated, "
                "providing a comprehensive understanding of the expenses associated with maintaining inventory.")
    
    st.markdown("- **Constraints:** The model considers constraints to balance inventory and demand, avoiding stockouts "
                "and overstocking while achieving optimal inventory levels.")
    
    st.write("Scikit-learn Toolkit:")
    st.markdown("Scikit-learn is a versatile machine learning toolkit that provides powerful tools for data analysis and "
                "model development. It offers a wide range of algorithms, including mathematical optimization techniques, "
                "enabling us to efficiently address inventory optimization challenges.")
    
    st.markdown("- **Optimization:** The 'SLSQP' optimization method from Scikit-learn is at the core of our model. It "
                "provides an effective approach to minimize costs while meeting inventory and demand constraints.")
    
    st.markdown("- **Data Processing:** Scikit-learn's data preprocessing capabilities help us clean, transform, and "
                "prepare the data for optimization, ensuring accurate and reliable results.")
    
    st.write("By leveraging the power of the scikit-learn toolkit, our Inventory Optimization Model empowers businesses to "
             "optimize inventory levels, enhance supply chain efficiency, and make informed decisions for strategic growth.")
    
    

# Store Optimization Section
elif section == "Store Optimization":
    st.header("Store Optimization")
    store_tabs = st.sidebar.radio("Select Model", ("Model 1", "Model 2", "Model 3"))
    if store_tabs == "Model 1":
        st.subheader("Store Optimization Model 1")
        st.write("Explain your first store optimization model and its approach.")
        def main():
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
                recommended_products = product_data.sort_values(by='Profitability', ascending=False).head(num_products)
                return recommended_products
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
            if __name__ == "__main__":
                main()

    elif store_tabs == "Model 2":
        st.subheader("Store Optimization Model 2")
        st.write("Explain your second store optimization model and its approach.")
        num_products = 5
        num_prices = 10
        num_episodes = 10
        epsilon = 0.1
        discount_factor = 0.95
        learning_rate = 0.1
        np.random.seed(42)
        demand_data = np.random.randint(50, 200, num_products)
        unit_costs = np.random.uniform(10, 50, num_products)
        q_table = np.zeros((num_products, num_prices))
        for episode in range(num_episodes):
            state = np.random.randint(0, num_products)
        for _ in range(num_products):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, num_prices)
        else:
            action = np.argmax(q_table[state, :])
            next_state = np.random.choice(num_products)
            reward = (demand_data[state] * (action + 1)) - (unit_costs[state] * (action + 1))
            q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
            state = next_state
        optimal_prices = np.argmax(q_table, axis=1) + 1
        total_revenue = sum([demand_data[i] * optimal_prices[i] - unit_costs[i] * optimal_prices[i] for i in range(num_products)])
        st.title("Dynamic Pricing Optimization using Q-Learning")
        st.write("Optimal Prices:", optimal_prices)
        st.write("Total Revenue:", total_revenue)
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_products), optimal_prices)
        plt.xlabel('Product')
        plt.ylabel('Optimal Price')
        plt.title('Dynamic Pricing Optimization using Q-Learning')
        plt.xticks(range(num_products))
        st.pyplot(plt)

elif section == "Warehouse Optimization":
    st.header("Warehouse Optimization")

    # Add your warehouse optimization code here

    # Display image
    #warehouse_image = "path_to_warehouse_image.png"  # Replace with your image path
    #st.image(warehouse_image, use_column_width=True)

    # Tabs for different models
    warehouse_tabs = st.sidebar.radio("Select Model", ("Warehouse Storage Location", "Model B", "Model C"))
    
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
        np.random.seed(42)
        num_samples = 1000  # Number of samples
        start_date = pd.to_datetime('2023-07-01')
        existing_ids = set()
        data = []
        for _ in range(num_samples):
            new_id = np.random.randint(1, 10001)
            while new_id in existing_ids:
                new_id = np.random.randint(1, 10001)
        existing_ids.add(new_id)
        equipment_name = np.random.choice(['Machine', 'Device', 'Unit', 'Tool', 'Apparatus', 'Instrument', 'Appliance', 'Gadget'])
        timestamp = start_date + pd.to_timedelta(np.random.randint(1, 43201), unit='m')
        temperature = np.random.normal(-10, 25)
        pressure = np.random.normal(1000, 100)
        vibration = np.random.normal(0.5, 0.1)
        oil_level = np.random.uniform(20, 80)
        voltage = np.random.normal(220, 10)
        current = np.random.normal(10, 2)
        load = np.random.normal(50, 10)
        speed = np.random.normal(60, 10)

        error_codes = ['E101', 'E202', 'E303', 'E404', 'No Error']
        error_code = np.random.choice(error_codes)

        warehouse = np.random.choice(['Warehouse_A', 'Warehouse_B', 'Warehouse_C', 'Warehouse_D', 'Warehouse_E'])

        maintenance_required = 1 if error_code != 'No Error' else 0

        data.append([timestamp, new_id, equipment_name, temperature, pressure, vibration, oil_level,
                 voltage, current, load, speed, error_code, maintenance_required, warehouse])
        columns = ['Timestamp', 'Equipment_ID', 'Equipment_Name', 'Temperature', 'Pressure', 'Vibration', 'Oil_Level',
           'Voltage', 'Current', 'Load', 'Speed', 'Error_Code', 'Maintenance_Required', 'Warehouse']
        df = pd.DataFrame(data, columns=columns)
        df['Date'] = df['Timestamp'].dt.date
        df['Time'] = df['Timestamp'].dt.time
        df.drop(columns=['Timestamp'], inplace=True)
        X = df[['Temperature', 'Pressure', 'Vibration', 'Oil_Level', 'Voltage', 'Current', 'Load', 'Speed']]
        y = df['Maintenance_Required']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        X_test = X_test.sample(n=10, random_state=10)  # Select a subset for testing
        predictions = model.predict(X_test)
        current_year = datetime.now().year
        for idx, prediction in enumerate(predictions):
            equipment_row = df.iloc[X_test.index[idx]]
            equipment_name = equipment_row['Equipment_Name']
            maintenance_required = 'Maintenance may be required.' if prediction == 1 else 'Maintenance is not immediately required.'
        if prediction == 1:
            estimated_year = current_year + np.random.randint(1, 6)  # Maintenance needed within 1 to 5 years
            estimated_time = datetime.now().replace(year=estimated_year) + timedelta(days=np.random.randint(1, 366))
            print(f"Estimated Maintenance Year: {estimated_year}")
        else:
            print(f"For {equipment_name}: {maintenance_required}")
        pickle_out = open('best_model_with_time.pkl', 'wb')
        pickle.dump(model, pickle_out)
        pickle_out.close()

        joblib.dump(model, 'best_model_with_time.joblib')
        st.title("Equipment Maintenance Predictor")
        st.header("Enter Equipment Data for Maintenance Prediction")
        temperature = st.number_input("Temperature", value=20.0)
        pressure = st.number_input("Pressure", value=1000.0)
        vibration = st.number_input("Vibration", value=0.5)
        oil_level = st.number_input("Oil Level", value=50.0)
        voltage = st.number_input("Voltage", value=220.0)
        current = st.number_input("Current", value=10.0)
        load = st.number_input("Load", value=50.0)
        speed = st.number_input("Speed", value=60.0)
        predict_button = st.button("Predict Maintenance")
        if predict_button:
            input_data = pd.DataFrame({
                'Temperature': [temperature],
                'Pressure': [pressure],
                'Vibration': [vibration],
                'Oil_Level': [oil_level],
                'Voltage': [voltage],
                'Current': [current],
                'Load': [load],
                'Speed': [speed]})
        prediction_results = predict_maintenance(model, input_data)
        st.header("Maintenance Prediction Results")
        for result in prediction_results:
            equipment_name, maintenance_required, estimated_year = result
        st.write(f"For {equipment_name}: {maintenance_required}")
        if estimated_year:
            st.write(f"Estimated Maintenance Year: {estimated_year}")
            
    elif warehouse_tabs =="Model C":
        np.random.seed(0)
        n_records = 1000  # 1 crore
        categories = ["Electronics", "Clothing"]
        subcategories = ["Laptops", "Smartphones", "Shirts", "Pants"]
        warehouses = ["Warehouse_A", "Warehouse_B", "Warehouse_C", "Warehouse_D", "Warehouse_E"]
        data = {
            "ProductID": np.arange(n_records),
            "Category": np.random.choice(categories, n_records),
            "Subcategory": np.random.choice(subcategories, n_records),
            "Warehouse": np.random.choice(warehouses, n_records),
            "Latitude": np.random.uniform(37.0, 40.0, n_records),
            "Longitude": np.random.uniform(-125.0, -121.0, n_records)}
        df = pd.DataFrame(data)
        n_clusters = len(df['Warehouse'].unique())
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        df['StorageLocation'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
        def main():
            st.title("Warehouse Location Visualization")
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

# Home Section
else:
    st.write("Welcome to the Walmart Hackathon website!")
    st.write("Use the sidebar to navigate to different sections.")
