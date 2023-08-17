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
    st.header("Store Optimization:")
    st.write("Store optimization refers to the strategic management and enhancement of retail store performance to achieve the highest level of operational efficiency, customer satisfaction, and profitability. This comprehensive approach involves a meticulous analysis of diverse factors, including store layout, inventory management, staffing, pricing strategies, and more. The objective is to ensure optimal allocation of resources, aligning with business goals. By undertaking store optimization initiatives, retailers aim to not only enrich the overall shopping experience but also to maximize revenue while minimizing costs.")
    st.header("Why Store Optimization is Essential:")
    st.write("Store optimization holds substantial significance for several compelling reasons:")
    st.write("1. **Efficiency**: The optimization of store operations results in streamlined processes, waste reduction, and the efficient allocation of resources. This contributes to heightened operational efficiency.")
    st.write("2. **Customer Experience**: A well-optimized store encompasses aspects like an inviting layout, well-managed inventory, and exceptional customer service. These factors collectively create a positive shopping experience, fostering customer loyalty and repeat business.")
    st.write("3. **Profitability**: Through the fine-tuning of inventory levels, strategic pricing approaches, and effective product placement, stores can experience increased sales and ultimately enhanced profitability.")
    st.write("4. **Data-Informed Decisions**: Store optimization is rooted in data analysis, empowering retailers to make informed decisions that adapt to shifting market trends and evolving customer preferences.")
    st.write("5. **Competitive Edge**: Well-optimized stores have the potential to outshine competitors by offering superior customer service and an aesthetically pleasing shopping environment.")
    store_tabs = st.sidebar.radio("Select Model", ("Location Prediction", "PricingOptimization"))
    if store_tabs == "Location Prediction":
        st.title("Store Optimization and Profitability Prediction")
        st.header("Machine Learning Model")
        image1 = "output_1.png"
        st.image(image1, caption='Local Image', use_column_width=True)
        image2 = "predict in 5 years.png"
        st.image(image2, caption='Local Image', use_column_width=True)
        image3 = "output_2.png"
        st.image(image3, caption='Local Image', use_column_width=True)
        image4 = "predict_image.png"
        st.image(image4, caption='Local Image', use_column_width=True)
        
        st.header("Neural Network for Profitability Prediction:")
        st.write("Neural Networks, a complex and powerful class of machine learning algorithms, are harnessed for predicting numerical values based on input features. In the context of store optimization, a Neural Network can be adeptly employed to forecast the profitability of diverse store locations, leveraging a multitude of intricate factors.")
        st.subheader("The Mechanism:")
        st.write("1. **Input Features**: Neural Networks process features such as Population, Competition Strength, Income Level, and Rent Cost of a given location.")
        st.write("2. **Training**: The Neural Network is educated using historical data, encompassing both the input features and actual profitability outcomes.")
        st.write("3. **Prediction**: Once trained, the Neural Network has the capability to predict the profitability of new locations by evaluating their input features.")
        st.write("4. **Interpretation**: Neural Networks reveal insights into the influence of each input feature on profitability through their complex interconnected layers.")
        st.header("Conclusion:")
        st.write("Utilizing a Neural Network for store optimization facilitates data-driven decisions regarding store modifications, openings, or closures, thereby fostering improved profitability and overall store performance.")
        st.write("It's important to note that while Neural Networks offer remarkable predictive capabilities, their implementation demands comprehensive data preprocessing, model tuning, and potentially more computational resources compared to simpler algorithms. The choice of algorithm should be tailored to the complexity and objectives of the specific store optimization task.")

    elif store_tabs == "PricingOptimization":
        st.title("Dynamic Pricing Optimization using Q-Learning")
        st.header("Dynamic Pricing:")
        st.write("Dynamic pricing is a strategy where prices of products or services are adjusted in real-time based on various factors, such as demand, competition, and market conditions. It allows businesses to maximize revenue by setting the optimal price at any given time, balancing customer demand and profit margins.")
        st.header("Q-Learning:")
        st.write("Q-learning is a model-free reinforcement learning algorithm that aims to find the optimal action-selection policy for an agent in a given environment. It learns to make decisions by updating a Q-table, which stores the expected cumulative rewards for taking specific actions in certain states.")
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
        st.write("Optimal Prices:", optimal_prices)
        st.write("Total Revenue:", total_revenue)
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_products), optimal_prices)
        plt.xlabel('Product')
        plt.ylabel('Optimal Price')
        plt.title('Dynamic Pricing Optimization using Q-Learning')
        plt.xticks(range(num_products))
        st.pyplot(plt)
        st.header("Code Explanation:")
        st.write("The provided Streamlit code illustrates the implementation of dynamic pricing optimization using Q-learning. Here's a step-by-step breakdown:")
        st.subheader("Initialization:")
        st.write("The code initializes various parameters, such as the number of products (num_products), number of possible prices (num_prices), number of episodes (num_episodes), exploration rate (epsilon), discount factor (discount_factor), and learning rate (learning_rate).")
        st.subheader("Random Data Generation:")
        st.write("Random data is generated for product demand (demand_data), and unit costs (unit_costs).")
        st.subheader("Q-Table Initialization:")
        st.write("A Q-table is initialized with zeros to store expected cumulative rewards for actions in different states.")
        st.subheader("Q-Learning Loop:")
        st.write("The core of the code is the Q-learning loop (for episode in range(num_episodes)). It simulates episodes where the agent (retailer) interacts with the environment (market) to learn optimal pricing strategies.")
        st.write("- The agent starts in a random state (state), which represents a product.")
        st.write("- For each product, the agent decides whether to explore (randomly choose an action) or exploit (choose the action with the highest expected reward from the Q-table).")
        st.write("- The agent selects an action (price) and transitions to a new state (next_state) based on the chosen action.")
        st.write("- The agent receives a reward based on the chosen action, demand for the product, and unit costs.")
        st.write("- The Q-table is updated using the Q-learning equation, incorporating the reward, discount factor, and potential future rewards.")
        st.subheader("Optimal Prices and Total Revenue:")
        st.write("After learning, the algorithm determines the optimal prices for each product (optimal_prices) by selecting the action (price) that yields the highest expected reward for each state (product). The total revenue (total_revenue) is calculated based on the optimal prices and product demand.")

elif section == "Warehouse Optimization":
    st.title("Warehouse Layout Optimization")
    st.header("Why Warehouse Optimization is Required:")
    st.write("Warehouse optimization is essential for several reasons:")
    st.write("1. **Efficiency**: An optimized warehouse layout minimizes the time and effort required to locate and retrieve products, resulting in improved operational efficiency and reduced labor costs.")
    st.write("2. **Inventory Management**: Efficient warehouse layout ensures proper organization of inventory, reducing the risk of stockouts and overstocking, which can lead to higher holding costs and lost sales opportunities.")
    st.write("3. **Order Fulfillment**: An optimized layout enables faster and more accurate order picking, packing, and shipping, leading to improved customer satisfaction and retention.")
    st.write("4. **Space Utilization**: Proper warehouse organization maximizes the use of available space, reducing storage costs and enabling the accommodation of increased inventory without the need for additional warehouse space.")
    st.write("5. **Cost Savings**: Efficient layouts minimize unnecessary movement of goods and personnel, reducing operational costs and improving resource allocation.")


    warehouse_tabs = st.sidebar.radio("Select Model", ("Warehouse", "Maintenance Prediction", "OptmizingWarehouseStorage"))
    
    if warehouse_tabs == "Warehouse":
        st.header("Warehouse Layout Optimization:")
        st.write("Warehouse optimization involves strategically arranging products within a warehouse to enhance operational efficiency and minimize costs. This process ensures that products are stored, picked, and shipped in the most efficient manner, thereby improving overall supply chain performance. Effective warehouse layout optimization can lead to reduced labor costs, faster order fulfillment, and optimized space utilization.")

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
        st.header("Model Explanation - Warehouse Layout Optimization:")
        st.write("The provided code demonstrates warehouse layout optimization using KMeans clustering and visualization techniques. Here's an explanation of the steps involved:")
        st.subheader("Step 1: Data Generation")
        st.write("Random demand patterns are generated for a given number of products and warehouses. Each demand pattern represents the demand of a product across different warehouses.")
        st.subheader("Step 2: KMeans Clustering")
        st.write("KMeans clustering is applied to group products with similar demand patterns into clusters. This helps in identifying products that share similar storage and handling requirements.")
        st.subheader("Step 3: Warehouse and Product Data Preparation")
        st.write("Warehouse and product data are created, including warehouse locations (latitude and longitude) and product-cluster assignments.")
        st.subheader("User Input")
        st.write("The user selects a warehouse for which the optimized layout will be displayed.")
        st.subheader("Step 4: Visualization")
        st.write("The code generates a scatter plot with warehouses represented by red squares (labeled with WarehouseIDs) and products represented by colored circles (labeled with ProductIDs). Products are color-coded based on their assigned clusters.")
        st.header("Algorithm and Machine Learning Model:")
        st.write("The algorithm used in this model is KMeans clustering. KMeans is an unsupervised machine learning algorithm that partitions data into K clusters based on similarity. In this context, KMeans is applied to group products with similar demand patterns, allowing for effective organization and layout optimization within a warehouse.")
        st.write("In summary, warehouse layout optimization using KMeans clustering assists in strategically organizing products within a warehouse to improve operational efficiency, reduce costs, and enhance overall supply chain performance. The visualization provides a clear understanding of how products and warehouses are arranged, facilitating better decision-making for warehouse management.")


    elif warehouse_tabs == "Maintenance Prediction":
        st.header("Predictive Maintenance Importance:")
        st.write("Predictive maintenance is a proactive approach that uses data analysis and machine learning algorithms to predict when equipment or machinery is likely to fail. By identifying potential issues before they lead to costly breakdowns, predictive maintenance helps organizations optimize maintenance schedules, reduce downtime, and enhance overall operational efficiency. Here's why predictive maintenance is crucial and how it contributes to various aspects:")
        st.subheader("Importance of Predictive Maintenance:")
        st.markdown("1. **Cost Savings**: Predictive maintenance minimizes unscheduled downtime, which can lead to substantial financial losses due to halted operations, decreased productivity, and emergency repairs.")
        st.markdown("2. **Increased Equipment Lifespan**: By addressing issues before they escalate, predictive maintenance helps extend the lifespan of equipment, reducing the need for frequent replacements.")
        st.markdown("3. **Efficient Resource Allocation**: Maintenance efforts are targeted based on actual equipment condition, optimizing the allocation of resources such as labor, parts, and maintenance schedules.")
        st.markdown("4. **Improved Safety**: Anticipating equipment failures helps mitigate safety risks associated with sudden breakdowns, protecting both personnel and assets.")
        st.markdown("5. **Data-Driven Insights**: Predictive maintenance generates valuable data insights that can be used to fine-tune maintenance strategies, improve equipment design, and optimize operational processes.")
        st.markdown("6. **Enhanced Customer Satisfaction**: Reliable equipment and uninterrupted service contribute to better customer satisfaction and loyalty.")
        st.write("Machine Learning Model: Random Forest Classifier")
        st.write("The machine learning model used in this app is a Random Forest Classifier. "
         "The Random Forest algorithm is an ensemble learning technique that constructs multiple decision trees "
         "during training and combines their predictions to make a final decision. It is suitable for classification "
         "tasks, such as predicting whether maintenance is needed for equipment based on sensor readings.")
        


    elif warehouse_tabs =="OptmizingWarehouseStorage":
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
        if __name__ == "__main__":
                main()


# Home Section
else:
    st.write("Welcome to the Walmart Hackathon website!")
    st.write("Use the sidebar to navigate to different sections.")
