import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# # Load your data
data_aisles = pd.read_csv("D:/project-internship/aisles.csv/aisles.csv")
data_departments = pd.read_csv("D:/project-internship/departments.csv/departments.csv")
data_order_products = pd.read_csv("D:/project-internship/order_products__prior.csv/order_products__prior.csv")
data_orders = pd.read_csv("D:/project-internship/orders.csv/orders.csv")
data_products = pd.read_csv("D:/project-internship/products.csv/products.csv")

df_order_products = pd.merge(data_order_products.sample(frac=0.1), data_products, on='product_id', how='left')
df_order_products = pd.merge(df_order_products, data_aisles, on='aisle_id', how='left')
df_order_products = pd.merge(df_order_products, data_orders, on='order_id', how='left')
df_order_products = pd.merge(df_order_products, data_departments, on='department_id', how='left')

# Streamlit app
def main():
    st.title("Instacart Market Basket Analysis EDA")

    option1=['top_reorder_products', 'Top products per department']
    option2=['Department wise reorder ratio with error bar','product sales per depart', 'Department wise reorder ratio', 'Top products per department']
    option3=[ 'Add to cart-reorder ratio', 'Department wise reorder ratio with error bar','Department wise reorder ratio']
    option4=['no.of unique orders per day of week', 'frequency heatmap']
    option5=['frequency dist by days since prior order', 'no.of unique orders per day of week', 'distribution of add to cart order','Add to cart-reorder ratio']
    #sidebar for data selection
    selected_analysis = st.sidebar.selectbox("Select Analysis", ["Top Products", "Department Distribution", "Reorder Ratio", "Busiest Days", "Order Distribution by Hour"])
    #1
    if selected_analysis == "Top Products":
        st.header("Top 10 Products by Frequency")
        plot_top_products_frequency(df_order_products)
        
        selected_option = st.selectbox('Select other EDAs realted to top products:', option1)

        if selected_option == 'top_reorder_products':
            st.header("Top Reordered products with highest reorder ratio")
            plot_top_reorder_products(df_order_products)

        elif selected_option == 'Top products per department':
            st.header("top products per department by aisles")
            plot_department_volume_by_aisles(df_order_products)

    #2
    elif selected_analysis == "Department Distribution":
        st.header("Departmental Distribution")
        plot_department_distribution(df_order_products)

        selected_option = st.selectbox('Select other EDAs realted to departments:', option2)

        if selected_option == 'Department wise reorder ratio with error bar':
            st.header("Department wise reorder ratio with error bar")
            plot_department_reorder_ratio_with_error_bars(df_order_products)

        elif selected_option == 'Top products per department':
            st.header("top products per department by aisles")
            plot_department_volume_by_aisles(df_order_products)

        elif selected_option == 'product sales per depart':
            st.header("Product sales per depart")
            plot_product_sales_by_department(df_order_products)

        elif selected_option == 'Department wise reorder ratio':
            st.header("Department wise reorder ratio")
            plot_department_reorder_ratio(df_order_products)
    #3
    elif selected_analysis == "Reorder Ratio":
        st.header("Department-wise Reorder Ratio")
        plot_department_distribution(df_order_products)

        selected_option = st.selectbox('Select other EDAs realted to Reorders:', option3)

        if selected_option == 'Department wise reorder ratio with error bar':
            st.header("Department wise reorder ratio with error bar")
            plot_department_reorder_ratio_with_error_bars(df_order_products)
        
        elif selected_option == 'Add to cart-reorder ratio':
            st.header("Add to cart-reorder ratio")
            plot_add_to_cart_order_reorder_ratio(df_order_products)

        elif selected_option == 'Department wise reorder ratio':
            st.header("Department wise reorder ratio")
            plot_department_reorder_ratio(df_order_products)
    #4
    elif selected_analysis == "Busiest Days":
        st.header("Busiest Days of The Week")
        plot_busiest_days(df_order_products)

        selected_option = st.selectbox('Select other EDAs realted to Days:', option4)

        if selected_option == 'No.of unique orders per day of week':
            st.header("No.of unique orders per day of week")
            plot_orders_per_day(df_order_products)

        elif selected_option ==  'frequency heatmap':
            st.header("Frequency heatmap")
            plot_orders_frequency_heatmap(df_order_products)
    #5
    elif selected_analysis == "Order Distribution by Hour":
        st.header("Order Distribution Across the Day")
        plot_order_distribution_by_hour(df_order_products)

        selected_option = st.selectbox('Select other EDAs realted to Orders:', option5)

        if selected_option == 'frequency dist by days since prior order':
            st.header('Frequency dist by days since prior order')
            plot_days_since_prior_order_distribution(df_order_products)
        
        elif selected_option == 'Add to cart-reorder ratio':
            st.header("Add to cart-reorder ratio")
            plot_add_to_cart_order_reorder_ratio(df_order_products)

        elif selected_option ==  'frequency heatmap':
            st.header("Frequency heatmap")
            plot_orders_frequency_heatmap(df_order_products)

        elif selected_option == 'Add to cart-reorder ratio':
            st.header("Add to cart-reorder ratio")
            plot_add_to_cart_order_distribution(df_order_products)




def plot_top_products_frequency(df_order_products):
    product_frequency_count = df_order_products['product_name'].value_counts()
    product_frequency_count= product_frequency_count.reset_index()

    top_product_frequency_count = product_frequency_count.head(10)
    top_product_frequency_count.columns = ['product_name', 'frequency_count']

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_product_frequency_count, x='product_name', y='frequency_count', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Product Name')
    ax.set_ylabel('Frequency Count')
    ax.set_title('Top 10 Products by Frequency')

    st.pyplot(fig)

    # Add an option to display raw data
    if st.checkbox("Show Raw Data", False):
        st.write(top_product_frequency_count)

    # Add interactivity - filter by aisle
    selected_aisle = st.selectbox("Select Aisle", df_order_products['aisle'].unique())
    filtered_data = df_order_products[df_order_products['aisle'] == selected_aisle]

    # Display a table with filtered data
    st.dataframe(filtered_data)

def plot_department_distribution(df_order_products):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df_order_products, x='department', order=df_order_products['department'].value_counts().index, palette='viridis', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Department')
    ax.set_ylabel('Count')
    ax.set_title('Departmental Distribution')

    st.pyplot(fig)

    # Add an option to display raw data
    if st.checkbox("Show Raw Data", False):
        st.write(df_order_products[['department', 'aisle', 'product_name']])

    # Add interactivity - filter by department
    selected_department = st.selectbox("Select Department", df_order_products['department'].unique())
    filtered_data = df_order_products[df_order_products['department'] == selected_department]

    # Display a table with filtered data
    st.dataframe(filtered_data)

def plot_department_reorder_ratio(df_order_products):
    grouped_df = df_order_products.groupby(["department"])["reordered"].mean().reset_index()

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 12))  # Create a 2-row subplot

    # First subplot: Department-wise Reorder Ratio
    sns.pointplot(x='department', y='reordered', data=grouped_df, color='b', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Department', fontsize=12)
    ax.set_ylabel('Reorder ratio', fontsize=12)
    ax.set_title('Department-wise Reorder Ratio')

    # Second subplot: Top Products with Highest Reorder Ratio
    top_reorder_products = df_order_products[df_order_products['reordered'] == 1]['product_name'].value_counts().sort_values(ascending=False)[:12]
    sns.barplot(x=top_reorder_products.index, y=top_reorder_products.values, color='powderblue', ax=bx)
    bx.set_xticklabels(bx.get_xticklabels(), rotation=90)
    bx.set_xlabel('Products')
    bx.set_ylabel('Reorder count')
    bx.set_title('Top Products with Highest Reorder Ratio')

    st.pyplot(fig)

    # Add an option to display raw data
    if st.checkbox("Show Raw Data", False):
        st.write(grouped_df)

    # Add interactivity - filter by reorder ratio
    reorder_ratio_slider = st.slider("Select Reorder Ratio", 0.0, 1.0, 0.5)
    filtered_data = grouped_df[grouped_df['reordered'] >= reorder_ratio_slider]

    # Display a table with filtered data
    st.dataframe(filtered_data)

def plot_busiest_days(df_order_products):
    weekday_map = {0:'Sunday', 1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday'}
    busiest_days = df_order_products['order_dow'].map(weekday_map).value_counts().loc[weekday_map.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=busiest_days.index, y=busiest_days.values, ax=ax)
    ax.set_title('Busiest Days of The Week')
    ax.set_ylabel('Number of Orders', fontsize=12)
    ax.set_xlabel('Day of The Week', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')  # Add this line if the weekday labels are overlapping

    st.pyplot(fig)

    # Add an option to display raw data
    if st.checkbox("Show Raw Data", False):
        st.write(busiest_days)

def plot_order_distribution_by_hour(df_order_products):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='order_hour_of_day', data=df_order_products, color='skyblue', ax=ax)
    ax.set_title('Order Distribution Across the Day')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Number of Orders')

    st.pyplot(fig)

    # Add an option to display raw data
    if st.checkbox("Show Raw Data", False):
        st.write(df_order_products[['order_hour_of_day', 'product_name']])

    # Add interactivity - filter by hour
    selected_hour = st.slider("Select Hour", 0, 23, 12)
    filtered_data = df_order_products[df_order_products['order_hour_of_day'] == selected_hour]

    # Display a table with filtered data
    st.dataframe(filtered_data)
    
    total_order_ids = filtered_data['order_id'].sum()
    st.write(f"Total Order IDs for Hour {selected_hour}: {total_order_ids}")

def plot_top_reorder_products(df_order_products):
    top_reorder_products = df_order_products[df_order_products['reordered']==1]['product_name'].value_counts().sort_values(ascending=False)[:12]
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=top_reorder_products.index, y=top_reorder_products.values, color='powderblue', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Products')
    ax.set_ylabel('Reorder Count')
    ax.set_title('Top Products with Highest Reorder Ratio')
    st.pyplot(fig)

def plot_department_reorder_ratio(df_order_products):
    grouped_df = df_order_products.groupby(["department"])["reordered"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.pointplot(x='department', y='reordered', data=grouped_df, color='b', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Department', fontsize=12)
    ax.set_ylabel('Reorder ratio', fontsize=12)
    ax.set_title('Department-wise Reorder Ratio')
    st.pyplot(fig)

def plot_product_sales_by_department(df_order_products):
    fig, ax = plt.subplots(figsize=(12,6))
    df_order_products['department'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Product Sales by Department')
    ax.set_xlabel('Department')
    ax.set_ylabel('Number of Products Sold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

def plot_department_reorder_ratio_with_error_bars(df_order_products):
    grouped_df = df_order_products.groupby(["department"])["reordered"].mean().reset_index()
    grouped_df['reordered_std'] = df_order_products.groupby(["department"])["reordered"].std().reset_index()["reordered"]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='department', y='reordered', data=grouped_df, color='b', ax=ax)
    ax.errorbar(x=grouped_df['department'], y=grouped_df['reordered'], yerr=grouped_df['reordered_std'], fmt='none', color='k')
    ax.set_ylabel('Reorder Ratio', fontsize=12)
    ax.set_xlabel('Department', fontsize=12)
    ax.set_title('Department-wise Reorder Ratio with Error Bars', fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
    st.pyplot(fig)

def plot_add_to_cart_order_reorder_ratio(df_order_products):
    df_order_products["add_to_cart_order_mod"] = df_order_products["add_to_cart_order"].copy()
    df_order_products.loc[df_order_products["add_to_cart_order_mod"] > 70, "add_to_cart_order_mod"] = 70
    grouped_df = df_order_products.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.pointplot(x=grouped_df["add_to_cart_order_mod"].values, y=grouped_df['reordered'].values, color='powderblue', markers='o', linestyles='-', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel('Reorder Ratio')
    ax.set_xlabel('Add to Cart Order')
    ax.set_title('Add to Cart Order - Reorder Ratio')

    st.pyplot(fig)

def plot_orders_per_day(df_order_products):
    weekday_names = {
        0: 'Sunday',
        1: 'Monday',
        2: 'Tuesday',
        3: 'Wednesday',
        4: 'Thursday',
        5: 'Friday',
        6: 'Saturday'
    }

    # Calculating the number of unique orders for each day of the week
    orders_per_day = df_order_products.groupby('order_dow')['order_id'].apply(lambda x: len(x.unique()))

    # Map numerical day of the week codes to week names
    weekdays = [weekday_names[day] for day in orders_per_day.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(weekdays, orders_per_day)
    ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
    ax.set_ylabel('Order Count')
    ax.set_xlabel('Day of Week')
    ax.set_title('Number of Unique Orders by Day of Week')
    st.pyplot(fig)

def plot_orders_frequency_heatmap(df_order_products):
    grp_df = df_order_products.groupby(["order_dow", "order_hour_of_day"])['order_number'].aggregate("count").reset_index()
    grp_df = grp_df.pivot(index="order_dow", columns="order_hour_of_day", values="order_number")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(grp_df, ax=ax)
    ax.set_title("Frequency of the Day of Week VS Hour of the Day")
    st.pyplot(fig)

def plot_days_since_prior_order_distribution(df_order_products):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(x='days_since_prior_order', data=df_order_products, color="purple", ax=ax)
    ax.set_title('Frequency Distribution by Days since Prior Order', fontsize=15)
    ax.set_xlabel('Days since Prior Order')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

def plot_add_to_cart_order_distribution(df_order_products):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_order_products, x='add_to_cart_order', bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Add-to-Cart Order')
    ax.set_xlabel('Add-to-Cart Order')
    ax.set_ylabel('Count')
    st.pyplot(fig)

def plot_department_volume_by_aisles(df_order_products):
    colors = sns.color_palette("Set2")
    unique_departments = df_order_products['department'].unique()
    num_rows = len(unique_departments)

    fig, axes = plt.subplots(num_rows, 1, figsize=(12, num_rows*4))
    for i, department in enumerate(unique_departments):
        ax = axes[i]
        department_df = df_order_products[df_order_products['department'] == department]
        aisle_counts = department_df['aisle'].value_counts().sort_values(ascending=False)
        sns.barplot(x=aisle_counts.index, y=aisle_counts.values, ax=ax, palette=colors)
        ax.set_title(f'Department: {department}')
        ax.set_xlabel('Aisle')
        ax.set_ylabel('Product Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()



