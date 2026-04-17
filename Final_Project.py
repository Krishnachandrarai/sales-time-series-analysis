import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
# Load dataset
df = pd.read_excel("C:/Users/kr360/Downloads/Online Retail.xlsx")

# Preview data
print(df.head())
print(df.info())



# REMOVE MISSING VALUES
# ==============================
df = df.dropna(subset=['CustomerID'])

#REMOVE DUPLICATES
# ==============================
df = df.drop_duplicates()

#FIX DATA TYPES
# ==============================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(int)
#print(df.info())


# Remove negative values
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# CREATING NEW FEATURES
# ==============================
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour



#HANDLE OUTLIERS (IMPORTANT)
# ==============================

# Remove extreme quantity outliers
q_low = df['Quantity'].quantile(0.01)
q_high = df['Quantity'].quantile(0.99)
df = df[(df['Quantity'] >= q_low) & (df['Quantity'] <= q_high)]

# Remove extreme price outliers
p_low = df['UnitPrice'].quantile(0.01)
p_high = df['UnitPrice'].quantile(0.99)
df = df[(df['UnitPrice'] >= p_low) & (df['UnitPrice'] <= p_high)]

# No outlier found

#SORT DATA (VERY IMPORTANT)
# ==============================
df = df.sort_values(by='InvoiceDate')

#FINAL CHECK
# ==============================
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print(df.head())

daily_sales = df.groupby('InvoiceDate')['TotalPrice'].sum() 
monthly_sales = df.groupby(['Year', 'Month'])['TotalPrice'].sum()

print(df.info())

#Time Series Graph
plt.figure(figsize=(12,5))
plt.plot(daily_sales)
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()


#Moving Average
rolling = daily_sales.rolling(window=7).mean()
plt.plot(daily_sales, label="Original")
plt.plot(rolling, label="7-Day Average")
plt.legend()
plt.show()

#Monthly Sale bar Chart
monthly_sales.plot(kind='bar')
plt.title("Monthly Sales")
plt.show()

#Top product graph
top_products = df.groupby('Description')['TotalPrice'].sum() \
                 .sort_values(ascending=False).head(10)

plt.figure(figsize=(8,8))
plt.pie(top_products, labels=top_products.index, autopct='%1.1f%%')
plt.title("Top 10 Products by Sales Share")
plt.show()

#Corelation heatmap
corr = df[['Quantity','UnitPrice','TotalPrice']].corr()

plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

#Scatter plot

slope, intercept, r, p, std_err = stats.linregress(df['Quantity'], df['TotalPrice'])

line = slope * df['Quantity'] + intercept

plt.scatter(df['Quantity'], df['TotalPrice'])
plt.plot(df['Quantity'], line)
plt.title("Regression Line")
plt.show()

#Normal Distribution
# Log transform
log_data = np.log1p(df['TotalPrice'])

# Fit distribution
mean = log_data.mean()
std = log_data.std()

x = np.linspace(log_data.min(), log_data.max(), 100)
y = norm.pdf(x, mean, std)

plt.hist(log_data, bins=50, density=True)
plt.plot(x, y)
plt.title("Normal Distribution (Log Transformed)")
plt.show()