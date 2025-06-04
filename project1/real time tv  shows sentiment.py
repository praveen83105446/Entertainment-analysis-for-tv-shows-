
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('large_real_time_tv_sentiment.csv')
print(df.head())



print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())


# Check for duplicates
print(df.duplicated().sum())
# Remove duplicates
df = df.drop_duplicates()
print(df.duplicated().sum())

df.to_csv("cleaned_large_real_time_tv_sentiment.csv", index=False)
print(df)

# EDA
# Univariate analysis
sns.histplot(df['Sentiment'], kde=True)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Bivariate analysis
sns.scatterplot(x='Timestamp', y='Sentiment', data=df)
plt.title('Sentiment vs Timestamp')
plt.xlabel('Timestamp')
plt.ylabel('Sentiment')
plt.show()

# Multivariate analysi
sns.pairplot(df, vars=['Sentiment', 'Likes', 'Shares'])
plt.title('Pairplot of Sentiment with Viewership and Genre')
plt.show()

# Linear regression model
X = df[['Likes']]
y = df['Shares']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Sentiment')
plt.xlabel('Actual Sentiment')
plt.ylabel('Predicted Sentiment')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()
