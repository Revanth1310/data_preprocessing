# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
df = pd.read_csv('titanic.csv')  # Ensure the file exists in the same folder

# Step 2: Understand the data
print("Initial Data:")
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Visualize missing data as a heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Step 3: Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Step 4: Drop columns not useful for modeling
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Step 5: Convert categorical to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 6: Normalize numerical features
scaler = MinMaxScaler()
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 7: Save the cleaned data to a new file
df.to_csv('titanic_cleaned.csv', index=False)

# Visualization: Survival count by sex
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Sex")
plt.xlabel("Sex (0 = Male, 1 = Female)")
plt.ylabel("Count")
plt.legend(title='Survived')
plt.show()

# Visualization: Fare distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title("Normalized Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# Sample preview
print("\nCleaned Data Sample:")
print(df.head())
print("\nCleaned data saved to 'titanic_cleaned.csv'")
