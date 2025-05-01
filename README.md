Import libraries: We import Pandas for data handling, Seaborn and Matplotlib for visualization, and MinMaxScaler for normalization.

Load dataset: The Titanic CSV file is loaded into a DataFrame.

Explore missing values: We print missing values and use a heatmap to visualize them.

Handle missing data: Missing 'Age' values are filled with the median, 'Embarked' with the most frequent value (mode), and the 'Cabin' column is dropped completely.

Drop useless columns: 'PassengerId', 'Name', and 'Ticket' are removed since they don't help in prediction.

Convert categorical to numeric: 'Sex' is mapped to 0 and 1, and 'Embarked' is converted into dummy variables.

Normalize features: 'Age' and 'Fare' are scaled between 0 and 1 using MinMaxScaler.

Save cleaned data: The processed data is saved as a new CSV file.

Visualizations: We create a survival count plot by sex and a histogram of normalized fare to understand data distribution better.

🌟 Why is Data Preprocessing Important?
It cleans messy data.

It prepares features in the right format.

It boosts model accuracy by a huge margin.

It removes biases from scale or irrelevant noise.

In short, good preprocessing = smart machine learning.
