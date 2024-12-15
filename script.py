import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

df = pd.read_csv("apartments.csv")

print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())


df = df.dropna(subset=['price', 'square'])

df[['current_floor', 'total_floors']] = df['floor'].str.split('/', expand=True).astype(float)

df['price_per_square'] = df['price'] / df['square']

le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])

df['price_category'] = pd.cut(df['price'], bins=[0, 200000, 400000, df['price'].max()], 
                              labels=['Low', 'Medium', 'High'])

X = df[['square', 'rooms', 'price_per_square', 'location_encoded', 'current_floor']]
y = df['price_category']

X, y = X.dropna(), y.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    results[name] = accuracy


plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("Accuracy")
plt.title("Classifier Performance Comparison")
plt.show()
