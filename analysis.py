# -----------------------------------------
# Analyzing Data with Pandas and Visualizing Results with Matplotlib
# -----------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# Download Iris dataset directly from seaborn’s GitHub repo
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Save it locally for next time
df.to_csv("iris.csv", index=False)
print("✅ Iris dataset downloaded and saved as iris.csv")


# Task 1: Load and Explore the Dataset
# -----------------------------------------
try:
    # Load dataset (you can replace "iris.csv" with any dataset of your choice)
    df = pd.read_csv("iris.csv")  
    
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ File not found. Please check the file path.")
except Exception as e:
    print("⚠️ An error occurred:", e)

# Display first few rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# Check data structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean data (drop missing values as example)
df = df.dropna()

# Task 2: Basic Data Analysis
# -----------------------------------------
print("\nBasic Statistics:")
print(df.describe())

# Example grouping: mean petal length per species
if "species" in df.columns:
    group_stats = df.groupby("species")["petal_length"].mean()
    print("\nAverage Petal Length per Species:")
    print(group_stats)

# Task 3: Data Visualization
# -----------------------------------------
plt.style.use("seaborn-v0_8")  # Optional style for nicer visuals

# 1. Line Chart (example: petal length over index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal_length"], label="Petal Length")
plt.title("Line Chart of Petal Length")
plt.xlabel("Index")
plt.ylabel("Petal Length")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length per species)
if "species" in df.columns:
    plt.figure(figsize=(8,5))
    group_stats.plot(kind="bar", color="skyblue")
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Average Petal Length")
    plt.show()

# 3. Histogram (distribution of petal length)
plt.figure(figsize=(8,5))
plt.hist(df["petal_length"], bins=20, color="green", alpha=0.7)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (sepal length vs petal length)
if "sepal_length" in df.columns and "petal_length" in df.columns:
    plt.figure(figsize=(8,5))
    plt.scatter(df["sepal_length"], df["petal_length"], alpha=0.7, c="purple")
    plt.title("Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length")
    plt.ylabel("Petal Length")
    plt.show()

# Findings & Observations
print("\n🔎 Observations:")
print("- The dataset has", len(df), "rows after cleaning.")
print("- Basic statistics show spread and averages of numerical values.")
if "species" in df.columns:
    print("- Different species have distinct average petal lengths, which may help classification.")
