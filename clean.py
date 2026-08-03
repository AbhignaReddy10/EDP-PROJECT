
import pandas as pd
#i created a new file called clean.py to clean the data and save it to a new csv file

#i created a dataframe from the raw csv file using pandas
df = pd.read_csv("employee_sample_raw.csv")

print("=== Raw data preview ===")
print(df.head(), "\n")

print("=== Shape (rows, columns) ===")
print(df.shape, "\n")

print("=== Data types before cleaning ===")
print(df.dtypes, "\n")

print("=== Missing values per column ===")
print(df.isna().sum(), "\n")

print("=== Duplicate rows ===")
print(f"{df.duplicated().sum()} duplicate rows exist\n")

df = df.drop_duplicates().reset_index(drop=True)

df["department"] = df["department"].str.strip().str.title()

df["salary"] = (
    df["salary"]
    .astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .replace("nan", pd.NA)
)
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

df["salary"] = df.groupby("department")["salary"].transform(
    lambda s: s.fillna(s.median())
)
df["age"] = df["age"].fillna(df["age"].median())


df["age"] = df["age"].astype(int)
df["salary"] = df["salary"].round(2)

#i used this to remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

print("=== Cleaned data preview ===")
print(df.head(), "\n")

print("=== Missing values after cleaning ===")
print(df.isna().sum(), "\n")

print("=== Duplicate rows after cleaning ===")
print(f"{df.duplicated().sum()} duplicate rows\n")

print("=== Data types after cleaning ===")
print(df.dtypes, "\n")

label_column = "performance_score"
feature_columns = [c for c in df.columns if c not in [label_column, "employee_name"]]

print(f"Label (what we'd want to predict): {label_column}")
print(f"Features (inputs used to predict it): {feature_columns}\n")

# I Saved the cleaned dataset to a new CSV file
df.to_csv("employee_sample_clean.csv", index=False)
print("Saved cleaned dataset to employee_sample_clean.csv")