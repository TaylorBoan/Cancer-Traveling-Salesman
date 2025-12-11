import pandas as pd

df = pd.read_csv("utah_ldct_access.csv")

# Basic sanity checks
print(df.head())
print("\n")
print(df.describe()[["population", "travel_time_to_ldct_min", "need_score"]])
print("\n")
# How many ZCTAs are missing access (no path)?
print("NaN travel time:", df["travel_time_to_ldct_min"].isna().sum())
print("\n")
# Top 20 by need_score
top_need = df.sort_values("need_score", ascending=False).head(20)
print(top_need[["zcta", "population", "travel_time_to_ldct_min", "need_score"]])
