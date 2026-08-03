Skills Learned This Week

1. Pandas basics

Loading data with pd.read_csv()
Inspecting data with .head(), .shape, .dtypes, .isna().sum()
Selecting and transforming columns
Grouping data with .groupby() to fill missing values more intelligently (e.g., filling missing salary with the median for that department, not the overall median)

2. Data cleaning techniques

Detecting and removing duplicate rows with .drop_duplicates()
Standardizing inconsistent text ("Sales" vs "sales" vs "ENGINEERING") with .str.strip() and .str.title()
Cleaning messy numeric strings (e.g. "$45,000" → 45000.0) using .str.replace() with regex + pd.to_numeric(errors="coerce")
Fixing data types (e.g., converting age to int after filling missing values)
Re-checking for duplicates after cleaning, since standardizing values can surface new duplicates that weren't obvious in the raw text

3. Core ML vocabulary: Features vs. Labels

Label (target): the column we would want a model to predict — here, performance_score
Features (inputs): the other columns used to predict the label — department, age, salary, years_experience
Understanding this split is the foundation for Week 2 (training a regression model)
