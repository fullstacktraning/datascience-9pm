# used to manipulate tabular data (DataFrame)
import pandas as pd

#TransactionEncoder
#apriori alogorithm, understands only True/False (or) 0 and 1
from mlxtend.preprocessing import TransactionEncoder

# apriori used to find the frequency
# association_rules used to create rules
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Dataset
transactions = [
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Butter']
]

# Step 2: Convert to DataFrame
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

#print("Dataset:\n", df)

# # Step 3: Apply Apriori
frequent_items = apriori(df, min_support=0.6, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_items)

# Step 4: Generate Rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)

print("\nAssociation Rules:\n", rules)