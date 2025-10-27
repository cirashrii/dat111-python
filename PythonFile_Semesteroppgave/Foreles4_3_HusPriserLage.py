import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate data
samples = 100
sizes = np.random.randint(5, 10, size=samples) * np.random.randint(4, 25, size=samples)
standards =  np.round(np.random.uniform(1, 4, size=samples),1)

# Calculate house prices
prices = [
    np.round(0.002 * (size**0.7) * np.sqrt(standard) * np.random.randint(95,105), 1)
    for size, standard in zip(sizes, standards)
]

# Create DataFrame
df = pd.DataFrame({
    'm2': sizes.astype(str),
    'Std': standards.astype(str),
    'Price': np.round(prices, 2).astype(str)
})

df['m2'] = df['m2'].str.rjust(10)
df['Std'] = df['Std'].str.rjust(10)
df['Price'] = df['Price'].str.rjust(10)
#
# Save to CSV
df.to_csv('house_data.csv', index=False)
print("CSV file 'house_data.csv' created successfully.")



