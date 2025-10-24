import json
import pandas as pd

# Load results
with open('results/benchmark_results.json') as f:
    results = json.load(f)

# Create DataFrame
df = pd.DataFrame(results)

# Select columns
df = df[['description', 'index_size_mb', 'latency_mean', 'latency_p95', 
         'latency_p99', 'throughput_qps']]

# Round numbers
df = df.round(2)

# Save as CSV
df.to_csv('results/comparison_table.csv', index=False)

# Print markdown table
print(df.to_markdown(index=False))