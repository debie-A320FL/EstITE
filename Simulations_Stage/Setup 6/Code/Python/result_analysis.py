#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

Nrow = int(1e5)
nb_sim = 80
non_tr_percentage = 2

suffix = f"{nb_sim}_tr_per_{non_tr_percentage}_size_{Nrow}"

basedir_setup_6_res = "/home/onyxia/work/EstITE/Simulations_Stage/Setup 6/Code/Results/"

# --- 1) load the CSVs ----
df_pehe  = pd.read_csv(f'{basedir_setup_6_res}pehe_{suffix}.csv', index_col='sim')
df_times = pd.read_csv(f'{basedir_setup_6_res}times_{suffix}.csv', index_col='sim')

# --- 2) PEHE boxplot ----
plt.figure()
# boxplot of each column (model) side by side
df_pehe.boxplot()
plt.title('PEHE by learner')
plt.ylabel('PEHE')
plt.xlabel('Learner')
# Set y-axis limits to avoid flat plot due to outliers
plt.ylim(bottom=df_pehe.min().min() * 0.9, top=df_pehe.quantile(0.95).max() * 1.1)
plt.tight_layout()
plt.savefig(f'{basedir_setup_6_res}pehe_boxplot_{suffix}.png')
plt.close()

# --- 3) Time boxplot ----
plt.figure()
df_times.boxplot()
plt.title('Execution time by learner')
plt.ylabel('Time (seconds)')
plt.xlabel('Learner')
plt.tight_layout()
plt.savefig(f'{basedir_setup_6_res}time_boxplot_{suffix}.png')
plt.close()

print("Saved: pehe_boxplot.png, time_boxplot.png")

# compute and print quantiles ---
# compute quantiles and both print and export to text 
def compute_and_export_quantiles(df, name, file_handle):
    qs = df.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    header = f"{name} quantiles:\n"
    print(header, end="")
    file_handle.write(header)
    for q in qs.index:
        pct = int(q * 100)
        line = f"  {pct:>2d}th percentile:\n"
        print(line, end="")
        file_handle.write(line)
        for col in df.columns:
            val = qs.at[q, col]
            subline = f"    {col:>2s}: {val:.6e}\n"
            print(subline, end="")
            file_handle.write(subline)
    print()
    file_handle.write("\n")

# open the text file for writing
with open(f'quantiles_{suffix}.txt', 'w') as txt:
    compute_and_export_quantiles(df_pehe,  "PEHE", txt)
    compute_and_export_quantiles(df_times, "Execution time (s)", txt)

print("Quantiles written to quantiles.txt")

