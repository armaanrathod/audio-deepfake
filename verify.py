import pandas as pd
import numpy as np

df = pd.read_csv("./labels_balanced.csv")

print(f"Total samples: {len(df)}")
print(f"Real: {len(df[df['label']==0])}")
print(f"Fake: {len(df[df['label']==1])}")

# Check ALL files have the same shape
print("\nChecking all files have consistent shape...")
shapes = set()

for idx, row in df.iterrows():
    try:
        spec = np.load(row["spec_path"])
        shapes.add(spec.shape)
    except:
        print(f"Missing file: {row['spec_path']}")

    if idx % 5000 == 0:
        print(f"Checked {idx}/{len(df)}...")

print(f"\nShapes found: {shapes}")

if len(shapes) == 1:
    print(f"ALL FILES CONSISTENT. Shape: {shapes}")
    print("Your dataset is ready for handoff.")
else:
    print("WARNING: inconsistent shapes found — tell me what shapes appear")