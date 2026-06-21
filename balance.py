import numpy as np
import pandas as pd
import random

#load 
label_df = pd.read_csv("./labels.csv")

print("\nBalancing dataset with augmentation...")

real_samples = label_df[label_df["label"] == 0].copy()
fake_samples = label_df[label_df["label"] == 1].copy()

print(f"Before balancing — Real: {len(real_samples)}, Fake: {len(fake_samples)}")

difference = len(fake_samples) - len(real_samples)

augmented_rows = []

for i in range(difference):
    row = real_samples.sample(n=1, random_state=i).iloc[0]
    spec = np.load(row["spec_path"])
    
    aug_type = i % 3

    if aug_type == 0:
        noise = np.random.normal(0, 0.5, spec.shape)
        spec = spec + noise

    elif aug_type == 1:
        shift = random.randint(-5, 5)
        spec = np.roll(spec, shift, axis=1)

    elif aug_type == 2:
        shift = random.randint(-2, 2)
        spec = np.roll(spec, shift, axis=0)

    new_file_id = row["file_id"] + f"_aug{i}"
    new_path = row["spec_path"].replace(row["file_id"], new_file_id)
    np.save(new_path, spec)

    augmented_rows.append({
        "file_id":   new_file_id,
        "spec_path": new_path,
        "label":     0
    })

    if i % 500 == 0:
        print(f"Augmenting: {i}/{difference} done...")

augmented_df = pd.DataFrame(augmented_rows)
balanced_df = pd.concat([real_samples, fake_samples, augmented_df], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"After balancing — Real: {len(balanced_df[balanced_df['label']==0])}, Fake: {len(balanced_df[balanced_df['label']==1])}")
print(f"Total samples: {len(balanced_df)}")

balanced_df.to_csv("./labels_balanced.csv", index=False)
print("Saved: ./labels_balanced.csv")