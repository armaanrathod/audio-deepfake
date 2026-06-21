import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("./dataset_ready/labels_balanced.csv")


row = df.iloc[1]

spec_path = "./dataset_ready/" + row["spec_path"].replace("./", "")


spec = np.load(spec_path)


plt.figure(figsize=(12, 5))

plt.imshow(
    spec,
    aspect='auto',
    origin='lower',
    cmap='magma'
)

plt.title(
    f"Spectrogram\nFile: {row['file_id']} | Label: {'REAL' if row['label']==0 else 'FAKE'}",
    fontsize=14
)

plt.xlabel("Time Frames")
plt.ylabel("Mel Frequency Bands")

plt.colorbar(label="Decibels (dB)")

plt.tight_layout()

plt.savefig("spectrogram_visualisation.png", dpi=150)

plt.show()

print("Saved as spectrogram_visualisation.png")