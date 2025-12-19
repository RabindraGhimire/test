import pandas as pd

# Load your model responses CSV
df = pd.read_csv("llamaguard3_results_falcon_7b.csv")

# Assign persona IDs (0, 1, 2) based on order of variants per prompt
df["persona_id"] = df.groupby("prompt").cumcount() % 3

# 'llamaguard3_label' already has 1=unsafe, 0=safe
df["success"] = df["llamaguard3_label"]

# Compute ASR per persona (across all prompts)
asr_persona = (
    df.groupby("persona_id")["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "ASR"})
)

print("ASR per persona:")
print(asr_persona)

# Optional: If you have multiple models, compute ASR per persona × model
# Here you can replace 'model_name' with actual column if you have multiple models
# For this CSV, we can treat it as Mistral responses, or add a 'model' column manually
df["model"] = "Mistral"

asr_persona_model = (
    df.groupby(["model", "persona_id"])["success"]
    .mean()
    .reset_index()
    .rename(columns={"success": "ASR"})
)

# Pivot table for clarity
asr_table = asr_persona_model.pivot(index="persona_id", columns="model", values="ASR")
print("\nASR per persona × model:")
print(asr_table)
