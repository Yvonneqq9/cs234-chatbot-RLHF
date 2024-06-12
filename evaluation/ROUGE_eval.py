import pandas as pd
from rouge import Rouge

# Load the CSV file
file_path = '/Users/dezhiyu/Downloads/generated_responses.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Replace NaN values with empty strings in 'answer_0' and 'answer_1' columns
df['answer_0'] = df['answer_0'].fillna('')
df['answer_1'] = df['answer_1'].fillna('')

# Extract reference answers and candidate answers as lists
candidates = df['answer_0'].tolist()
references = df['answer_1'].tolist()

# Filter out empty candidate answers and their corresponding reference answers
filtered_candidates = []
filtered_references = []

for candidate, reference in zip(candidates, references):
    if candidate.strip() != "" and reference.strip() != "":
        filtered_candidates.append(candidate)
        filtered_references.append(reference)

# Initialize ROUGE evaluator
rouge = Rouge()

# Compute ROUGE scores
scores = rouge.get_scores(filtered_candidates, filtered_references, avg=True)

# Print ROUGE scores
print("ROUGE Scores: ", scores)
