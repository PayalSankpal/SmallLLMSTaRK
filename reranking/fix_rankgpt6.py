import re

with open('reranking/rankgpt_nvidia.py', 'r') as f:
    text = f.read()

text = text.replace("df_train = pd.read_csv('../teacher_cot_evaluations.csv')", "df_train = pd.read_csv('../teacher_cot_evaluations.csv').head(1)")

with open('reranking/rankgpt_nvidia.py', 'w') as f:
    f.write(text)
