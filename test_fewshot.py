import pandas as pd
import ast

df = pd.read_csv('teacher_cot_evaluations.csv')
few_shot_str = ""
for i, row in df.iterrows():
    r = row['teacher_response']
    if not isinstance(r, str): continue
    try:
        reasoning = r.split('<reasoning>')[1].split('</reasoning>')[0].strip()
        ranking_str = r.split('<ranking>')[1].split('</ranking>')[0].strip()
        ranking = ast.literal_eval(ranking_str)
        # We need the input format too
        # Wait, the input docs_str isn't saved cleanly in the CSV?
        
    except Exception as e:
        print(e)
print("Finished")
