with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'docs_str = "\n".join(docs_info)' in line or 'docs_str =' in line and '.join(docs_info)' in line:
        continue
    new_lines.append(line)

for i, line in enumerate(new_lines):
    if "docs_info.append(" in line:
        new_lines.insert(i+1, "    docs_str = '\\n'.join(docs_info)\n")
        break

with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "w") as f:
    f.writelines(new_lines)
