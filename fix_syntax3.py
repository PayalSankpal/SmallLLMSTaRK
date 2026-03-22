with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    if 'docs_str = "' in lines[i] and 'join' not in lines[i]:
        lines[i] = "    docs_str = '\\n'.join(docs_info)\n"
    if '".join(docs_info)' in lines[i]:
        lines[i] = ""

with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "w") as f:
    f.writelines(lines)
