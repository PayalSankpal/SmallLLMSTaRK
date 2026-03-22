with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "r") as f:
    text = f.read()
import re
text = re.sub(r'docs_str = .*\n.*?\.join\(docs_info\)', 'docs_str = "\\n".join(docs_info)', text)
text = text.replace('docs_str = "\\n"\n".join(docs_info)', 'docs_str = "\\n".join(docs_info)')
with open("/raid/adityasd314/BTechProject/reranking/rankgpt_nvidia.py", "w") as f:
    f.write(text)
