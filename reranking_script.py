import argparse
import ast
import functools
import pandas as pd
import torch
import openai
from tqdm import tqdm

from stark_qa import load_skb
from custom_pipeline.llm_bridge import LlmBridge


def method0_reranking(top_k_node_ids, query, node_id_mask, kb, llm,
                      sim_weight, max_k, compact_docs, add_rel):
    cand_len = len(top_k_node_ids)
    pred_dict = {}

    prompts = []
    for idx, node_id in enumerate(top_k_node_ids):
        doc_info = kb.get_doc_info(node_id, add_rel=add_rel, compact=compact_docs)
        node_type = kb.get_node_type_by_id(node_id)

        prompts.append(
            f'Examine if a {node_type} '
            f'satisfies a given query and assign a score from 0.0 to 1.0. '
            f'If the {node_type} does not satisfy the query, the score should be 0.0. '
            f'If there exists explicit and strong evidence supporting that {node_type} '
            f'satisfies the query, the score should be 1.0. If partial evidence or weak '
            f'evidence exists, the score should be between 0.0 and 1.0.\n'
            f'Here is the query:\n\"{query}\"\n'
            f'Here is the information about the {node_type}:\n' +
            doc_info + '\n\n' +
            f'Please score the {node_type} based on how well it satisfies the query. '
            f'ONLY output the floating point score WITHOUT anything else. '
            f'Output: The numeric score of this {node_type} is: '
        )

    answers, _ = llm.ask_llm_batch(prompts, chat_logs=None)
    for idx, node_id in enumerate(top_k_node_ids):
        try:
            llm_score = float(answers[idx])
        except TypeError:
            if answers[idx] is None:
                if add_rel:
                    raise RuntimeError()
                else:
                    llm_score = 0.5
        except ValueError:
            llm_score = 0.5
        sim_score = (cand_len - idx) / cand_len
        score = llm_score + sim_weight * sim_score

        if node_id_mask is not None:
            score /= 2
            if idx < len(node_id_mask):
                score += 0.5
        pred_dict[node_id] = score

    node_scores = torch.FloatTensor(list(pred_dict.values()))
    top_k_idx = torch.topk(
        node_scores,
        min(max_k, len(node_scores)),
        dim=-1,
        largest=True,
        sorted=True
    ).indices.tolist()
    top_k_node_ids = [list(pred_dict.keys())[i] for i in top_k_idx]
    return top_k_node_ids


def method1_reranking(top_k_node_ids, query, node_id_mask, kb, llm, compact_docs):
    def method1_for_list_of_nodes(node_ids_to_rerank, query):
        if len(node_ids_to_rerank) == 0:
            return []

        possible_answers = "\n"
        for node_id in node_ids_to_rerank:
            doc_info = kb.get_doc_info(node_id, add_rel=False, compact=compact_docs)
            possible_answers += str(node_id) + " " + doc_info + "\n"

        prompt = (
            f'The rows of the following list consist of an ID number, a type and a corresponding descriptive text:\n'
            f'{possible_answers} \n'
            f'Please sort this list in descending order according to how well the elements can be considered as '
            f'answers to the following query: \n'
            f'{query} \n'
            f'Please make absolutely sure that the element which satisfies the query best is the first element in your order. '
            f'Return ONLY the corresponding ID numbers separated by commas in the asked order.'
        )

        output, _ = llm.ask_llm_batch([prompt], chat_logs=None)

        try:
            answer = [int(node_id_str.strip()) for node_id_str in output[0].split(",")]
            answer = list(dict.fromkeys(answer))  # Remove duplicate Node_ids
            sorted_IDs = [node_id for node_id in answer if node_id in node_ids_to_rerank]  # remove invented IDs
            invented_ids = len(answer) - len(sorted_IDs)
            print("LLM has invented: ", invented_ids, " node IDs in it's answer.")
            missing_ids = len(node_ids_to_rerank) - len(sorted_IDs)
            print("LLM output does not contain ", missing_ids, " IDs from the input.")
        except Exception:
            sorted_IDs = []
            print("LLM output contains elements that cannot be cast to integer.")
            print("Erroneous LLM output: ", output[0])

        sorted_IDs += [node_id for node_id in node_ids_to_rerank if node_id not in sorted_IDs]
        return sorted_IDs

    to_rerank = top_k_node_ids
    answer = method1_for_list_of_nodes(to_rerank, query)
    return answer


def pairwise_comparison(node1_id, node2_id, query, kb, llm, compact_docs, add_rel):
    if node1_id == node2_id:
        return 0

    node_type_1 = kb.get_node_type_by_id(node1_id)
    node_type_2 = kb.get_node_type_by_id(node2_id)

    doc_info_1 = kb.get_doc_info(node1_id, add_rel=add_rel, compact=compact_docs)
    doc_info_2 = kb.get_doc_info(node2_id, add_rel=add_rel, compact=compact_docs)

    prompt = (
        f'The following two elements consist of an ID number, a type and a corresponding descriptive text:\n \n'
        f'{node1_id}, {node_type_1}, {doc_info_1}. \n'
        f'{node2_id}, {node_type_2}, {doc_info_2}. \n\n'
        f'Find out which of the elements satisfies the following query better: \n'
        f'{query} \n'
        f'Return ONLY the corresponding ID number which corresponds to the element that satisfies '
        f'the given query best. Nothing else.'
    )

    answer, _ = llm.ask_llm_batch([prompt], chat_logs=None)
    answer = answer[0]
    if isinstance(answer, str):
        answer = answer.replace("'", "").replace('"', "").strip()
    if answer == "A":
        answer = node1_id
    elif answer == "B":
        answer = node2_id

    try:
        answer = int(answer)
    except Exception:
        print("LLM output cannot be cast to int.")
        print("Erroneous LLM output: ", answer)
        return 0
    if answer == node1_id:
        return 1
    elif answer == node2_id:
        return -1
    else:
        print("LLM output is neither of the given node IDs")
        print("Erroneous LLM output: ", answer)
        return 0


def method2_reranking(top_k_node_ids, query, node_id_mask, kb, llm, compact_docs, max_k):
    to_rerank = top_k_node_ids
    try:
        answer = sorted(
            to_rerank,
            key=functools.cmp_to_key(
                lambda node1_id, node2_id: pairwise_comparison(
                    node1_id, node2_id, query=query, kb=kb, llm=llm,
                    compact_docs=compact_docs, add_rel=True
                )
            ),
            reverse=True
        )
    except RuntimeError:
        answer = sorted(
            to_rerank,
            key=functools.cmp_to_key(
                lambda node1_id, node2_id: pairwise_comparison(
                    node1_id, node2_id, query=query, kb=kb, llm=llm,
                    compact_docs=compact_docs, add_rel=False
                )
            ),
            reverse=True
        )
    return answer


def rerank(top_k_node_ids, query, reranking_method, node_id_mask,
           kb, llm, sim_weight=0.1, max_k=20, compact_docs=False, add_rel=False):

    top_k_node_ids = top_k_node_ids[:max_k]

    if reranking_method == 0:
        try:
            sorted_node_ids = method0_reranking(
                top_k_node_ids, query, node_id_mask, kb, llm,
                sim_weight, max_k, compact_docs, add_rel=True
            )
        except RuntimeError:
            sorted_node_ids = method0_reranking(
                top_k_node_ids, query, node_id_mask, kb, llm,
                sim_weight, max_k, compact_docs, add_rel=False
            )
        except openai.BadRequestError:
            sorted_node_ids = method0_reranking(
                top_k_node_ids, query, node_id_mask, kb, llm,
                sim_weight, max_k, compact_docs, add_rel=False
            )
    elif reranking_method == 1:
        sorted_node_ids = method1_reranking(
            top_k_node_ids, query, node_id_mask, kb, llm, compact_docs
        )
    elif reranking_method == 2:
        sorted_node_ids = method2_reranking(
            top_k_node_ids, query, node_id_mask, kb, llm, compact_docs, max_k
        )
    else:
        raise NotImplementedError("Reranking_method_not_specified!")

    return sorted_node_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank SKB answers with LLM.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="prime",
        help="Dataset name for load_skb (e.g., 'prime', 'amazon', 'mag')."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV (full_data_dump.csv)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV (reranked_full_data_dump.csv)."
    )
    parser.add_argument(
        "--method",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Reranking method: 0=score-based, 1=list sort, 2=pairwise."
    )
    parser.add_argument(
        "--metrics_on",
        type=int,
        default=0,
        choices=[0, 1],
        help="Reranking method: 0=answer_list, 1=vss_merged"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta/llama-3.3-70b-instruct",
        help="LLM model name for LlmBridge."
    )
    parser.add_argument(
        "--configs_path",
        type=str,
        default="configs.json",
        help="Path to LlmBridge configs."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load CSV
    df = pd.read_csv(args.input_csv)

    # Parse results column
    df["results"] = df["results"].map(lambda x: ast.literal_eval(x))

    if(args.metrics_on == 1):
        df["vss_merged_candidates"] = df["vss_merged_candidates"].map(lambda x: ast.literal_eval(x))
        df["answer_list"] = df["vss_merged_candidates"]
    else:
        answers = []
        for x in df["results"]:
            answers.append((x['answer_list']))
        df["answer_list"] = answers

    df["answer_list"] = df["answer_list"].map(lambda x: x[:20])

    # Load SKB
    kb = load_skb(args.dataset_name, download_processed=True)
    print("Loaded SKB nodes.")

    # Init LLM
    llm = LlmBridge(
        model_name=args.model_name,
        configs_path=args.configs_path,
        verbose=False,
    )

    sim_weight = 0.1
    max_k = 20
    compact_docs = False
    add_rel = True

    reranked_answers = []
    for i in tqdm(range(len(df)), desc="Reranking answers", unit="query"):
        reranked_answers.append(
            rerank(
                df["answer_list"][i],
                df["query"][i],
                reranking_method=args.method,
                node_id_mask=None,
                kb=kb,
                llm=llm,
                sim_weight=sim_weight,
                max_k=max_k,
                compact_docs=compact_docs,
                add_rel=add_rel,
            )
        )

    df_new = pd.DataFrame()
    df_new["id"] = df["id"] 
    df_new["ground_truths"] = df["ground_truths"]
    df_new["query"] = df["query"]
    df_new["original_answers"] = df["answer_list"]
    df_new["reranked_answers"] = reranked_answers
    
    df_new.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
