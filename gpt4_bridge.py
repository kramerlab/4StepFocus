import os
from dotenv import load_dotenv
import openai

from experiment import Experiment
from pathfinding import bfs_all_shortest_paths, reduce_num_paths, get_target_neighbors_of_certain_type

# Import environment variables including OpenAi api key
load_dotenv()
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Take question or prompt as first argument, followed by an optional chat log
def ask(question: str, experiment: Experiment, chat_log=None):
    config = experiment.config

    if chat_log is None:
        chat_log = [{
            'role': 'system',  # system: Only used in first message, to give guideline
            'content': experiment.prompts["initial_system_message"]
        }]
    chat_log.append({'role': 'user', 'content': question})  # user: questions or prompts entered by user
    response = client.chat.completions.create(model=experiment.llm_model, messages=chat_log,
                                              temperature=config["llm_temperature"], seed=config["llm_seed"])
    answer = response.choices[0].message.content
    chat_log.append({'role': 'assistant', 'content': answer})  # assistant: answers given by ChatGPT
    experiment.logger.log(f"[Ask ChatGPT Question]: {question}\n[Ask ChatGPT Answer]: {answer}")
    return answer, chat_log


def ask_for_unknown_entity(experiment: Experiment, node_type_dict: {}, chat_log: str):
    question = experiment.prompts["ask_for_unknown_entity"]
    question = question.replace("{node_type_dict}", str(node_type_dict))
    answer, chat_log = ask(question, experiment, chat_log)
    return answer, chat_log


def prepare_informed_query(prompts: dict, node_ids: [int], question: str, type_of_unknown: str, experiment: Experiment,
                       max_depth: int) -> str:
    skb_b = experiment.skb_b
    max_target_visits = experiment.config["max_printed_paths_to_one_target"]
    max_path_to_unknowns = experiment.config["max_num_path_to_unknowns"]

    answers_counter = 1

    # out = prompts["ask_informed_query_head_5_answers"]
    out = prompts["ask_informed_query_head"]
    out = out.replace("{question}", question)

    for node_description in skb_b.nodes2str(node_ids):
       out += str(answers_counter) + ") " + node_description + "\n"
       answers_counter += 1

    out += '\nRelations:\n'
    for i in range(len(node_ids)):
        targets = [node_ids[j] for j in range(i + 1, len(node_ids))]
        if not skb_b.is_directed:
            targets += [node_ids[j] for j in range(0, i)]

        if len(targets) > 0:
            paths = bfs_all_shortest_paths(node_ids[i], targets, skb_b.skb, max_depth=max_depth)
            paths, removed_targets = reduce_num_paths(paths, targets, limit=max_target_visits)
            for path in paths:
                out += str(answers_counter) + ") " + skb_b.path2str(path) + "\n"
                answers_counter += 1
            for target in removed_targets:
                out += (f"{answers_counter}) There are more than {max_target_visits} paths connecting "
                        f"{skb_b.entity_id2name(node_ids[i])} with {skb_b.entity_id2name(target)}.\n")
                answers_counter += 1

    if type_of_unknown is not None:
        paths, truncated = get_target_neighbors_of_certain_type(node_ids, max_path_to_unknowns, type_of_unknown,
                                                                skb_b.skb)

        if len(paths) <= max_path_to_unknowns:
            if truncated:
                out += (f"Found more than {max_path_to_unknowns} paths to any {type_of_unknown}. "
                        f"The following might be most relevant:")
            else:
                out += f"\nPaths to any {type_of_unknown}:\n"
            for path in paths:
                out += f"{answers_counter}) {skb_b.path2str(path)} ({type_of_unknown})\n"
                answers_counter += 1
    return out
