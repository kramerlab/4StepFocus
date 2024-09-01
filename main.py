from experiment import Experiment
from gpt4_bridge import prepare_informed_query, ask
from utils import prepare_entities_search_list
from stark_main.models.vss import VSS


def main(question_ids, dataset_name, experiment_name, data_split):
    experiment = Experiment(experiment_name, dataset_name, data_split)
    # default: llm_model = gpt-4o-2024-05-13, alternative: llm_model='gpt-4-1106-preview')

    for question_id in question_ids:
        query = experiment.get_query(question_id)
        answer_node_ids = eval_on_stark_query2(experiment, question_id, query)

        # answer, chat_log_w_skb = ask(
        #     "Are you sure? Quantify your certainty with a number in the range from 0.00 to 1.00.", chat_log_w_skb)

    print('the end')


def eval_on_stark_query2(experiment: Experiment, query_id: int, query: str) -> list[str]:
    """
    :param experiment:
    :param query_id:
    :param query:
    :return:
    list of answers in descending order
    """
    prompts = experiment.prompts
    skb = experiment.skb_b.skb

    experiment.logger.log(experiment.qa_pair2str(query_id))

    llm_query = f"{prompts['asking_for_entities_head']}{query} ---\n\n{prompts['asking_for_entities_tail']}"
    llm_query = llm_query.replace("{node_type_list}", str(skb.node_type_lst()))
    llm_query = llm_query.replace("{edge_type_list}", str([x for x in skb.edge_type_dict.values()]))

    # answer, chat_log = ask(llm_query, experiment)
    # mag
    # q_id: 0
    answer = "paper |?1| -> paper___has_topic___field_of_study -> field_of_study 'holograms', paper |?1| -> paper___has_topic___field_of_study -> field_of_study 'monte carlo', paper |?1| -> paper___has_topic___field_of_study -> field_of_study 'quantum dimensions', paper |?1| -> paper___has_topic___field_of_study -> field_of_study '3-partite'"
    chat_log = [{'role': 'system', 'content': 'You are a helpful assistant. You give short and precise answers. If you are asked for a topic, subject, or object, you only answer with the name of the topic, subject, or object. You use no more words than needed. Do not use dots or quotation marks in your answers. Your answers need to be in a machine-readable format. If you are asked for a number, you only respond with a float number. If you are asked if something exists, do not answer yes, do answer with the best example. If you are given background information by a user, favor answers derived from the background information over your own immanent answer. '}, {'role': 'user', 'content': "The following statement or question between the three dashes is given: --- I am interested in holograms, monte carlo, quantum dimensions and 3-partite -- what is a paper I should read that publishes on this topic? ---\n\nI want to lookup the the answer in a paper authorship knowledge base. The knowledge base has the entities: ['author', 'institution', 'field_of_study', 'paper'], and it has the relations: ['author___affiliated_with___institution', 'paper___cites___paper', 'paper___has_topic___field_of_study', 'author___writes___paper']. Return a sequence of triplets (entity1_type entity1_name -> relation -> entity_2_type entity_2_name), separated by commas, describing which entities I should lookup in the knowledge base to answer the question between the three dashes. Use |x1|, |x2|, |x3|, etc. for names unknown entities. Put the type of an entity in front of its name. Here is an example of such a sequence of triplets: author 'Hemmingway' -> author___writes___paper -> paper |x1|, paper |x1| -> paper___has_topic___field_of_study -> field_of_study 'how mice learn to run'. Answer without description or explanation."}, {'role': 'assistant', 'content': "paper |?1| -> paper___has_topic___field_of_study -> field_of_study 'holograms', paper |?1| -> paper___has_topic___field_of_study -> field_of_study 'monte carlo', paper |?1| -> paper___has_topic___field_of_study -> field_of_study 'quantum dimensions', paper |?1| -> paper___has_topic___field_of_study -> field_of_study '3-partite'"}]
    # q_id: 1
    # answer = "paper 'Numerical Methods for Finding Stationary Gravitational Solutions' -> paper___cites___paper -> paper |x1|, author |x2| -> author___writes___paper -> paper |x1|, author |x2| -> author___affiliated_with___institution -> institution 'Imperial College London'"
    # chat_log = [{'role': 'system', 'content': 'You are a helpful assistant. You give short and precise answers. If you are asked for a topic, subject, or object, you only answer with the name of the topic, subject, or object. You use no more words than needed. Do not use dots or quotation marks in your answers. Your answers need to be in a machine-readable format. If you are asked for a number, you only respond with a float number. If you are asked if something exists, do not answer yes, do answer with the best example. If you are given background information by a user, favor answers derived from the background information over your own immanent answer. '}, {'role': 'user', 'content': 'The following statement or question between the three dashes is given: --- Find me a paper written by a co-author from Imperial College London and is cited by a paper called "Numerical Methods for Finding Stationary Gravitational Solutions ---\n\nI want to lookup the the answer in a paper authorship knowledge base. The knowledge base has the entities: [\'author\', \'institution\', \'field_of_study\', \'paper\'], and it has the relations: [\'author___affiliated_with___institution\', \'paper___cites___paper\', \'paper___has_topic___field_of_study\', \'author___writes___paper\']. Return a sequence of triplets (entity1_type entity1_name -> relation -> entity_2_type entity_2_name), separated by commas, describing which entities I should lookup in the knowledge base to answer the question between the three dashes. Use |x1| to label an unknown entity. Use |x2|, |x3|, etc. to label distinct unknown entities. Put the type of an entity in front of its name. Here is an example of such a sequence of triplets: author \'Hemingway\' -> author___writes___paper -> paper |x1|, paper |x1| -> paper___has_topic___field_of_study -> field_of_study \'how mice learn to run\'. Answer without description or explanation.'}, {'role': 'assistant', 'content': "paper 'Numerical Methods for Finding Stationary Gravitational Solutions' -> paper___cites___paper -> paper |x1|, author |x2| -> author___writes___paper -> paper |x1|, author |x2| -> author___affiliated_with___institution -> institution 'Imperial College London'"}]

    answer = answer.split(",")
    unknowns = experiment.skb_b.find_unknowns_from_triplets(answer, experiment.logger,
                                                            experiment.config["find_closest_nodes_cut_off"])

    experiment.logger.log(f"{unknowns = }")
    max_k = experiment.config["k_of_vss_top_k"]
    if len(unknowns > 0):
        prompt_body = "The knowledge base returned the following:\n"
        for key_of_unknown, cand_list in unknowns.items():
            if len(cand_list == 0):
                prompt_body += f"No candidates were found for {key_of_unknown}. "
            if cand_list == 1:
                prompt_body += (f"´{key_of_unknown} = {top_hits[0]}. ")
            elif len(cand_list > max_k):
                cand_list = experiment.skb_b.vss.get_top_k_nodes(query_id=query_id, node_id_mask=cand_list,
                                                     max_k=experiment.config["k_of_vss_top_k"])
                top_hits = [experiment.skb_b.entity_id2name(x) for x in cand_list]
                prompt_body += (f"{len(cand_list)} candidates were found for {key_of_unknown}. "
                                f"The most relevant candidates for {key_of_unknown} = are in descending order:"
                                f" {top_hits}. ")
            else:
                cand_list = experiment.skb_b.vss.get_top_k_nodes(query_id=query_id, node_id_mask=cand_list,
                                                     max_k=experiment.config["k_of_vss_top_k"])
                top_hits = [experiment.skb_b.entity_id2name(x) for x in cand_list]
                prompt_body += (f"{len(cand_list)} candidates were found for {key_of_unknown}. "
                                f"The candidates for {key_of_unknown} = are in descending order: {top_hits}.\n")
            if (len(cand_list) > 0):
                prompt_body += f"Here are some background information about the candidates for {key_of_unknown}:\n"
                for node_id in cand_list:
                    prompt_body += skb.get_doc_info(idx=node_id, add_rel=True, compact=False) + "\n"

        prompt_body += ("\nBased on this results, what is the answer to the initial question between the three dashes?"
                        "Give five alternative answers, separated by comma, starting with the most likely answer")
        answer = ask(prompt_body, experiment, chat_log)

    else:
        # VSS as fallback solution:
        cand_list = experiment.skb_b.vss.get_top_k_nodes(query_id=query_id, node_id_mask=None,
                                                     max_k=experiment.config["k_of_vss_top_k"])
        answer = [experiment.skb_b.entity_id2name(x) for x in cand_list]
        experiment.logger.log(f"unknowns set is empty. The alternative answer by VSS is: {answer}")

    return answer


def eval_on_stark_query(experiment: Experiment, query_id: int, query: str) -> list[str]:
    """
    :param experiment:
    :param query_id:
    :param query:
    :return:
    list of answers in descending order
    """
    prompts = experiment.prompts
    skb = experiment.skb_b.skb
    logger = experiment.logger

    llm_query = f"{prompts['asking_for_entities_head']}{query} ---\n\n{prompts['asking_for_entities_tail']}"
    if experiment.dataset_name == "mag":
        llm_query = llm_query.replace("{node_type_dict}", str(skb.node_type_dict))
    # answer, chat_log = ask(llm_query, experiment)
    # answer = 'pharyngitis, chemosis, skin disease'
    # chat_log = [{'role': 'system', 'content': 'You are a helpful assistant. You give short and precise answers. If you are asked for a topic, subject, or object, you only answer with the name of the topic, subject, or object. You use no more words than needed. Do not use dots or quotation marks in your answers. Your answers need to be in a machine-readable format. If you are asked for a number, you only respond with a float number. If you are asked if something exists, do not answer yes, do answer with the best example.'}, {'role': 'user', 'content': 'The following statement or question between the three dashes is given: --- I have pharyngitis and chemosis. What skin disease might I have? ---\n\nI want to look up the main entities and their relations in a medical knowledge base. Which distinctive entities from the question should I look up that could support you to answer the question profoundly? Only list the terms, separated by commas. Answer without description or explanation. Do not list very general terms.'}, {'role': 'assistant', 'content': 'pharyngitis, chemosis, skin disease'}]
    # answer = 'WNT pathway, DKK1, KREMEN, DKK family genes'
    # chat_log = [{'role': 'system', 'content': 'You are a helpful assistant. You give short and precise answers. If you are asked for a topic, subject, or object, you only answer with the name of the topic, subject, or object. You use no more words than needed. Do not use dots or quotation marks in your answers. Your answers need to be in a machine-readable format. If you are asked for a number, you only respond with a float number. If you are asked if something exists, do not answer yes, do answer with the best example.'}, {'role': 'user', 'content': 'The following statement or question between the three dashes is given: --- Which pathway is a promising therapeutic target in breast cancer that renders the protein insensitive to inhibition by the WNT antagonist DKK1 and interacts with KREMEN, DKK family genes? ---\n\nI want to look up the main entities and their relations in a medical knowledge base. Which distinctive entities from the question should I look up that could support you to answer the question profoundly? Only list the terms, separated by commas. Answer without description or explanation. Do not list very general terms.'}, {'role': 'assistant', 'content': 'WNT pathway, DKK1, KREMEN, DKK family genes'}]
    # MAG human nr. 0:
    answer = 'Imperial College London (1), Numerical Methods for Finding Stationary Gravitational Solutions (3)'
    chat_log = [{'role': 'system',
                 'content': 'You are a helpful assistant. You give short and precise answers. If you are asked for a topic, subject, or object, you only answer with the name of the topic, subject, or object. You use no more words than needed. Do not use dots or quotation marks in your answers. Your answers need to be in a machine-readable format. If you are asked for a number, you only respond with a float number. If you are asked if something exists, do not answer yes, do answer with the best example. If you are given background information by a user, favor answers derived from the background information over your own immanent answer. '},
                {'role': 'user',
                 'content': 'The following statement or question between the three dashes is given: --- Find me a paper written by a co-author from Imperial College London and is cited by a paper called "Numerical Methods for Finding Stationary Gravitational Solutions ---\n\nI want to look up the main entities and their relations in a paper authorship knowledge base. Which distinctive entities of the following types from the question should I look up that could support you to answer the question profoundly? I can look up entities of the following types:\n{0: \'author\', 1: \'institution\', 2: \'field_of_study\', 3: \'paper\'}\nList the entities separated by commas. Add the number indicating the type in round brackets after each entity. Answer without description or explanation. Do not list very general terms.'},
                {'role': 'assistant',
                 'content': 'Imperial College London (1), Numerical Methods for Finding Stationary Gravitational Solutions (3)'}]

    entities = prepare_entities_search_list(answer, experiment.dataset_name, skb)
    logger.log(f"searched entities: {entities}")
    node_ids = experiment.skb_b.find_closest_nodes(entities, cutoff=experiment.config["find_closest_nodes_cut_off"],
                                                   drop_duplicates=True)

    logger.log(f"found entities: {[experiment.skb_b.entity_id2name(id) for id in node_ids]}")

    # ____________________________________________________________-

    # answer, chat_log = ask_for_unknown_entity(experiment, skb_b.skb.node_type_dict, chat_log)
    answer = '3'
    chat_log = [{'role': 'system',
                 'content': 'You are a helpful assistant. You give short and precise answers. If you are asked for a topic, subject, or object, you only answer with the name of the topic, subject, or object. You use no more words than needed. Do not use dots or quotation marks in your answers. Your answers need to be in a machine-readable format. If you are asked for a number, you only respond with a float number. If you are asked if something exists, do not answer yes, do answer with the best example. If you are given background information by a user, favor answers derived from the background information over your own immanent answer. '},
                {'role': 'user',
                 'content': 'The following statement or question between the three dashes is given: --- Find me a paper written by a co-author from Imperial College London and is cited by a paper called "Numerical Methods for Finding Stationary Gravitational Solutions ---\n\nI want to look up the main entities and their relations in a paper authorship knowledge base. Which distinctive entities of the following types from the question should I look up that could support you to answer the question profoundly? I can look up entities of the following types:\n{0: \'author\', 1: \'institution\', 2: \'field_of_study\', 3: \'paper\'}\nList the entities separated by commas. Add the number indicating the type in round brackets after each entity. Answer without description or explanation. Do not list very general terms.'},
                {'role': 'assistant',
                 'content': 'Imperial College London (1), Numerical Methods for Finding Stationary Gravitational Solutions (3)'},
                {'role': 'user',
                 'content': "Is the question between the three dashes asking for an unknown entity of one of the following types? If yes, is the type of the entity in the following dictionary? If yes, return the number in front. In all other cases return -1.\nThe dictionary is:\n{0: 'author', 1: 'institution', 2: 'field_of_study', 3: 'paper'}"},
                {'role': 'assistant', 'content': '3'}]

    type_of_unknown = experiment.skb_b.get_node_type_from_key_str(answer, experiment.logger)

    informed_query = prepare_informed_query(prompts, node_ids, query, type_of_unknown, experiment, max_depth=4)
    raw_answer, chat_log_w_skb = ask(informed_query, experiment, chat_log=None)

    # answer = answer.split('\n', 1)[0]
    answer = raw_answer.split(',', 5)
    if len(answer) > 5:
        answer = answer[0:5]
    answer = [a.strip() for a in answer]

    expected_answers = experiment.expected_answers(query_id, separator=', ')
    logger.log(f"{expected_answers = }", print_to_console=True)
    logger.save_result(query_id, query, raw_answer, expected_answers)

    return answer


if __name__ == '__main__':
    main(question_ids=[0, 1, 2], dataset_name='mag', experiment_name='tmp_2024-08_08_mag',
         data_split='human_generated_eval')

