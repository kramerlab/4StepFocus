import json
from pathlib import Path

from dotenv import load_dotenv
from stark_qa import load_qa, load_skb
from stark_qa.skb import SKB

from logger import Logger
from skb_bridge import SKBbridge
from vss import VSS


class Experiment:
    def __init__(self, name: str, dataset_name: str, data_split: str, config_path: Path = None,
                 llm_model: str = None, skb: SKB = None, enable_vss: bool = True):
        if dataset_name not in ['prime', 'mag', 'amazon']:
            raise ValueError(f"Dataset {dataset_name} not found. It should be in ['prime', 'mag,', 'amazon']")
        if data_split not in ["train", "val", "test", "human_generated_eval"]:
            raise ValueError(f"split {data_split} not found. "
                             f"It should be in ['train', 'val,', 'test', 'human_generated_eval']")

        self.name = name
        self.dataset_name = dataset_name

        # load configs and prompts
        if config_path is None:
            config_path = Path(__file__).parent
        with open(config_path / 'prompts.json') as json_file:
            prompts = json.load(json_file)
        with open(config_path / "config.json", "r") as json_file:
            self.config = json.load(json_file)
        data_specific_prompts = prompts[dataset_name]
        self.prompts = prompts["general"]
        self.prompts.update(data_specific_prompts)

        # load SKB bridge and llm_model
        if llm_model is None:
            self.llm_model = self.config["llm_model"]
        else:
            self.llm_model = llm_model
        if skb is None:
            skb_path = Path(__file__).parent / self.config["stark_data_path"]
            skb = self.skb = load_skb(name=dataset_name, download_processed=True, root=skb_path)

        if enable_vss:
            query_emb_dir = Path(__file__).parent / self.config["embeddings_path"] / dataset_name
            vss = VSS(skb, query_emb_dir, data_split, self.config["embedding_model"])
        else:
            vss = None
        self.skb_b = SKBbridge(dataset_name, skb=skb, vss=vss)

        # load data
        self.eval_data = load_qa(name=dataset_name, human_generated_eval=data_split == "human_generated_eval",
                                 root=Path(__file__).parent / self.config["stark_data_path"])
        if data_split != "human_generated_eval":
            self.eval_data = self.eval_data.get_subset(data_split)

        self.logger = Logger(Path(__file__).parent / self.config["output_path"] / self.dataset_name)

    def qa_pair2str(self, q_id: int) -> str:
        query, q_id, answer_ids, _ = self.eval_data[q_id]

        out = f"\n++++++++++ question nr {q_id} ++++++++++++++\n"
        out += query + "\nAnswers:\n"
        expected_answers = self.expected_answers(q_id, separator=" OR ", answer_ids=answer_ids)
        out += expected_answers
        out += f"\n++++++++++ end of question nr {q_id} ++++++++++++++\n\n"
        return out

    def expected_answers(self, q_id: int, separator: str = ", ", answer_ids=None) -> str:
        if answer_ids is None:
            _, _, answer_ids, _ = self.eval_data[q_id]
        out = ""
        if self.dataset_name == 'prime':
            out += separator.join([self.skb_b.skb[aid].name for aid in answer_ids])
        elif self.dataset_name == 'mag':
            for aid in answer_ids:
                if "title" in self.skb_b.skb.node_info[aid]:
                    out += self.skb_b.skb.node_info[aid]['title'] + separator
                else:
                    out += self.skb_b.skb[aid]['DisplayName'] + separator
        else:
            raise NotImplementedError()
        return out

    def get_query(self, q_id: int) -> str:
        return self.eval_data[q_id][0]
