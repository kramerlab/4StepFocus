import stark_qa.skb
from typing import Any, Union, List, Dict

from experiment import Experiment
from main import eval_on_stark_query
from skb_bridge import SKBbridge
from stark_main.models.model import ModelForSTaRKQA


class LLMConsultant(ModelForSTaRKQA):

    def __init__(self,
                 skb: stark_qa.skb.SKB,
                 llm_model: str,
                 dataset_name: str,
                 split: str
                 ):
        """
        """
        super(LLMConsultant, self).__init__(skb)
        self.skb_b = SKBbridge(dataset_name, skb=skb)
        self.experiment = Experiment("from_stark_benchmarking_hit5_MAG", dataset_name, split,
                                     llm_model=llm_model, skb=skb)

    def forward(self,
                query: Union[str, List[str]],
                candidates: List[int] = None,
                query_id: Union[int, List[int]] = None,
                **kwargs: Any) -> Dict[str, Any]:
        """
        Forward pass to compute predictions for the given query.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            candidates (Union[list, None]): A list of candidate ids (optional).
            query_id (Union[int, list, None]): Query index (optional).

        Returns:
            pred_dict (dict): A dictionary of predicted scores or answer ids.
        """
        pred = eval_on_stark_query(self.experiment, query_id, query)
        node_ids = self.skb_b.find_closest_nodes(pred, cutoff=0.0, drop_duplicates=True)

        out = {}
        for i in self.skb_b.skb.node_info.keys():
            out[i] = 0.
        for i, node_id in enumerate(node_ids):
            out[node_id] = 1.0 - i * 0.1

        return out
