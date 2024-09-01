import os
import os.path as osp
from pathlib import Path

import torch

from typing import Any
from tqdm import tqdm

from stark_qa.tools.api import get_openai_embedding
from stark_qa.skb import SKB


class VSS:
    def __init__(self,
                 skb: SKB,
                 emb_dir: Path,
                 data_split: str,
                 emb_model: str = 'text-embedding-ada-002'):
        self.skb = skb
        self.candidate_ids = skb.candidate_ids
        self.query_emb_dict = {}
        self.emb_model = emb_model

        emb_dir /= emb_model
        if data_split == "human_generated_eval":
            self.query_emb_dir = emb_dir / "query_human_generated_eval_no_rel_no_compact"
        else:
            self.query_emb_dir = emb_dir / "query"
        candidates_emb_dir = emb_dir / "doc"
        entities_emb_dir = emb_dir / "entities"
        entities_emb_dir.mkdir(parents=True, exist_ok=True)

        candidate_emb_path = osp.join(candidates_emb_dir, 'candidate_emb_dict.pt')
        if osp.exists(candidate_emb_path):
            candidate_emb_dict = torch.load(candidate_emb_path)
            print(f'Loaded candidate_emb_dict from {candidate_emb_path}!')
        else:
            print('Loading candidate embeddings...')
            candidate_emb_dict = {}
            for idx in tqdm(self.candidate_ids):
                candidate_emb_dict[idx] = torch.load(osp.join(candidates_emb_dir, f'{idx}.pt'))
            torch.save(candidate_emb_dict, candidate_emb_path)
            print(f'Saved candidate_emb_dict to {candidate_emb_path}!')

        assert len(candidate_emb_dict) == len(self.candidate_ids)
        self.candidate_embs = [candidate_emb_dict[idx] for idx in self.candidate_ids]
        self.candidate_embs = torch.cat(self.candidate_embs, dim=0)

        self.entity_emb_path = entities_emb_dir / 'entity_emb_dict.pt'
        if self.entity_emb_path.exists():
            self.entity_emb_dict = torch.load(self.entity_emb_path)
        else:
            self.entity_emb_dict = {}
        print(f'Loaded entity_emb_dict from {self.entity_emb_path}!')

    def forward(self,
                query: str,
                query_id: int,
                **kwargs: Any) -> dict:
        """
        Forward pass to compute similarity scores for the given query.

        Args:
            query (str): Query string.
            query_id (int): Query index.

        Returns:
            pred_dict (dict): A dictionary of candidate ids and their corresponding similarity scores.
        """
        query_emb = self.get_query_emb(query, query_id, emb_model=self.emb_model)
        similarity = torch.matmul(query_emb, self.candidate_embs.T).view(-1)
        pred_dict = {self.candidate_ids[i]: similarity[i] for i in range(len(self.candidate_ids))}
        return pred_dict

    def get_query_emb(self,
                      query: str,
                      query_id: int,
                      emb_model: str = 'text-embedding-ada-002') -> torch.Tensor:
        """
        Retrieves or computes the embedding for the given query.

        Args:
            query (str): Query string.
            query_id (int): Query index.
            emb_model (str): Embedding model to use.

        Returns:
            query_emb (torch.Tensor): Query embedding.
        """
        if query_id is None:
            if query in self.entity_emb_dict:
                query_emb = self.entity_emb_dict[query]
            else:
                query_emb = get_openai_embedding(query, model=emb_model)
                self.entity_emb_dict[query] = query_emb
                torch.save(self.entity_emb_dict, self.entity_emb_path)
        elif len(self.query_emb_dict) > 0:
            query_emb = self.query_emb_dict[query_id]
        else:
            query_emb_dic_path = osp.join(self.query_emb_dir, 'query_emb_dict.pt')
            if os.path.exists(query_emb_dic_path):
                print(f'Load query embeddings from {query_emb_dic_path}')
                self.query_emb_dict = torch.load(query_emb_dic_path)
                query_emb = self.query_emb_dict[query_id]
            else:
                query_emb_dir = osp.join(self.query_emb_dir, 'query_embs')
                if not os.path.exists(query_emb_dir):
                    os.makedirs(query_emb_dir)
                query_emb_path = osp.join(query_emb_dir, f'query_{query_id}.pt')
                query_emb = get_openai_embedding(query, model=emb_model)
                torch.save(query_emb, query_emb_path)
        return query_emb

    def get_top_k_nodes(self, search_str: str = None, node_id_mask: list[int] = None, max_k: int = 1, query_id: int = None):

        print(len(node_id_mask))

        score_dict = self.forward(search_str, query_id=query_id)
        print(len(score_dict))
        torch.save(score_dict, "TMP_score_dict.pt")
        torch.save(node_id_mask, "TMP_node_id_mask.pt")
        if node_id_mask is not None:
            score_dict = {node_id: score_dict[node_id] for node_id in node_id_mask if node_id in score_dict}
        node_ids = list(score_dict.keys())
        node_scores = list(score_dict.values())

        print(len(node_ids))

        # Get the ids with top k highest scores
        top_k_idx = torch.topk(
            torch.FloatTensor(node_scores),
            min(max_k, len(node_scores)),
            dim=-1,
            largest=True,
            sorted=True
        ).indices.tolist()

        print(len(top_k_idx))

        top_k_node_ids = [node_ids[i] for i in top_k_idx]
        print(len(top_k_node_ids))
        print("HI THERE ! ")

        return top_k_node_ids
