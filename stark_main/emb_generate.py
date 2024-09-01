import os
import os.path as osp
import random
import sys
import argparse

import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append('.')
from stark_qa import load_skb, load_qa
from stark_qa.tools.api import get_openai_embeddings


def parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser()

    # Dataset and embedding model selection
    parser.add_argument('--dataset', default='amazon', choices=['amazon', 'prime', 'mag'])
    parser.add_argument('--emb_model', default='text-embedding-ada-002', 
                        choices=[
                            'text-embedding-ada-002', 
                            'text-embedding-3-small', 
                            'text-embedding-3-large'
                            ]
                        )

    # Mode settings
    parser.add_argument('--mode', default='doc', choices=['doc', 'query'])

    # Path settings
    parser.add_argument("--emb_dir", default="emb/", type=str)

    # Text settings
    parser.add_argument('--add_rel', action='store_true', default=False, help='add relation to the text')
    parser.add_argument('--compact', action='store_true', default=False, help='make the text compact when input to the model')

    # Evaluation settings
    parser.add_argument("--human_generated_eval", action="store_true", help="if mode is `query`, then generating query embeddings on human generated evaluation split")

    # Batch and node settings
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--n_max_nodes", default=10, type=int)

    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()
    mode_surfix = '_human_generated_eval' if args.human_generated_eval and args.mode == 'query' else ''
    mode_surfix += '_no_rel' if not args.add_rel else ''
    mode_surfix += '_no_compact' if not args.compact else ''
    emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f'{args.mode}{mode_surfix}')
    print(f'Embedding directory: {emb_dir}')
    os.makedirs(emb_dir, exist_ok=True)

    if args.mode == 'doc':
        kb = load_skb(args.dataset)
        lst = kb.candidate_ids
        emb_path = osp.join(emb_dir, f'candidate_emb_dict.pt')
    if args.mode == 'query':
        qa_dataset = load_qa(args.dataset, human_generated_eval=args.human_generated_eval,
                             )# root=f"../stark_datasets")
        lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
        emb_path = osp.join(emb_dir, f'query_emb_dict.pt')
    random.shuffle(lst)
            
    if osp.exists(emb_path):
        emb_dict = torch.load(emb_path)
        exisiting_indices = list(emb_dict.keys())
        print(f'Loaded existing embeddings from {emb_path}. Size: {len(emb_dict)}')
    else:
        emb_dict = {}
        exisiting_indices = []

    texts, indices = [], []
    for idx in tqdm(lst):
        if idx in exisiting_indices:
            continue
        if args.mode == 'query':
            text = qa_dataset.get_query_by_qid(idx)
        elif args.mode == 'doc':
            text = kb.get_doc_info(idx, add_rel=args.add_rel, compact=args.compact)
        texts.append(text)
        indices.append(idx)
        
    print(f'Generating embeddings for {len(texts)} texts...')
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i:i+args.batch_size]
        batch_embs = get_openai_embeddings(
            batch_texts, 
            model=args.emb_model, 
            n_max_nodes=args.n_max_nodes
            ).view(len(batch_texts), -1).cpu()
        batch_indices = indices[i:i+args.batch_size]
        for idx, emb in zip(batch_indices, batch_embs):
            emb_dict[idx] = emb
        
    torch.save(emb_dict, emb_path)
    print(f'Saved {len(emb_dict)} embeddings to {emb_path}!')
