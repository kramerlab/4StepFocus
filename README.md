# 4StepNarrower
This Readme is currently under construction. Further notes will be added soon.



## Getting started
The code in this repository supplements our recent paper that we aim to publish soon.

We integrated *stark_main* in this directory from https://github.com/snap-stanford/stark/tree/main.

We added *bridge_to_llm_consultant.py* to stark_main -> models and added it in stark_main -> models -> __init__.py in this directory.

Add a file **.env* with your OpenAI api key ```OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXX ´´´ in the main directory.

### Download data

### Install from pip (recommended)

With python >=3.8 and <3.12
```
pip install stark-qa
pip install openai
```

### Start evaluation
```
python -m stark_main.eval
```
For example:
```
python -m stark_main.eval
 --dataset mag --model LLMConsultant --split test --output_dir output --llm_model gpt-4o-2024-05-13 --emb_dir emb --llm_topk 20 --max_retry 2 --save_pred --test_ratio 0.1
```