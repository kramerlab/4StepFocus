from stark_qa.skb import SKB


def edge_type2str(dataset_name: str, key: str) -> str:
    if dataset_name == 'prime':
        conv = {
            'ppi': 'interacts with other protein',
            'carrier': 'is a carrier of or carried by',
            'enzyme': 'enzymes or is enzymed by',
            'target': 'targets or is targeted by',
            'transporter': 'transports or is transported by',
            'contraindication': 'contraindicates or is contraindicated by',
            'indication': 'indicates or is indicated by',
            'off-label use': 'has a off-label use relation with',
            'synergistic interaction': 'synergistically interacts with',
            'associated with': 'is associated with',
            'parent-child': 'is a parent or a subtype of',
            'phenotype absent': 'has a phenotype-absent relation with',
            'phenotype present': 'has a phenotype-present relation with',
            'side effect': 'has a side-effect relation with',
            'interacts with': 'interacts with',
            'linked to': 'is linked to',
            'expression present': 'has an expression present relation with',
            'expression absent': 'has an expression absent relation with'}
    else:
        conv = {}
    if key in conv:
        return conv[key]
    else:
        return f"is {key} of"


def prepare_entities_search_list(entities_string: str, dataset_name: str, skb: SKB) -> set[str]:
    raw_entities = [item.strip().lower() for item in entities_string.split(",")]

    # remove basic hub nodes
    for e in raw_entities.copy():
        if e in (skb.node_type_lst() + ['sub_type']):
            raw_entities.remove(e)

    if dataset_name == "prime":
        # generate (name, type) entity tuples
        entities = [(e, None) for e in raw_entities]

        # add combined entities:
        for u in raw_entities:
            for v in raw_entities:
                if u != v:
                    entities.append((u + " " + v, None))
                    entities.append((v + " " + u, None))

    elif dataset_name == "mag":
        # separate node type and node name from string
        entities = []
        for e in raw_entities:
            e = e.split("(", 1)
            entity_name = e[0]

            entity_type = None
            # is there a bracket after the entity?
            if len(e) > 1:
                entity_type = e[1].split(")")[0]
                try:
                    entity_type = int(entity_type)
                    if entity_type in skb.node_type_dict:
                        entity_type = skb.node_type_dict[entity_type]
                except ValueError:
                    entity_type = None
            # entity type not used at the current state of development
            entities.append((entity_name.strip(), entity_type))
    else:
        raise ValueError(f"dataset_name must be in [prime, mag, amazon] but is {dataset_name}.")
    return entities