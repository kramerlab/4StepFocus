import difflib
from collections.abc import Iterable

from stark_qa.skb import SKB

from logger import Logger
from pathfinding import find_edge_type
from utils import edge_type2str
from vss import VSS


def create_node_dict_prime(skb: SKB):
    nodes_alias2id = {}
    for n_type in skb.node_type_lst():
        nodes_alias2id[n_type] = {}

    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        n_type = node["type"]
        n_name = node['name'].lower()
        nodes_alias2id[n_type][n_name] = i
        if 'details' in node:
            if 'alias' in node['details']:
                alias = node['details']['alias']
                if isinstance(alias, list):
                    for a in alias:
                        nodes_alias2id[n_type][a.lower()] = i
                else:
                    nodes_alias2id[n_type][alias.lower()] = i
    return nodes_alias2id


def create_node_dict_mag(skb: SKB):
    nodes_alias2id = {}
    for n_type in skb.node_type_lst():
        nodes_alias2id[n_type] = {}

    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        n_type = node["type"]
        if 'title' in node:
            nodes_alias2id[n_type][node['title'].lower()] = i
        elif 'DisplayName' in node and node['DisplayName'] != -1 and node['DisplayName'] != "-1":
            nodes_alias2id[n_type][node['DisplayName'].lower()] = i
    return nodes_alias2id


def create_node_dict_amazon(skb: SKB):
    raise NotImplementedError("ID: 456klj23hed (create_node_dict_amazon not implemented)")


def find_closest_nodes_mag(targets: Iterable, nodes_alias2id: {}, all_node_types: list[str], cutoff: float) -> set:
    raise NotImplementedError("Not further implemented, because nodes are not sorted by node type at the current state"
                              "of development.")
    node_ids = []
    # TODO /// targets ... /// get their types then search
    for n_name, n_type in targets:
        if n_type == -1:
            n_type = all_node_types
        else:
            n_type = [n_type]

        closest_key = difflib.get_close_matches(target.lower(), nodes_alias2id[n_type], n=1, cutoff=0.9)
        if len(closest_key) > 0:
            closest_key = closest_key[0]
            node_ids.append(nodes_alias2id[closest_key])
    return node_ids


class SKBbridge:
    def __init__(self, dataset_name: str, skb: SKB = None, vss: VSS = None):
        if dataset_name not in ['prime', 'mag', 'amazon']:
            raise ValueError(f"Dataset {dataset_name} not found. It should be in ['prime', 'mag,', 'amazon']")

        self.name = dataset_name
        self.vss = vss
        self.skb = skb

        self.nodes_alias2id = None
        self.create_node_dict()  # replaces self.nodes_alias2id

        self.is_directed = False
        if self.name == 'prime':
            self.is_directed = True

    def create_node_dict(self):
        if self.name == 'prime':
            self.nodes_alias2id = create_node_dict_prime(self.skb)
        elif self.name == 'mag':
            self.nodes_alias2id = create_node_dict_mag(self.skb)
        elif self.name == 'amazon':
            self.nodes_alias2id = create_node_dict_amazon(self.skb)
        else:
            raise ValueError(f"dataset name should be in ['prime', 'mag,', 'amazon'], but '{self.name}' is given")

    def find_closest_nodes(self, targets: Iterable, cutoff: float, drop_duplicates: bool) -> set:
        # if self.name == "mag":
        #     return find_closest_nodes_mag(targets, self.nodes_alias2id, cutoff)
        node_ids = []
        for target in targets:
            n_type = target[1]
            n_name = target[0].lower()
            if n_type is None:
                closest_node_name = ""
                highest_similarity = 0.
                closest_node_type = ""
                for n_type in self.skb.node_type_lst():
                    potential_closest_node = difflib.get_close_matches(n_name, self.nodes_alias2id[n_type],
                                                                       n=1, cutoff=cutoff)
                    if len(potential_closest_node) > 0:
                        similarity = difflib.SequenceMatcher(None, n_name, potential_closest_node[0]).ratio()
                        if similarity > highest_similarity:
                            closest_node_name = potential_closest_node[0]
                            highest_similarity = similarity
                            closest_node_type = n_type
                if closest_node_name != "":
                    node_ids.append(self.nodes_alias2id[closest_node_type][closest_node_name])
                elif self.vss is not None:
                    node_ids.append(self.vss.get_top_k_nodes(n_name, node_id_mask=None, max_k=1)[0])
            else:
                closest_node_name = difflib.get_close_matches(n_name, self.nodes_alias2id[n_type], n=1, cutoff=cutoff)
                if len(closest_node_name) > 0:
                    closest_node_name = closest_node_name[0]
                    node_ids.append(self.nodes_alias2id[n_type][closest_node_name])
                else:
                    filtered_node_ids = self.skb.get_node_ids_by_type(n_type)
                    node_ids.append(self.vss.get_top_k_nodes(n_name, node_id_mask=filtered_node_ids, max_k=1)[0])
        if drop_duplicates:
            node_ids = list(set(node_ids))
        return node_ids

    def find_closest_nodes_w_VSS(self, targets: Iterable, cutoff: float, drop_duplicates: bool) -> set:
        # if self.name == "mag":
        #     return find_closest_nodes_mag(targets, self.nodes_alias2id)
        node_ids = []
        for target in targets:
            n_type = target[1]
            n_name = target[0].lower()
            if n_type is None:
                closest_node_name = ""
                highest_similarity = 0.
                closest_node_type = ""
                for n_type in self.skb.node_type_lst():
                    potential_closest_node = difflib.get_close_matches(n_name, self.nodes_alias2id[n_type],
                                                                       n=1, cutoff=cutoff)
                    if len(potential_closest_node) > 0:
                        similarity = difflib.SequenceMatcher(None, n_name, potential_closest_node[0]).ratio()
                        if similarity > highest_similarity:
                            closest_node_name = potential_closest_node[0]
                            highest_similarity = similarity
                            closest_node_type = n_type
                if closest_node_name != "":
                    node_ids.append(self.nodes_alias2id[closest_node_type][closest_node_name])
                else:
                    print('TODO') # vss.
            else:
                closest_node_name = difflib.get_close_matches(n_name, self.nodes_alias2id[n_type], n=1, cutoff=cutoff)
                if len(closest_node_name) > 0:
                    closest_node_name = closest_node_name[0]
                    node_ids.append(self.nodes_alias2id[n_type][closest_node_name])
        if drop_duplicates:
            node_ids = list(set(node_ids))
        return node_ids

    def entity_id2name(self, id: int):
        if self.name == 'prime':
            return self.skb.node_info[id]['name']
        if self.name == 'mag':
            node = self.skb.node_info[id]
            if 'title' in node:
                return node['title']
            elif 'DisplayName' in node and node['DisplayName'] != -1 and node['DisplayName'] != "-1":
                return node['DisplayName']
            else:
                return f"node without name. id: {id}"

        raise NotImplementedError(f"Not implemented for dataset {self.name}")

    def path2str(self, path):
        out = ""
        for i in range(len(path) - 1):
            edge_type = find_edge_type(path[i], path[i + 1], self.skb)
            out += (f"{self.skb.node_info[path[i]]['name']} {edge_type2str(self.name, edge_type)} "
                    f"{self.skb.node_info[path[i + 1]]['name']}, ")
        return out[:-2]

    def nodes2str(self, node_ids: int | list[str]):
        if isinstance(node_ids, list):
            out = []
            for node_id in node_ids:
                out.append(self.skb.get_doc_info(node_id, add_rel=False, compact=False))
            return out
        else:
            return self.skb.get_doc_info(node_ids, add_rel=False, compact=False)

    def get_node_type_from_key_str(self, key_str, logger: Logger):
        type_of_unknown = None
        try:
            key_str = int(key_str)
            if key_str in self.skb.node_type_dict:
                type_of_unknown = self.skb.node_type_dict[key_str]
        except ValueError("Input should be natural number or -1."):
            logger.log("ValueError(Input should be natural number or -1.)")
        return type_of_unknown

    def find_unknowns_from_triplets(self, triplets: list[str], logger: Logger, cutoff: float) -> {}:
        skb = self.skb

        unknowns = {}
        unknowns_sizes = []
        nothing_changed = False
        rounds = 0

        while (not nothing_changed):
            logger.log(f"Round {rounds}:")

            for triplet in triplets:
                triplet = triplet.split("->")
                if len(triplet) != 3:
                    logger.log("triplet" + str(triplet) + "is not valid.")
                    continue
                u = triplet[0].strip()
                e = triplet[1].strip()
                v = triplet[2].strip()

                if e not in skb.edge_type_dict.values():
                    logger.log("Warning: Edge type " + e + " not found in knowledge base. Replacing it with None.")
                    e = None
                # u:
                if "'" in u:
                    u = u.split("'")
                    u_type = u[0].strip()
                    u_name = u[1].strip()
                    if u_type not in skb.node_type_lst():
                        logger.log("Warning: Node type " + u_type + " not found in knowledge base. Replacing it with None.")
                        u_type = None
                    node_ids = self.find_closest_nodes([(u_name, u_type)],
                                                        cutoff=cutoff,
                                                        # experiment.config["find_closest_nodes_cut_off"],
                                                        drop_duplicates=True)
                    if len(node_ids) == 0:
                        logger.log("Node " + u + " not found in knowledge base.")
                        continue
                    u = node_ids[0]
                    u_is_constant = True
                    logger.log(f"Found entity: {self.entity_id2name(u)} ({u_type})")
                elif "|" in u:
                    u_is_constant = False
                    u = u.split("|")
                    u_type = u[0].strip()
                    u_name = u[1].strip()
                else:
                    logger.log("first node " + u + "is not valid.")
                    continue

                # v:
                if "'" in v:
                    v = v.split("'")
                    v_type = v[0].strip()
                    v_name = v[1].strip()
                    if v_type not in skb.node_type_lst():
                        v_type = None
                    node_ids = self.find_closest_nodes([(v_name, v_type)],
                                                        cutoff=cutoff,
                                                        # experiment.config["find_closest_nodes_cut_off"],
                                                        drop_duplicates=True)
                    if len(node_ids) == 0:
                        logger.log("Node " + v + " not found in knowledge base.")
                        continue
                    v = node_ids[0]
                    v_is_constant = True
                    logger.log(f"Found entity: {self.entity_id2name(v)} ({v_type})")
                elif "|" in v:
                    v = v.split("|")
                    v_type = v[0].strip()
                    v_name = v[1].strip()
                    v_is_constant = False
                else:
                    logger.log("second node " + v + "is not valid.")
                    continue

                if u_is_constant and not v_is_constant:
                    candidates = set(skb.get_neighbor_nodes(u, e))

                    if v_name in unknowns:
                        unknowns[v_name].intersection_update(candidates)
                    else:
                        unknowns[v_name] = candidates
                    logger.log(
                        f"Found triplet: {self.entity_id2name(u)} ({u_type}) -> {e} -> {len(unknowns[v_name])} candidates of type {v_type}")
                elif not u_is_constant and v_is_constant:
                    candidates = set(skb.get_neighbor_nodes(v, e))

                    if u_name in unknowns:
                        unknowns[u_name].intersection_update(candidates)
                    else:
                        unknowns[u_name] = candidates
                    logger.log(
                        f"Found triplet: {len(unknowns[u_name])} candidates of type {u_type} -> {e} -> {self.entity_id2name(v)} ({v_type})")
                elif not u_is_constant and not v_is_constant:
                    if u_name in unknowns:
                        candidates = set()
                        for u_candidate in unknowns[u_name]:
                            candidates = candidates.union(set(skb.get_neighbor_nodes(u_candidate, e)))
                        if v_name in unknowns:
                            unknowns[v_name].intersection_update(candidates)
                        else:
                            unknowns[v_name] = candidates

                    if v_name in unknowns:
                        candidates = set()
                        for v_candidate in unknowns[v_name]:
                            candidates = candidates.union(set(skb.get_neighbor_nodes(v_candidate, e)))
                        if u_name in unknowns:
                            unknowns[u_name].intersection_update(candidates)
                        else:
                            unknowns[u_name] = candidates
                    logger.log(f"Found {len(candidates)} candidates for triplet: type {u_type} -> {e} -> {v_type}")

            new_unknown_lengths = [len(x) for x in unknowns.values()]
            if unknowns_sizes == new_unknown_lengths:
                nothing_changed = True
            else:
                unknowns_sizes = new_unknown_lengths
        return unknowns
