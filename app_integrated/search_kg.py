#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

# We'll reuse rapidfuzz for text similarity
# pip install rapidfuzz
from rapidfuzz import fuzz


def load_kg(kg_path):
    """Loads the knowledge graph JSON from disk."""
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"KG file not found: {kg_path}")
    with open(kg_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def search_nodes(kg_obj, query, top_k=5):
    """
    Find the top_k nodes in the KG with the highest text similarity to `query`.
    We'll compare against node["text"] plus node["label"].
    Returns a list of tuples: [(score, node), ...].
    """
    results = []
    for node in kg_obj["nodes"]:
        candidate_text = node["label"] + " " + node["text"]
        score = fuzz.partial_ratio(query.lower(), candidate_text.lower())
        results.append((score, node))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def gather_related_edges(kg_obj, node_ids):
    """
    Return all edges that involve any of the given node_ids.
    """
    relevant_edges = []
    for edge in kg_obj["edges"]:
        if edge["source"] in node_ids or edge["target"] in node_ids:
            relevant_edges.append(edge)
    return relevant_edges

def build_id_to_text_map(kg_obj):
    """
    Create a dict mapping node_id -> node_text for easy lookups.
    We combine the label and text, or just text—adapt to your preference.
    """
    mapping = {}
    for node in kg_obj["nodes"]:
        # If you only want the node text, do node["text"]
        # or combine label + text, if you prefer
        mapping[node["id"]] = node["text"]
    return mapping

def query_kg(kg_path, query, top_k=5):
    kg_obj = load_kg(kg_path)
    top_matches = search_nodes(kg_obj, query, top_k=top_k)
    return [{"label": node["label"], "text": node["text"], "score": score} for score, node in top_matches]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_path", "-k", default="my_knowledge_graph.json",
                        help="Path to the persistent KG JSON file.")
    parser.add_argument("--query", "-q", required=True,
                        help="Your natural language question or search query.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top matching nodes to return.")
    args = parser.parse_args()

    # 1) Load the KG
    kg_obj = load_kg(args.kg_path)

    # 2) Search for relevant nodes
    top_matches = search_nodes(kg_obj, args.query, top_k=args.top_k)

    # 3) Print the matched nodes WITHOUT showing their IDs
    matched_node_ids = []
    print(f"Query: {args.query}\n")
    print(f"Top {args.top_k} matching nodes:")
    for rank, (score, node) in enumerate(top_matches, start=1):
        matched_node_ids.append(node["id"])
        # Do not display node ID here
        print(f"  {rank}. (score={score:.2f}) Label={node['label']} | Text={node['text']}")

    # 4) Gather relevant edges
    relevant_edges = gather_related_edges(kg_obj, matched_node_ids)

    # 4.1) Build a map from node IDs to text, so we can print edges with text
    id_to_text = build_id_to_text_map(kg_obj)

    if relevant_edges:
        print("\nRelevant edges involving these matched nodes:")
        for e in relevant_edges:
            source_text = id_to_text[e["source"]]
            target_text = id_to_text[e["target"]]
            relation_type = e["type"]
            # Print in the requested format:
            # [text of source] RELATION_TYPE [text of object]
            print(f"  {source_text} {relation_type} {target_text}")
    else:
        print("\nNo edges found for these matched nodes.")

    print("\n--- Search Complete ---")


if __name__ == "__main__":
    main()