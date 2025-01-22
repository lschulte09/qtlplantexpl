#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import re
import json
import math
import argparse
from tqdm import tqdm

import pandas as pd
import transformers as huggingface_transformers

from uie.extraction.record_schema import RecordSchema
from uie.sel2record.record import MapConfig
from uie.sel2record.sel2record import SEL2Record

# ------------------------------------------------------------------------------
# Regex & sets for cleaning decoded model outputs
split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {'<pad>', '</s>'}

def post_processing(x: str) -> str:
    """Remove special tokens and extra whitespace."""
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()

def schema_to_ssi(schema: RecordSchema) -> str:
    """
    Convert RecordSchema into the special “spot/asoc” prompt
    that the UIE T5 model expects.
    """
    ssi = "<spot> " + "<spot> ".join(sorted(schema.type_list))
    ssi += "<asoc> " + "<asoc> ".join(sorted(schema.role_list))
    ssi += "<extra_id_2> "
    return ssi

def read_jsonl(path):
    """Reads a JSON file line-by-line (each line is a dict)."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# --------------------------------------------------------------------------
# Simple T5-based predictor
# --------------------------------------------------------------------------
class HuggingfacePredictor:
    def __init__(self, model_path, schema_file,
                 max_source_length=512, max_target_length=512):
        # Load T5 tokenizer & model
        self._tokenizer = huggingface_transformers.T5TokenizerFast.from_pretrained(model_path)
        self._model = huggingface_transformers.T5ForConditionalGeneration.from_pretrained(model_path)
        self._model.cuda()

        # Load schema & build the special ssi prompt
        self._schema = RecordSchema.read_from_file(schema_file)
        self._ssi = schema_to_ssi(self._schema)

        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    def predict(self, text_list):
        """Given a list of raw text strings, prepend the schema prompt and run inference."""
        # Prepend schema prompt
        text_list = [self._ssi + t for t in text_list]

        # Tokenize
        inputs = self._tokenizer(
            text_list,
            padding=True,
            return_tensors='pt',
            truncation=True
        )
        inputs = inputs.to(self._model.device)

        # Truncate if needed
        inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self._max_source_length]

        # Generate predictions
        outputs = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length,
        )
        # Decode
        decoded = self._tokenizer.batch_decode(
            outputs,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        decoded = [post_processing(d) for d in decoded]
        return decoded

# --------------------------------------------------------------------------
# Manual matching code
# --------------------------------------------------------------------------
def get_entity_set_gold(gold_entities):
    """Convert gold entity list -> list of (type, text)."""
    results = []
    for ent in gold_entities:
        etype = ent.get("type", "").strip()
        etext = ent.get("text", "").strip()
        results.append((etype, etext))
    return results

def get_entity_set_pred(pred_entities):
    """
    Convert predicted entity structure into list of (type, text).
    Format can differ depending on how the predictions are structured.
    """
    # If the predicted structure is a dict with a "string" key:
    if isinstance(pred_entities, dict) and "string" in pred_entities:
        predicted_list = pred_entities["string"]
        return [(item[0].strip(), item[1].strip()) for item in predicted_list]
    elif isinstance(pred_entities, list):
        results = []
        for ent in pred_entities:
            etype = ent.get("type", "").strip()
            etext = ent.get("text", "").strip()
            results.append((etype, etext))
        return results
    else:
        return []

def get_relation_set_gold(gold_relations):
    """
    Convert gold relations -> list of (relation_type, (arg1_type, arg1_text), (arg2_type, arg2_text)).
    """
    results = []
    for rel in gold_relations:
        rtype = rel.get("type", "").strip()
        args = rel.get("args", [])
        if len(args) == 2:
            arg1 = (args[0].get("type", "").strip(), args[0].get("text", "").strip())
            arg2 = (args[1].get("type", "").strip(), args[1].get("text", "").strip())
            results.append((rtype, arg1, arg2))
    return results

def get_relation_set_pred(pred_relations):
    """
    Convert predicted relation structure -> list of
    (relation_type, (arg1_type, arg1_text), (arg2_type, arg2_text)).
    """
    if isinstance(pred_relations, dict) and "string" in pred_relations:
        relation_list = pred_relations["string"]
        results = []
        for rel_item in relation_list:
            # Example: [rtype, type1, text1, type2, text2]
            if len(rel_item) == 5:
                rtype = rel_item[0].strip()
                arg1 = (rel_item[1].strip(), rel_item[2].strip())
                arg2 = (rel_item[3].strip(), rel_item[4].strip())
                results.append((rtype, arg1, arg2))
        return results
    elif isinstance(pred_relations, list):
        results = []
        for rel in pred_relations:
            rtype = rel.get("type", "").strip()
            args = rel.get("args", [])
            if len(args) == 2:
                arg1 = (args[0].get("type", "").strip(), args[0].get("text", "").strip())
                arg2 = (args[1].get("type", "").strip(), args[1].get("text", "").strip())
                results.append((rtype, arg1, arg2))
        return results
    else:
        return []

def match_items(gold_list, pred_list):
    """
    One-to-one matching approach:
      - For each pred item p, if p is in gold_list => TP; remove from gold_list.
        Otherwise => FP.
      - Any leftover in gold_list => FN
    Returns: (tp_list, fp_list, fn_list)
    """
    gold_pool = gold_list[:]
    tp_list = []
    fp_list = []

    for p in pred_list:
        if p in gold_pool:
            tp_list.append(p)
            gold_pool.remove(p)
        else:
            fp_list.append(p)

    fn_list = gold_pool
    return tp_list, fp_list, fn_list

def precision(tp, fp):
    return 0.0 if (tp + fp) == 0 else tp / (tp + fp)

def recall(tp, fn):
    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)

def f1_score(p, r):
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

# --------------------------------------------------------------------------
# Evaluate only val.json (ignore test.json)
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='data/text2spotasoc/relation/GWAS',
                        help='Folder containing val.json')
    parser.add_argument('--model_path',
                        required=True,
                        help='Path to your fine-tuned T5 UIE model')
    parser.add_argument('--schema_file',
                        required=True,
                        help='Path to record.schema file')
    parser.add_argument('--max_source_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', default='uie_manual_eval',
                        help='Folder to write the FP/FN tables (CSV)')
    args = parser.parse_args()

    # 1) Build the predictor
    predictor = HuggingfacePredictor(
        model_path=args.model_path,
        schema_file=args.schema_file,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )

    # 2) Build SEL2Record
    map_config = MapConfig.load_from_yaml('config/offset_map/closest_offset_en.yaml')
    schema_dict = SEL2Record.load_schema_dict(os.path.dirname(args.schema_file))
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema='spotasoc',
        map_config=map_config,
    )

    # Prepare accumulators for FP/FN
    entity_fp_rows = []
    entity_fn_rows = []
    relation_fp_rows = []
    relation_fn_rows = []

    # Prepare counters
    ent_tp = ent_fp = ent_fn = 0
    rel_tp = rel_fp = rel_fn = 0

    # Evaluate only val.json
    val_file = os.path.join(args.data_dir, 'val.json')
    if not os.path.isfile(val_file):
        print(f"[ERROR] val.json not found at: {val_file}")
        return

    # ---- Evaluate val.json
    print(f"\n--- Evaluating on val.json from: {val_file} ---")
    gold_data = read_jsonl(val_file)
    text_list = [d['text'] for d in gold_data]
    token_list = [d['tokens'] for d in gold_data]

    batch_num = math.ceil(len(text_list) / args.batch_size)
    all_predictions = []

    # 1) Inference in batches
    for i in tqdm(range(batch_num), desc="Inference on val"):
        start = i * args.batch_size
        end = start + args.batch_size
        batch_texts = text_list[start:end]
        seq2seq_outs = predictor.predict(batch_texts)
        all_predictions.extend(seq2seq_outs)

    # 2) Convert seq2seq => record
    records = []
    for seq_pred, text, tokens in zip(all_predictions, text_list, token_list):
        r = sel2record.sel2record(pred=seq_pred, text=text, tokens=tokens)
        records.append(r)

    # 3) Compare gold vs. pred
    for doc_idx, (item, pred_record) in enumerate(zip(gold_data, records)):
        gold_entities = item.get("entity", [])
        gold_relations = item.get("relation", [])
        pred_entities = pred_record.get("entity", {})
        pred_relations = pred_record.get("relation", {})

        # Convert to sets/lists
        gold_ent_list = get_entity_set_gold(gold_entities)
        gold_rel_list = get_relation_set_gold(gold_relations)
        pred_ent_list = get_entity_set_pred(pred_entities)
        pred_rel_list = get_relation_set_pred(pred_relations)

        # Entities
        ent_tp_list, ent_fp_list, ent_fn_list = match_items(gold_ent_list, pred_ent_list)
        ent_tp += len(ent_tp_list)
        ent_fp += len(ent_fp_list)
        ent_fn += len(ent_fn_list)

        for (pred_type, pred_text) in ent_fp_list:
            entity_fp_rows.append({
                "doc_idx": doc_idx,
                "text": item["text"],
                "pred_entity_type": pred_type,
                "pred_entity_text": pred_text,
                "all_gold_entities": str(gold_ent_list),
            })
        for (gold_type, gold_text) in ent_fn_list:
            entity_fn_rows.append({
                "doc_idx": doc_idx,
                "text": item["text"],
                "gold_entity_type": gold_type,
                "gold_entity_text": gold_text,
                "all_pred_entities": str(pred_ent_list),
            })

        # Relations
        rel_tp_list, rel_fp_list, rel_fn_list = match_items(gold_rel_list, pred_rel_list)
        rel_tp += len(rel_tp_list)
        rel_fp += len(rel_fp_list)
        rel_fn += len(rel_fn_list)

        for (rtype, (a1t, a1x), (a2t, a2x)) in rel_fp_list:
            relation_fp_rows.append({
                "doc_idx": doc_idx,
                "text": item["text"],
                "pred_relation_type": rtype,
                "pred_arg1_type": a1t, "pred_arg1_text": a1x,
                "pred_arg2_type": a2t, "pred_arg2_text": a2x,
                "all_gold_relations": str(gold_rel_list),
            })
        for (rtype, (a1t, a1x), (a2t, a2x)) in rel_fn_list:
            relation_fn_rows.append({
                "doc_idx": doc_idx,
                "text": item["text"],
                "gold_relation_type": rtype,
                "gold_arg1_type": a1t, "gold_arg1_text": a1x,
                "gold_arg2_type": a2t, "gold_arg2_text": a2x,
                "all_pred_relations": str(pred_rel_list),
            })

    # 4) Compute metrics
    ent_p = precision(ent_tp, ent_fp)
    ent_r = recall(ent_tp, ent_fn)
    ent_f = f1_score(ent_p, ent_r)

    rel_p = precision(rel_tp, rel_fp)
    rel_r = recall(rel_tp, rel_fn)
    rel_f = f1_score(rel_p, rel_r)

    print("\n=== Validation-Set Metrics ===")
    print(f"Entities: TP={ent_tp}, FP={ent_fp}, FN={ent_fn} "
          f"=> Precision={ent_p:.4f}, Recall={ent_r:.4f}, F1={ent_f:.4f}")
    print(f"Relations: TP={rel_tp}, FP={rel_fp}, FN={rel_fn} "
          f"=> Precision={rel_p:.4f}, Recall={rel_r:.4f}, F1={rel_f:.4f}")

    # 5) Write out CSVs
    os.makedirs(args.output_dir, exist_ok=True)

    df_entity_fp = pd.DataFrame(entity_fp_rows, columns=[
        "doc_idx", "text",
        "pred_entity_type", "pred_entity_text",
        "all_gold_entities"
    ])
    df_entity_fn = pd.DataFrame(entity_fn_rows, columns=[
        "doc_idx", "text",
        "gold_entity_type", "gold_entity_text",
        "all_pred_entities"
    ])
    df_relation_fp = pd.DataFrame(relation_fp_rows, columns=[
        "doc_idx", "text",
        "pred_relation_type",
        "pred_arg1_type", "pred_arg1_text",
        "pred_arg2_type", "pred_arg2_text",
        "all_gold_relations"
    ])
    df_relation_fn = pd.DataFrame(relation_fn_rows, columns=[
        "doc_idx", "text",
        "gold_relation_type",
        "gold_arg1_type", "gold_arg1_text",
        "gold_arg2_type", "gold_arg2_text",
        "all_pred_relations"
    ])

    df_entity_fp.to_csv(os.path.join(args.output_dir, "entity_fp.csv"), index=False, encoding="utf-8")
    df_entity_fn.to_csv(os.path.join(args.output_dir, "entity_fn.csv"), index=False, encoding="utf-8")
    df_relation_fp.to_csv(os.path.join(args.output_dir, "relation_fp.csv"), index=False, encoding="utf-8")
    df_relation_fn.to_csv(os.path.join(args.output_dir, "relation_fn.csv"), index=False, encoding="utf-8")

    print(f"\n[INFO] CSVs written to {args.output_dir}/")

    # 6) Consistency checks for FN
    print("\n--- Consistency Checks ---")
    print(f"Entity FN in code: {ent_fn}")
    print(f"Rows in entity_fn.csv: {len(df_entity_fn)}")
    if ent_fn == len(df_entity_fn):
        print("[OK] The number of missed entities matches the number of rows in entity_fn.csv")
    else:
        print("[WARN] Mismatch: The code's ent_fn != the row count in entity_fn.csv.\n"
              "Possibility: duplicate lines or data issues in gold/pred sets.")


if __name__ == "__main__":
    main()
