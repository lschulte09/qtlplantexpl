#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import math
import os
import re
from tqdm import tqdm

import transformers as huggingface_transformers

from uie.extraction.record_schema import RecordSchema
from uie.sel2record.record import MapConfig
from uie.extraction.scorer import EntityScorer, RelationScorer, EventScorer
from uie.sel2record.sel2record import SEL2Record

# --------------------------------------------------------------------------
# Helper regex / sets from your original inference script
split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
special_to_remove = {'<pad>', '</s>'}

def post_processing(x):
    for special in special_to_remove:
        x = x.replace(special, '')
    return x.strip()
# --------------------------------------------------------------------------

def schema_to_ssi(schema: RecordSchema):
    """Convert RecordSchema to a special sequence of tokens that prepends each example."""
    ssi = "<spot> " + "<spot> ".join(sorted(schema.type_list))
    ssi += "<asoc> " + "<asoc> ".join(sorted(schema.role_list))
    ssi += "<extra_id_2> "
    return ssi

# --------------------------------------------------------------------------
# HuggingfacePredictor class from your inference script
# Just re-used here for convenience
# --------------------------------------------------------------------------
class HuggingfacePredictor:
    def __init__(self, model_path, schema_file, max_source_length=256, max_target_length=192):
        self._tokenizer = huggingface_transformers.T5TokenizerFast.from_pretrained(model_path)
        self._model = huggingface_transformers.T5ForConditionalGeneration.from_pretrained(model_path)
        self._model.cuda()

        self._schema = RecordSchema.read_from_file(schema_file)
        self._ssi = schema_to_ssi(self._schema)

        self._max_source_length = max_source_length
        self._max_target_length = max_target_length

    def predict(self, text_list):
        """Run T5 model inference on a list of text strings."""
        # Prepend the schema special sequence (ssi)
        text_list = [self._ssi + t for t in text_list]

        # Tokenize
        inputs = self._tokenizer(text_list, padding=True, return_tensors='pt')
        inputs = inputs.to(self._model.device)

        # Truncate to max_source_length
        inputs['input_ids'] = inputs['input_ids'][:, :self._max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self._max_source_length]

        # Generate predictions
        outputs = self._model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self._max_target_length,
        )
        # Decode
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # Clean out special tokens
        decoded = [post_processing(d) for d in decoded]
        return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', required=True,
                        help="Path to the text file where each line is one sentence.")
    parser.add_argument('--output_dir', '-o', default='inference_results',
                        help="Directory to save output predictions.")
    parser.add_argument('--model', '-m', required=True,
                        help="Path to your fine-tuned UIE T5 model (e.g. output/Model_V3).")
    parser.add_argument('--schema_file', '-s', required=True,
                        help="Path to `record.schema` file, typically in your data folder.")
    parser.add_argument('--max_source_length', default=512, type=int)
    parser.add_argument('--max_target_length', default=512, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('-c', '--config', dest='map_config',
                        help='Offset Re-mapping Config',
                        default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc',
                        help="Decoding schema type (spotasoc, etc.).")
    # Whether to print intermediate details
    parser.add_argument('--verbose', action='store_true')
    # 'match_mode' is used in your original script for matching logic
    parser.add_argument('--match_mode', default='normal', 
                        choices=['set', 'normal', 'multimatch'])
    args = parser.parse_args()

    # -----------------------------------------------------------
    # 1) Prepare output directory
    # -----------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------
    # 2) Load the model & schema
    # -----------------------------------------------------------
    predictor = HuggingfacePredictor(
        model_path=args.model,
        schema_file=args.schema_file,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )

    # -----------------------------------------------------------
    # 3) Load SEL2Record for converting text -> record
    # -----------------------------------------------------------
    map_config = MapConfig.load_from_yaml(args.map_config)
    schema_dict = SEL2Record.load_schema_dict(os.path.dirname(args.schema_file))
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=args.decoding,
        map_config=map_config
    )

    # -----------------------------------------------------------
    # 4) Read lines from input file
    # -----------------------------------------------------------
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # We'll create a naive "tokens" list for each line
    # Typically you might do something like predictor._tokenizer.tokenize
    # but that might not exactly match your training data.
    # For demonstration, let's do a simple approach:
    token_lists = [predictor._tokenizer.tokenize(line) for line in lines]

    # -----------------------------------------------------------
    # 5) Batch predict
    # -----------------------------------------------------------
    batch_size = args.batch_size
    batch_num = math.ceil(len(lines) / batch_size)

    all_seq2seq_preds = []
    for i in tqdm(range(batch_num), desc="Inference Batches"):
        start = i * batch_size
        end = start + batch_size
        batch_lines = lines[start:end]

        # Predict with the T5 model
        pred_seq2seq = predictor.predict(batch_lines)
        all_seq2seq_preds.extend(pred_seq2seq)

    # -----------------------------------------------------------
    # 6) Convert seq2seq -> record structures
    # -----------------------------------------------------------
    # We do the same "sel2record" step from your original code
    all_records = []
    for seq2seq_pred, text, tokens in tqdm(zip(all_seq2seq_preds, lines, token_lists),
                                        total=len(all_seq2seq_preds),
                                        desc="Converting seq2seq to records"):
    	record = sel2record.sel2record(pred=seq2seq_pred, text=text, tokens=tokens)
    	all_records.append(record)

    # -----------------------------------------------------------
    # 7) Write outputs
    #    - We’ll store final record JSON (entity & relation info) and
    #      the raw seq2seq predictions in two separate files
    # -----------------------------------------------------------
    # 7a) Write seq2seq predictions
    seq2seq_out_file = os.path.join(args.output_dir, 'preds_seq2seq.txt')
    with open(seq2seq_out_file, 'w', encoding='utf-8') as f_out:
        for pred in all_seq2seq_preds:
            f_out.write(pred + "\n")

    # 7b) Write the final record structures as JSON
    records_out_file = os.path.join(args.output_dir, 'preds_record.jsonl')
    with open(records_out_file, 'w', encoding='utf-8') as f_out:
        for rec in all_records:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[INFO] Finished inference!")
    print(f"[INFO] Seq2Seq predictions written to: {seq2seq_out_file}")
    print(f"[INFO] Record predictions written to: {records_out_file}")


if __name__ == "__main__":
    main()
