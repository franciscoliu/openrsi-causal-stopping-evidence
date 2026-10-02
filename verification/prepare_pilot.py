import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

from audit_fineharm import (
    AlignedRow, annotation_class, deduplicate_records, file_hash, load_rows,
    make_record, private_text_file, stable_hash, validate_row,
)


LABEL_MAP = {"int:0": 0, "int:1": 1}
MODEL_REVISION = "74e1479150e9029d6778993f00491108323bb6f8"
DATA_REVISION = "e20a087e2ec0c5550f2a53a508503bb9a6741752"
SAMPLE_SEED = "openrsi-fineharm-dev-plumbing-v1"
CONTEXT_LIMIT = 8192
RESPONSE_SUFFIX = "<|im_end|>\n"


def serialize_response(tokenizer, prompt, response):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prefix, str):
        raise ValueError("Chat template must return a string.")
    return prefix + response + RESPONSE_SUFFIX, len(prefix), len(prefix) + len(response)


def response_token_mask(text, response_start, response_stop, offsets, word_spans):
    if not 0 <= response_start < response_stop <= len(text):
        raise ValueError("Invalid response character boundaries.")
    response = text[response_start:response_stop]
    harmful_characters = bytearray(len(response))
    for start, stop, label in word_spans:
        if not 0 <= start < stop <= len(response) or label not in (0, 1):
            raise ValueError("Invalid annotated character span.")
        if label == 1:
            for position in range(start, stop):
                harmful_characters[position] = int(not response[position].isspace())
    harmful_prefix = [0]
    for value in harmful_characters:
        harmful_prefix.append(harmful_prefix[-1] + value)
    coverage = bytearray(len(text))
    response_positions = []
    harmful_mask = []
    reasons = set()
    previous_start = previous_stop = 0
    for token_position, span in enumerate(offsets):
        if not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(value) is not int for value in span):
            reasons.add("invalid_offset_pair")
            continue
        start, stop = span
        if not 0 <= start < stop <= len(text):
            reasons.add("offset_out_of_bounds_or_empty")
            continue
        if start < previous_start or stop < previous_stop:
            reasons.add("nonmonotone_offsets")
        previous_start, previous_stop = start, stop
        coverage[start:stop] = b"\x01" * (stop - start)
        if start < response_stop and stop > response_start:
            if start < response_start or stop > response_stop:
                reasons.add("token_crosses_response_template_boundary")
                continue
            response_positions.append(token_position)
            harmful_mask.append(harmful_prefix[stop - response_start] > harmful_prefix[start - response_start])
    if any(not covered and not character.isspace() for character, covered in zip(text, coverage)):
        reasons.add("uncovered_nonwhitespace_characters")
    if not response_positions:
        reasons.add("no_response_tokens")
    if response_positions and response_positions != list(range(response_positions[0], response_positions[-1] + 1)):
        reasons.add("noncontiguous_response_positions")
    return response_positions, harmful_mask, reasons


def tokenize_aligned(tokenizer, aligned):
    text, response_start, response_stop = serialize_response(tokenizer, aligned.prompt, aligned.response)
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    input_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    reasons = set()
    if len(input_ids) != len(offsets) or any(type(value) is not int for value in input_ids):
        return None, {"invalid_token_ids_or_offset_count"}
    if len(input_ids) > CONTEXT_LIMIT:
        reasons.add("formatted_input_over_8192_tokens")
    if tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        reasons.add("full_decode_roundtrip_mismatch")
    positions, harmful_mask, offset_reasons = response_token_mask(
        text, response_start, response_stop, offsets, aligned.word_spans,
    )
    reasons.update(offset_reasons)
    if aligned.harmful and not any(harmful_mask):
        reasons.add("harmful_row_without_marked_token")
    if not reasons:
        segments = (
            (input_ids[:positions[0]], text[:response_start]),
            ([input_ids[position] for position in positions], aligned.response),
            (input_ids[positions[-1] + 1:], text[response_stop:]),
        )
        if any(tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != expected
               for token_ids, expected in segments):
            reasons.add("decoded_segment_boundary_mismatch")
    if reasons:
        return None, reasons
    return {"input_ids": input_ids, "response_positions": positions, "harmful_mask": harmful_mask}, reasons


def check_serializer(tokenizer):
    prompt, response = "An ordinary question.", "An ordinary answer."
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    if serialize_response(tokenizer, prompt, response)[0] != expected:
        raise ValueError("Structured serializer differs from the ordinary full-chat template.")
    cases = [
        (prompt, response),
        ("Question\nline", "\n First line\n\n<think>\nReasoning.\n</think>\n\nLast line. \n"),
        ("Content <|im_end|> remains inside the prompt.", "Content <|im_start|> and <|im_end|> remains inside the response."),
        ("Unicode question: 你好", "你好🙂 café"),
    ]
    expected_boundary_exclusions = 0
    for prompt, response in cases:
        text, start, stop = serialize_response(tokenizer, prompt, response)
        if text[start:stop] != response:
            raise ValueError("Serializer changed canonical response text.")
        aligned = AlignedRow(prompt, response, [], [], [(0, len(response), 1)])
        payload, reasons = tokenize_aligned(tokenizer, aligned)
        if reasons and reasons <= {"token_crosses_response_template_boundary"}:
            expected_boundary_exclusions += 1
        elif reasons or not payload:
            raise ValueError("Synthetic serializer/tokenizer checks failed: " + ",".join(sorted(reasons)))
    return {"ordinary_template_equivalence": True, "verbatim_newline_think_preservation": True,
            "literal_delimiter_content": True, "unicode_offsets": True, "case_count": len(cases),
            "synthetic_boundary_exclusion_count": expected_boundary_exclusions}


def freeze_sample(records, sample_size):
    if sample_size < 2 or len(records) < sample_size:
        raise ValueError("Need at least two selected rows and the requested number of eligible development rows.")
    chosen = sorted(records, key=lambda record: (stable_hash([SAMPLE_SEED, record["row_id"]]), record["row_id"]))[:sample_size]
    causal_candidates = [record for record in chosen if len(record["model_input"]["response_positions"]) >= 2]
    if len(causal_candidates) < 2:
        raise ValueError("The frozen sample must contain two responses with at least two tokens; do not resample.")
    causal = sorted(causal_candidates, key=lambda record: (len(record["model_input"]["response_positions"]), record["row_id"]))[:2]
    return chosen, [record["row_id"] for record in causal]


def prepare(data_dir, model_dir, tokenizer, sample_size, label_evidence):
    if not label_evidence.strip():
        raise ValueError("Verified label-direction evidence is required.")
    started = time.monotonic()
    serializer_checks = check_serializer(tokenizer)
    records, summaries = [], {}
    for split, suffix in (("train", "train"), ("dev", "val"), ("test", "test")):
        path = Path(data_dir) / ("FineHarm-" + suffix + ".json")
        digest = file_hash(path)
        rows, outer_schema = load_rows(path)
        raw_sentence_classes, raw_word_classes = Counter(), Counter()
        structural_reasons, token_reasons = Counter(), Counter()
        structural_classes, token_classes = Counter(), Counter()
        structural_excluded = token_excluded = eligible_tokens = 0
        for row_index, row in enumerate(rows):
            if isinstance(row, dict):
                raw_sentence_classes[annotation_class(row.get("sentence_labels"), LABEL_MAP)] += 1
                raw_word_classes[annotation_class(row.get("word_labels"), LABEL_MAP, nested=True)] += 1
            else:
                raw_sentence_classes["unknown"] += 1
                raw_word_classes["unknown"] += 1
            aligned, reasons = validate_row(row, LABEL_MAP)
            if reasons:
                structural_reasons.update(reasons)
                structural_excluded += 1
                continue
            structural_classes["harmful" if aligned.harmful else "benign"] += 1
            model_input, reasons = tokenize_aligned(tokenizer, aligned)
            if reasons:
                token_reasons.update(reasons)
                token_excluded += 1
                continue
            row_id = stable_hash([split, digest, row_index])
            record = make_record(aligned, split, row_id, row_index)
            record["input_token_count"] = len(model_input["input_ids"])
            record["response_token_count"] = len(model_input["response_positions"])
            if split == "dev":
                record["model_input"] = model_input
            records.append(record)
            token_classes[record["class"]] += 1
            eligible_tokens += record["input_token_count"]
        summaries[split] = {
            "file_name": path.name, "sha256": digest, "outer_schema": outer_schema,
            "raw_rows": len(rows), "raw_sentence_label_classes": dict(raw_sentence_classes),
            "raw_word_label_classes": dict(raw_word_classes),
            "structural_excluded_rows": structural_excluded, "structural_reason_counts": dict(structural_reasons),
            "structurally_eligible_classes": dict(structural_classes),
            "token_excluded_rows": token_excluded, "token_reason_counts": dict(token_reasons),
            "token_eligible_classes_before_dedup": dict(token_classes),
            "eligible_input_tokens_before_dedup": eligible_tokens,
        }
        del rows
        print(json.dumps({"prepared_split": split, "raw_rows": summaries[split]["raw_rows"],
                          "structural_excluded": structural_excluded, "token_excluded": token_excluded}), flush=True)
    retained, rejected, overlap = deduplicate_records(records)
    record_splits = {record["row_id"]: record["split"] for record in records}
    dedup_reasons = defaultdict(Counter)
    for row_id, reason in rejected.items():
        dedup_reasons[record_splits[row_id]][reason] += 1
    for split, summary in summaries.items():
        kept = [record for record in retained if record["split"] == split]
        summary.update({
            "deduplication_reason_counts": dict(dedup_reasons[split]), "retained_rows": len(kept),
            "retained_classes": dict(Counter(record["class"] for record in kept)),
            "retained_input_tokens": sum(record["input_token_count"] for record in kept),
            "retained_response_tokens": sum(record["response_token_count"] for record in kept),
            "retained_row_ids_sha256": stable_hash([record["row_id"] for record in kept]),
        })
    if any(not all(summary["retained_classes"].get(label) for label in ("benign", "harmful")) for summary in summaries.values()):
        raise ValueError("Both retained classes are required in every supplied split.")
    chosen, causal_ids = freeze_sample([record for record in retained if record["split"] == "dev"], sample_size)
    selected = [{"row_id": record["row_id"], "class": record["class"], **record["model_input"]} for record in chosen]
    metadata = {
        "model_revision": MODEL_REVISION, "dataset_revision": DATA_REVISION,
        "sample_seed": SAMPLE_SEED, "sample_rule": "Lowest SHA256 of JSON [fixed_seed,row_id]; no guard outputs consulted.",
        "context_limit": CONTEXT_LIMIT, "sample_size": len(selected),
        "selected_row_ids_sha256": stable_hash([record["row_id"] for record in selected]),
        "causal_selection": "Two shortest response-token sequences with at least two tokens within the unchanged selected sample; ties by row_id.",
        "label_mapping": LABEL_MAP, "label_evidence_supplied_by_caller": label_evidence,
        "serializer": "Pinned user generation prefix + verbatim canonical response + <|im_end|>\\n.",
    }
    bundle = {"records": selected, "causal_row_ids": causal_ids, "tolerances": {"atol": 0.03, "rtol": 0.0}, "metadata": metadata}
    report = {
        "preparation_version": 1, "scope": "All-split structure/token eligibility and exact normalized-prompt deduplication; no guard inference or policy training.",
        "metadata": metadata, "serializer_checks": serializer_checks,
        "tokenizer_file_sha256": {path.name: file_hash(path) for path in sorted(Path(model_dir).glob("*"))
                                   if path.is_file() and path.name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")},
        "all_splits_present": True, "cross_split_prompt_group_counts_before_dedup": overlap,
        "deduplication_scope": "Only structurally and token-eligible rows, with test > dev > train priority.",
        "splits": summaries, "sample_classes": dict(Counter(record["class"] for record in selected)),
        "sample_input_tokens": sum(len(record["input_ids"]) for record in selected),
        "sample_response_tokens": sum(len(record["response_positions"]) for record in selected),
        "causal_response_tokens": [len(record["response_positions"]) for row_id in causal_ids for record in selected if record["row_id"] == row_id],
        "preparation_seconds": time.monotonic() - started,
        "limitations": ["Reason counts are independent and can overlap.", "Near duplicates and inherited contamination are not detected.",
                        "Eligibility filtering changes composition; this pilot does not establish task performance or population risk."],
    }
    return report, bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze a private CRC-only FineHarm development plumbing sample.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--label-evidence", required=True)
    args = parser.parse_args(argv)
    if args.sample_size < 2:
        parser.error("--sample-size must be at least 2.")
    try:
        from transformers import AutoTokenizer

        selected_path = args.output_dir / "selected_dev.json"
        report_path = args.output_dir / "preparation_report.json"
        if selected_path.exists() or report_path.exists():
            raise ValueError("Frozen outputs already exist; use a new explicitly chosen output directory.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, local_files_only=True, trust_remote_code=False)
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with character offsets is required.")
        report, bundle = prepare(args.data_dir, args.model_dir, tokenizer, args.sample_size, args.label_evidence)
        args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with private_text_file(selected_path) as destination:
            json.dump(bundle, destination, ensure_ascii=False, sort_keys=True)
            destination.write("\n")
        report["selected_dev_file_sha256"] = file_hash(selected_path)
        with private_text_file(report_path) as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2, sort_keys=True)
            destination.write("\n")
    except Exception as error:
        print("Preparation failed (" + type(error).__name__ + "); raw content is not logged.", file=sys.stderr)
        return 2
    print(json.dumps({"report": str(report_path), "sample_size": len(bundle["records"]),
                      "sample_classes": report["sample_classes"], "selected_dev_file_sha256": report["selected_dev_file_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
