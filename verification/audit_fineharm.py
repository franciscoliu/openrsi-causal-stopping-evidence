import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unicodedata


SPLIT_PRIORITY = {"train": 0, "dev": 1, "test": 2}
REQUIRED_FIELDS = ("prompt", "sentences", "sentence_labels", "words", "word_labels")


def stable_hash(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(prompt):
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def type_name(value):
    if value is None:
        return "null"
    return type(value).__name__


def label_key(value):
    if type(value) is str:
        return "str:" + value
    if type(value) is int:
        return "int:" + str(value)
    if type(value) is float and math.isfinite(value):
        return "float:" + repr(value)
    if type(value) is bool:
        return "bool:" + str(value).lower()
    return type_name(value) + ":unsupported"


def parse_label_map(serialized):
    mapping = json.loads(serialized)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Label map must be a nonempty JSON object.")
    for raw_key, canonical_value in mapping.items():
        if not isinstance(raw_key, str) or not raw_key.startswith(("int:", "str:", "float:")):
            raise ValueError("Only explicitly named int:, str:, and float: label keys are accepted.")
        if type(canonical_value) is not int or canonical_value not in (0, 1):
            raise ValueError("Canonical labels must be integer 0 (benign) or 1 (harmful).")
    return mapping


def decode_label(value, mapping):
    if mapping is None:
        return None, "label_mapping_unconfirmed"
    if type(value) not in (int, float, str):
        return None, "unsupported_label_type"
    key = label_key(value)
    if key not in mapping:
        return None, "unmapped_label_value"
    return mapping[key], None


def align_words(sentence, words):
    if not isinstance(words, list):
        return None, "words_not_list"
    if not words:
        return None, "empty_word_list"
    spans = []
    cursor = 0
    for word in words:
        if not isinstance(word, str) or not word or not word.strip():
            return None, "invalid_word_string"
        start = sentence.find(word, cursor)
        if start < 0:
            return None, "word_not_found_in_order"
        if any(not character.isspace() for character in sentence[cursor:start]):
            return None, "uncovered_nonwhitespace_between_words"
        stop = start + len(word)
        spans.append((start, stop))
        cursor = stop
    if any(not character.isspace() for character in sentence[cursor:]):
        return None, "uncovered_nonwhitespace_after_words"
    return spans, None


@dataclass
class AlignedRow:
    prompt: str
    response: str
    sentence_labels: list
    sentence_spans: list
    word_spans: list

    @property
    def harmful(self):
        return any(label == 1 for _, _, label in self.word_spans)

    @property
    def annotation_hash(self):
        return stable_hash([self.sentence_labels, self.sentence_spans, self.word_spans])

    def internal_payload(self):
        return {
            "prompt": self.prompt,
            "canonical_response": self.response,
            "sentence_labels": self.sentence_labels,
            "sentence_spans": self.sentence_spans,
            "word_spans": self.word_spans,
            "harmful": self.harmful,
        }


def validate_row(row, label_mapping):
    reasons = set()
    if not isinstance(row, dict):
        return None, {"row_not_object"}
    missing = set(REQUIRED_FIELDS) - set(row)
    if missing:
        reasons.update("missing_field:" + field for field in missing)
        return None, reasons
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        reasons.add("invalid_prompt")
    sentences = row["sentences"]
    if not isinstance(sentences, list) or not sentences:
        reasons.add("invalid_sentences_container")
        return None, reasons
    for field in ("sentence_labels", "words", "word_labels"):
        if not isinstance(row[field], list):
            reasons.add(field + "_not_list")
        elif len(row[field]) != len(sentences):
            reasons.add(field + "_sentence_count_mismatch")
    if reasons.intersection({field + "_not_list" for field in ("sentence_labels", "words", "word_labels")}):
        return None, reasons
    sentence_labels = []
    sentence_spans = []
    word_spans = []
    response_offset = 0
    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, str) or not sentence.strip():
            reasons.add("invalid_sentence_string")
            continue
        sentence_spans.append((response_offset, response_offset + len(sentence)))
        if index >= min(len(row["sentence_labels"]), len(row["words"]), len(row["word_labels"])):
            response_offset += len(sentence) + 1
            continue
        sentence_label, label_error = decode_label(row["sentence_labels"][index], label_mapping)
        if label_error:
            reasons.add(label_error)
        sentence_labels.append(sentence_label)
        words = row["words"][index]
        raw_word_labels = row["word_labels"][index]
        aligned_spans, alignment_error = align_words(sentence, words)
        if alignment_error:
            reasons.add(alignment_error)
        if not isinstance(raw_word_labels, list):
            reasons.add("word_labels_not_nested_list")
            response_offset += len(sentence) + 1
            continue
        if isinstance(words, list) and len(words) != len(raw_word_labels):
            reasons.add("word_label_count_mismatch")
        normalized_word_labels = []
        for raw_label in raw_word_labels:
            normalized_label, word_label_error = decode_label(raw_label, label_mapping)
            normalized_word_labels.append(normalized_label)
            if word_label_error:
                reasons.add(word_label_error)
        if sentence_label is not None and all(label is not None for label in normalized_word_labels):
            if sentence_label != int(any(label == 1 for label in normalized_word_labels)):
                reasons.add("sentence_word_label_disagreement")
        if aligned_spans is not None and len(aligned_spans) == len(normalized_word_labels):
            word_spans.extend(
                (start + response_offset, stop + response_offset, label)
                for (start, stop), label in zip(aligned_spans, normalized_word_labels)
            )
        response_offset += len(sentence) + 1
    if reasons:
        return None, reasons
    return AlignedRow(prompt, "\n".join(sentences), sentence_labels, sentence_spans, word_spans), reasons


def summarize_structure(value, prefix, counter):
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        counter[prefix + ":depth=" + str(depth) + ":" + type_name(current)] += 1
        if isinstance(current, list):
            pending.extend((element, depth + 1) for element in current)


def inventory_labels(value, representations, raw_types):
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, list):
            pending.extend(current)
            continue
        raw_types[type_name(current)] += 1
        key = label_key(current)
        if len(key) > 80:
            key = type_name(current) + ":long_value_sha256=" + stable_hash(current)
        representations[key] += 1


def annotation_class(value, mapping, nested=False):
    if not isinstance(value, list) or not value:
        return "unknown"
    if nested:
        if any(not isinstance(group, list) or not group for group in value):
            return "unknown"
        values = [label for group in value for label in group]
    else:
        values = value
    labels = [decode_label(label, mapping)[0] for label in values]
    if any(label is None for label in labels):
        return "unknown"
    return "harmful" if any(label == 1 for label in labels) else "benign"


def load_rows(path, rows_key=None, rows_format="list"):
    with Path(path).open(encoding="utf-8") as source:
        outer = json.load(source)
    outer_schema = {"type": type_name(outer)}
    if isinstance(outer, dict):
        outer_schema["keys"] = sorted(outer)
        outer_schema["value_types"] = dict(Counter(type_name(value) for value in outer.values()))
    selected = outer
    if rows_key is not None:
        if not isinstance(outer, dict) or rows_key not in outer:
            raise ValueError("The requested rows key is not present in the outer object.")
        selected = outer[rows_key]
    if rows_format == "list" and isinstance(selected, list):
        return selected, outer_schema
    if rows_format == "mapping" and isinstance(selected, dict):
        outer_schema.pop("keys", None)
        outer_schema["mapping_key_count"] = len(selected)
        return list(selected.values()), outer_schema
    raise ValueError("Unsupported outer schema; explicitly choose --rows-key or --rows-format mapping if appropriate.")


def make_record(aligned, split, row_id, row_index, include_payload=False):
    prompt_group = stable_hash(normalize_prompt(aligned.prompt))
    record = {
        "split": split,
        "row_id": row_id,
        "row_index": row_index,
        "prompt_group": prompt_group,
        "pair_hash": stable_hash([normalize_prompt(aligned.prompt), aligned.response]),
        "annotation_hash": aligned.annotation_hash,
        "class": "harmful" if aligned.harmful else "benign",
        "response_characters": len(aligned.response),
    }
    if include_payload:
        record["payload"] = aligned.internal_payload()
    return record


def deduplicate_records(records):
    prompt_splits = defaultdict(set)
    for record in records:
        prompt_splits[record["prompt_group"]].add(record["split"])
    highest_splits = {
        group: max(splits, key=lambda split: SPLIT_PRIORITY[split]) for group, splits in prompt_splits.items()
    }
    rejected = {}
    pair_groups = defaultdict(list)
    for record in records:
        if record["split"] != highest_splits[record["prompt_group"]]:
            rejected[record["row_id"]] = "split_priority_excluded"
        else:
            pair_groups[(record["split"], record["pair_hash"])].append(record)
    retained = []
    for members in pair_groups.values():
        if len({member["annotation_hash"] for member in members}) > 1:
            rejected.update((member["row_id"], "conflicting_pair_annotations") for member in members)
            continue
        ordered = sorted(members, key=lambda member: member["row_index"])
        retained.append(ordered[0])
        rejected.update((member["row_id"], "identical_pair_duplicate") for member in ordered[1:])
    overlap = Counter()
    for splits in prompt_splits.values():
        if len(splits) > 1:
            overlap["+".join(sorted(splits, key=lambda split: SPLIT_PRIORITY[split]))] += 1
    retained.sort(key=lambda record: (SPLIT_PRIORITY[record["split"]], record["row_index"]))
    return retained, rejected, dict(overlap)


def audit_files(split_paths, label_mapping=None, label_evidence=None, rows_key=None, rows_format="list", include_dev=False):
    if label_mapping is not None and not label_evidence:
        raise ValueError("An explicit label map also requires a source-evidence description.")
    records = []
    summaries = {}
    all_statuses = []
    for split, path in split_paths.items():
        if split not in SPLIT_PRIORITY:
            raise ValueError("Unknown split name.")
        digest = file_hash(path)
        rows, outer_schema = load_rows(path, rows_key, rows_format)
        row_types = Counter()
        field_types = defaultdict(Counter)
        nesting = Counter()
        label_representations = defaultdict(Counter)
        label_rawtypes = defaultdict(Counter)
        structural_reasons = Counter()
        structurally_eligible_classes = Counter()
        raw_sentence_classes = Counter()
        raw_word_classes = Counter()
        structural_excluded = 0
        for row_index, row in enumerate(rows):
            row_id = stable_hash([split, digest, row_index])
            row_types[type_name(row)] += 1
            if isinstance(row, dict):
                raw_sentence_classes[annotation_class(row.get("sentence_labels"), label_mapping)] += 1
                raw_word_classes[annotation_class(row.get("word_labels"), label_mapping, nested=True)] += 1
                for field, value in row.items():
                    field_types[field][type_name(value)] += 1
                    if field in REQUIRED_FIELDS and field != "prompt":
                        summarize_structure(value, field, nesting)
                    if field in ("sentence_labels", "word_labels"):
                        inventory_labels(value, label_representations[field], label_rawtypes[field])
            else:
                raw_sentence_classes["unknown"] += 1
                raw_word_classes["unknown"] += 1
            aligned, reasons = validate_row(row, label_mapping)
            all_statuses.append({"row_id": row_id, "split": split, "structural_reasons": sorted(reasons)})
            if reasons:
                structural_excluded += 1
                structural_reasons.update(reasons)
                continue
            record = make_record(aligned, split, row_id, row_index, include_dev and split == "dev")
            records.append(record)
            structurally_eligible_classes[record["class"]] += 1
        summaries[split] = {
            "file_name": Path(path).name,
            "sha256": digest,
            "outer_schema": outer_schema,
            "raw_rows": len(rows),
            "row_types": dict(row_types),
            "field_types": {field: dict(counts) for field, counts in sorted(field_types.items())},
            "nested_type_counts": dict(sorted(nesting.items())),
            "label_representations": {field: dict(counts) for field, counts in label_representations.items()},
            "label_rawtypes": {field: dict(counts) for field, counts in label_rawtypes.items()},
            "raw_sentence_label_classes": dict(raw_sentence_classes),
            "raw_word_label_classes": dict(raw_word_classes),
            "raw_class_note": "Derived separately from supplied raw annotations and the caller's mapping, before eligibility; not token-based evaluation classes.",
            "structural_excluded_rows": structural_excluded,
            "structural_reason_counts": dict(sorted(structural_reasons.items())),
            "structurally_eligible_classes": dict(structurally_eligible_classes),
        }
        del rows
    retained, duplicate_rejections, overlap = deduplicate_records(records)
    retained_by_split = defaultdict(list)
    records_by_id = {record["row_id"]: record for record in records}
    dedup_reasons_by_split = defaultdict(Counter)
    for row_id, reason in duplicate_rejections.items():
        dedup_reasons_by_split[records_by_id[row_id]["split"]][reason] += 1
    for record in retained:
        retained_by_split[record["split"]].append(record)
    for split, summary in summaries.items():
        kept = retained_by_split[split]
        summary["deduplication_reason_counts"] = dict(dedup_reasons_by_split[split])
        summary["retained_rows"] = len(kept)
        summary["retained_classes"] = dict(Counter(record["class"] for record in kept))
        summary["retained_row_ids_sha256"] = stable_hash([record["row_id"] for record in kept])
    for status in all_statuses:
        status["deduplication_reason"] = duplicate_rejections.get(status["row_id"])
        status["retained_structurally"] = not status["structural_reasons"] and status["deduplication_reason"] is None
    report = {
        "audit_version": 1,
        "scope": "Structural and character-alignment audit only; no tokenizer, model inference, or final evaluation eligibility.",
        "canonical_labels": {"0": "benign", "1": "harmful"},
        "label_mapping": label_mapping,
        "label_evidence": label_evidence,
        "label_direction_verified_by_auditor": False,
        "label_evidence_note": "The caller must verify the supplied mapping against source evidence; the script does not verify its truth.",
        "supplied_splits": sorted(split_paths, key=lambda split: SPLIT_PRIORITY[split]),
        "all_splits_present": set(split_paths) == set(SPLIT_PRIORITY),
        "split_leakage_audit_complete": False,
        "deduplication_scope": "Supplied structurally eligible splits only; rerun deduplication after token-length/alignment exclusions.",
        "split_leakage_note": "Even with all splits supplied, token-level eligibility is not audited and near duplicates are not detected.",
        "cross_split_prompt_group_counts": overlap,
        "reason_count_note": "Structural reasons count affected rows independently, so their sum may exceed excluded rows.",
        "invalid_word_policy": "Reject empty or whitespace-only word strings as ambiguous annotations.",
        "splits": summaries,
    }
    return report, retained, all_statuses


def private_text_file(path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Audit FineHarm structure without exposing response text in the report.")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--rows-key")
    parser.add_argument("--rows-format", choices=("list", "mapping"), default="list")
    parser.add_argument("--label-map", help='Explicit raw-to-canonical map, e.g. {"str:0":0,"str:1":1}, only after verification.')
    parser.add_argument("--label-evidence", help="Source commit/path and verified raw-label meaning.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, help="Optional private JSONL with hashed IDs and status, never response text.")
    parser.add_argument("--eligible-dev-jsonl", type=Path, help="Optional private INTERNAL JSONL containing retained dev text and character spans.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    split_paths = {split: getattr(args, split) for split in SPLIT_PRIORITY if getattr(args, split) is not None}
    if not split_paths:
        parser.error("Supply at least one of --train, --dev, or --test.")
    if args.label_map is not None and not args.label_evidence:
        parser.error("--label-map requires --label-evidence.")
    if args.eligible_dev_jsonl and (args.dev is None or args.label_map is None):
        parser.error("--eligible-dev-jsonl requires --dev and an explicitly confirmed label map.")
    input_paths = {path.resolve() for path in split_paths.values()}
    output_paths = [path.resolve() for path in (args.output, args.row_manifest, args.eligible_dev_jsonl) if path is not None]
    if input_paths.intersection(output_paths) or len(output_paths) != len(set(output_paths)):
        parser.error("Input and output paths must be distinct.")
    try:
        label_mapping = parse_label_map(args.label_map) if args.label_map is not None else None
        report, retained, statuses = audit_files(
            split_paths, label_mapping, args.label_evidence, args.rows_key, args.rows_format, bool(args.eligible_dev_jsonl)
        )
    except (OSError, ValueError, TypeError) as error:
        print("Audit failed (" + type(error).__name__ + "); no response text is logged. Check file/schema and explicit options.", file=sys.stderr)
        return 2
    with private_text_file(args.output) as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    if args.row_manifest:
        with private_text_file(args.row_manifest) as output:
            for status in statuses:
                output.write(json.dumps(status, sort_keys=True) + "\n")
    if args.eligible_dev_jsonl:
        with private_text_file(args.eligible_dev_jsonl) as output:
            for record in retained:
                if record["split"] == "dev":
                    payload = {key: value for key, value in record.items() if key != "payload"}
                    payload.update(record["payload"])
                    payload["eligibility_scope"] = "structural_only_before_token_filters"
                    output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "scope": "structural_only",
        "report": str(args.output),
        "label_mapping_supplied": label_mapping is not None,
        "splits": {split: {key: summary[key] for key in ("raw_rows", "retained_rows", "retained_classes")} for split, summary in report["splits"].items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
