"""Chunking-only pipeline for LongMemEval (see chunk_locomo.py for details).

Same three boundary modes (attention / cosine / combined); only the data
loading differs: LongMemEval items already provide alternating user/assistant
sessions in `haystack_sessions` with dates in `haystack_dates`.

Usage (Seraph):
  python chunk_longmemeval.py --data /path/to/longmemeval_s.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "locomo"))
from chunk_locomo import build_arg_parser, run_dataset


def main():
    args = build_arg_parser("LongMemEval chunking-only pipeline (3 boundary modes)",
                            default_data=None).parse_args()

    data = json.load(open(args.data, "r"))
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} samples from {args.data}")

    samples = []
    for item in data:
        # session/time_stamp annotation happens in build_compressed_stream
        samples.append((item["question_id"], item["haystack_sessions"], item["haystack_dates"]))

    run_dataset(args, samples)


if __name__ == "__main__":
    main()
