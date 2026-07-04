"""Chunking-only pipeline for LoCoMo.

Replicates LightMem's pre-extraction pipeline (llmlingua-2 compression ->
sensory buffer -> topic segmentation) and dumps the resulting chunks for
three boundary modes:
  - attention: cut at segmenter (attention peak) boundaries only
  - cosine:    cut at embedding cosine-similarity boundaries only
  - combined:  original LightMem logic (cosine boundaries within +-3 of an
               attention boundary; falls back to all cosine boundaries)

Deviation from the original: on the final force_segment flush the remainder
is emitted as a chunk and the buffer fully cleared (the original overwrites
segments from the last turn and leaves stale buffer state), so that the
concatenation of all chunks always equals the full message stream.

Also provides the shared runner (build_arg_parser / run_dataset) used by
experiments/longmemeval/chunk_longmemeval.py.

Usage (Seraph):
  python chunk_locomo.py \
      --data ../../dataset/locomo10.json \
      --llmlingua /data/delta9043/models/llmlingua-2 \
      --embedder /data/delta9043/models/all-MiniLM-L6-v2
"""
import argparse
import json
import os
import time

import numpy as np

from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager

MODES = ["attention", "cosine", "combined"]


# ============ LoCoMo loading (copied from add_locomo.py to avoid its import side effects) ============

def parse_locomo_timestamp(timestamp_str):
    import datetime
    timestamp_str = timestamp_str.strip("()")
    try:
        dt = datetime.datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def extract_locomo_sessions(conversation_dict):
    speaker_a = conversation_dict.get('speaker_a', 'Speaker_A')
    speaker_b = conversation_dict.get('speaker_b', 'Speaker_B')

    session_nums = set()
    for key in conversation_dict.keys():
        if key.startswith('session_') and not key.endswith('_date_time'):
            try:
                session_nums.add(int(key.split('_')[1]))
            except Exception:
                continue

    sessions = []
    timestamps = []
    for num in sorted(session_nums):
        session_key = f'session_{num}'
        if session_key not in conversation_dict:
            continue
        timestamp = conversation_dict.get(f'{session_key}_date_time', '')

        messages = []
        for turn in conversation_dict[session_key]:
            speaker_name = turn['speaker']
            speaker_id = 'speaker_a' if speaker_name == speaker_a else 'speaker_b'
            content = turn['text']
            if 'blip_caption' in turn and turn['blip_caption']:
                content = f"{content} (image description: {turn['blip_caption']})"
            messages.append({"role": "user", "content": content,
                             "speaker_id": speaker_id, "speaker_name": speaker_name})
            messages.append({"role": "assistant", "content": "",
                             "speaker_id": speaker_id, "speaker_name": speaker_name})
        sessions.append(messages)
        timestamps.append(parse_locomo_timestamp(timestamp))

    return sessions, timestamps, speaker_a, speaker_b


# ============ Buffer manager with selectable boundary mode ============

class ChunkOnlyBufferManager(SenMemBufferManager):
    """SenMemBufferManager whose cut step uses attention / cosine / combined boundaries."""

    def __init__(self, mode, max_tokens, tokenizer, diagnostics):
        super().__init__(max_tokens=max_tokens, tokenizer=tokenizer)
        assert mode in MODES
        self.mode = mode
        self.diagnostics = diagnostics

    def _fine_boundaries(self, text_embedder):
        # identical to original: turn = user + " " + assistant, threshold 0.2 -> 0.5
        turns = []
        for i in range(0, len(self.buffer), 2):
            turns.append(self.buffer[i]["content"] + " " + self.buffer[i + 1]["content"])
        embeddings = np.vstack([np.array(text_embedder.embed(t), dtype=np.float32) for t in turns])

        fine = []
        threshold = 0.2
        while threshold <= 0.5 and not fine:
            for i in range(len(turns) - 1):
                if self._cosine_similarity(embeddings[i], embeddings[i + 1]) < threshold:
                    fine.append(i + 1)
            if not fine:
                threshold += 0.05
        return fine, threshold

    def cut_with_segmenter(self, segmenter, text_embedder, force_segment: bool = False):
        segments = []
        if not self.buffer:
            return segments

        record = {"mode": self.mode, "num_turns": len(self.buffer) // 2,
                  "force_segment": force_segment, "coarse": None, "fine": None,
                  "threshold": None, "adjusted": None, "fallback": False}

        coarse = None
        if self.mode in ("attention", "combined"):
            buffer_texts = [m["content"] for m in self.buffer if m["role"] == "user"]
            coarse = segmenter.propose_cut(buffer_texts)
            record["coarse"] = list(coarse)

        if self.mode == "attention":
            final = sorted(set(coarse))
        elif self.mode == "cosine":
            fine, threshold = self._fine_boundaries(text_embedder)
            record["fine"], record["threshold"] = list(fine), round(threshold, 2)
            final = sorted(set(fine))
        else:  # combined: original two-stage logic incl. whole-buffer early exits
            if not coarse:
                final = []
            else:
                fine, threshold = self._fine_boundaries(text_embedder)
                record["fine"], record["threshold"] = list(fine), round(threshold, 2)
                if not fine:
                    final = []
                else:
                    adjusted = [fb for fb in fine if any(abs(fb - cb) <= 3 for cb in coarse)]
                    record["adjusted"] = list(adjusted)
                    if not adjusted:
                        adjusted = fine
                        record["fallback"] = True
                    final = sorted(set(adjusted))

        record["final"] = list(final)
        self.diagnostics.append(record)

        if not final:
            segments.append(self.buffer.copy())
            self.buffer.clear()
            self.token_count = 0
            return segments

        start_idx = 0
        for boundary in final:
            end_idx = 2 * boundary
            segments.append(self.buffer[start_idx:end_idx])
            start_idx = end_idx

        if force_segment:
            # emit remainder and clear (original leaves stale buffer state here)
            segments.append(self.buffer[start_idx:])
            start_idx = len(self.buffer)

        if start_idx > 0:
            del self.buffer[:start_idx]
            self._recount_tokens()
        return segments


# ============ Model loading (heavy imports kept local) ============

def build_models(args):
    from lightmem.configs.pre_compressor.llmlingua_2 import LlmLingua2Config
    from lightmem.factory.pre_compressor.llmlingua_2 import LlmLingua2Compressor
    from lightmem.factory.topic_segmenter.llmlingua_2 import LlmLingua2Segmenter
    from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
    from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface

    compressor = LlmLingua2Compressor(config=LlmLingua2Config(
        llmlingua_config={
            "model_name": args.llmlingua,
            "device_map": args.device,
            "use_llmlingua2": True,
        },
        compress_config={"instruction": "", "rate": args.rate, "target_token": -1},
    ))
    # shared mode, same as precomp_topic_shared=True (buffer_len = max_position_embeddings)
    segmenter = LlmLingua2Segmenter(config={}, shared=True, compressor=compressor)
    embedder = TextEmbedderHuggingface(BaseTextEmbedderConfig(
        model=args.embedder,
        embedding_dims=384,
        model_kwargs={"device": args.device},
    ))
    return compressor, segmenter, embedder


# ============ Per-sample chunking ============

def build_compressed_stream(sessions, timestamps, compressor, tokenizer):
    """Compress the whole conversation once; returns list of turn pairs."""
    stream = []
    for s_idx, (session, ts) in enumerate(zip(sessions, timestamps)):
        while session and session[0]["role"] != "user":
            session.pop(0)
        for turn_idx in range(len(session) // 2):
            pair = session[turn_idx * 2: turn_idx * 2 + 2]
            if pair[0]["role"] != "user" or pair[1]["role"] != "assistant":
                continue
            for m in pair:
                m["time_stamp"] = ts
                m["session"] = s_idx + 1
            stream.append(compressor.compress(pair, tokenizer))
    return stream


def verify_integrity(stream, chunks, mode, sample_id):
    expected = [(m["role"], m["content"]) for pair in stream for m in pair]
    got = [(m["role"], m["content"]) for chunk in chunks for m in chunk]
    if expected != got:
        raise RuntimeError(
            f"[{sample_id}][{mode}] integrity check failed: "
            f"expected {len(expected)} messages, got {len(got)} (or order/content mismatch)")


def chunk_sample(sample_id, segmenter, embedder, stream):
    results, diagnostics = {}, {}
    for mode in MODES:
        diag = []
        mgr = ChunkOnlyBufferManager(mode, max_tokens=segmenter.buffer_len,
                                     tokenizer=segmenter.tokenizer, diagnostics=diag)
        chunks = []
        for pair in stream:
            # fresh dicts per mode: buffer managers hold references across cuts
            chunks.extend(mgr.add_messages([dict(m) for m in pair], segmenter, embedder))
        if mgr.buffer:
            chunks.extend(mgr.cut_with_segmenter(segmenter, embedder, force_segment=True))
        chunks = [c for c in chunks if c]

        verify_integrity(stream, chunks, mode, sample_id)
        results[mode] = chunks
        diagnostics[mode] = diag
    return results, diagnostics


# ============ Shared runner ============

def build_arg_parser(description, default_data):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data', default=default_data, required=default_data is None)
    parser.add_argument('--llmlingua', default='/data/delta9043/models/llmlingua-2')
    parser.add_argument('--embedder', default='/data/delta9043/models/all-MiniLM-L6-v2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rate', type=float, default=0.6, help='llmlingua-2 compression rate')
    parser.add_argument('--output-dir', default='./chunk_results')
    parser.add_argument('--limit', type=int, default=None, help='process only first N samples')
    return parser


def run_dataset(args, samples):
    """samples: list of (sample_id, sessions, timestamps)."""
    os.makedirs(args.output_dir, exist_ok=True)
    compressor, segmenter, embedder = build_models(args)
    print(f"Models ready (buffer_len={segmenter.buffer_len})")

    all_chunks = {mode: {} for mode in MODES}
    all_diag = {}

    for sample_id, sessions, timestamps in samples:
        t0 = time.time()
        stream = build_compressed_stream(sessions, timestamps, compressor, segmenter.tokenizer)
        t1 = time.time()
        results, diagnostics = chunk_sample(sample_id, segmenter, embedder, stream)
        t2 = time.time()

        all_diag[sample_id] = diagnostics
        counts = {}
        for mode in MODES:
            chunks = results[mode]
            all_chunks[mode][sample_id] = [
                {"chunk_id": i, "num_turns": len(c) // 2, "messages": c}
                for i, c in enumerate(chunks)
            ]
            counts[mode] = len(chunks)
        print(f"{sample_id}: turns={len(stream)}, chunks={counts} "
              f"(compress {t1 - t0:.1f}s, chunk {t2 - t1:.1f}s)")

    for mode in MODES:
        path = os.path.join(args.output_dir, f"chunks_{mode}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_chunks[mode], f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")

    diag_path = os.path.join(args.output_dir, "boundary_diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(all_diag, f, ensure_ascii=False, indent=2)
    print(f"Wrote {diag_path}")

    # summary
    print("\n=== Summary ===")
    for mode in MODES:
        n_chunks = sum(len(v) for v in all_chunks[mode].values())
        turns = [c["num_turns"] for v in all_chunks[mode].values() for c in v]
        avg = sum(turns) / len(turns) if turns else 0
        print(f"{mode:>9}: {n_chunks} chunks, avg {avg:.1f} turns/chunk")


# ============ Main (LoCoMo) ============

def main():
    args = build_arg_parser("LoCoMo chunking-only pipeline (3 boundary modes)",
                            default_data='../../dataset/locomo10.json').parse_args()

    data = json.load(open(args.data, "r"))
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} samples from {args.data}")

    samples = []
    for sample in data:
        sessions, timestamps, _, _ = extract_locomo_sessions(sample['conversation'])
        samples.append((sample['sample_id'], sessions, timestamps))

    run_dataset(args, samples)


if __name__ == "__main__":
    main()
