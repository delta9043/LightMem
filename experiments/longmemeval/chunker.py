import json
import os
import time
from datetime import datetime

from lightmem.configs.pre_compressor.base import PreCompressorConfig
from lightmem.configs.topic_segmenter.base import TopicSegmenterConfig
from lightmem.configs.text_embedder.base import TextEmbedderConfig
from lightmem.factory.pre_compressor.factory import PreCompressorFactory
from lightmem.factory.topic_segmenter.factory import TopicSegmenterFactory
from lightmem.factory.text_embedder.factory import TextEmbedderFactory
from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager

LLMLINGUA_MODEL_PATH = "/data/delta9043/models/llmlingua-2"
EMBEDDING_MODEL_PATH = "/data/delta9043/models/all-MiniLM-L6-v2"
DATA_PATH             = "/data/delta9043/datasets/longmemeval/longmemeval_s_cleaned.json"
RESULTS_DIR           = "/data/delta9043/repos/LightMem/results/chunking_only"


def compress_with_chunking(compressor, tokenizer, content, max_tokens, depth=0):
    """512 토큰 초과 메시지를 청크로 나눠 각각 압축 후 합친다.
    압축이 반복 실패하면 depth 한계에서 truncate로 fallback."""
    token_ids = tokenizer.encode(content)
    if len(token_ids) < max_tokens:
        return content

    # 압축이 계속 실패해 줄어들지 않으면 truncate
    if depth >= 3:
        return tokenizer.decode(token_ids[:max_tokens - 1], skip_special_tokens=True)

    chunks = []
    for i in range(0, len(token_ids), max_tokens - 1):
        chunk_ids = token_ids[i : i + max_tokens - 1]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunk_msg = [{"role": "user", "content": chunk_text}]
        compressed = compressor.compress(chunk_msg, tokenizer)
        chunks.append(compressed[0]["content"])

    combined = " ".join(chunks)

    if len(tokenizer.encode(combined)) >= max_tokens:
        combined = compress_with_chunking(compressor, tokenizer, combined, max_tokens, depth + 1)

    return combined


def init_components():
    pre_cfg = PreCompressorConfig(
        model_name="llmlingua-2",
        configs={"llmlingua_config": {
            "model_name": LLMLINGUA_MODEL_PATH,
            "device_map": "cuda",
            "use_llmlingua2": True,
        }}
    )
    compressor = PreCompressorFactory.from_config(pre_cfg)

    seg_cfg   = TopicSegmenterConfig(model_name="llmlingua-2")
    segmenter = TopicSegmenterFactory.from_config(seg_cfg, shared=True, compressor=compressor)

    emb_cfg = TextEmbedderConfig(
        model_name="huggingface",
        configs={
            "model": EMBEDDING_MODEL_PATH,
            "embedding_dims": 384,
            "model_kwargs": {"device": "cuda"},
        }
    )
    embedder = TextEmbedderFactory.from_config(emb_cfg)

    return compressor, segmenter, embedder


def process_sample(item, compressor, segmenter, embedder):
    sessions   = item["haystack_sessions"]
    timestamps = item["haystack_dates"]

    # buffer는 sample마다 새로 생성 (재사용 시 이전 샘플 데이터가 섞임)
    buffer = SenMemBufferManager(
        max_tokens=segmenter.buffer_len,
        tokenizer=segmenter.tokenizer
    )

    all_segments = []

    for sess_idx, (session, timestamp) in enumerate(zip(sessions, timestamps)):
        while session and session[0]["role"] != "user":
            session.pop(0)

        num_turns = len(session) // 2
        for turn_idx in range(num_turns):
            turn_messages = session[turn_idx*2 : turn_idx*2 + 2]

            if (len(turn_messages) < 2
                    or turn_messages[0]["role"] != "user"
                    or turn_messages[1]["role"] != "assistant"):
                continue

            for msg in turn_messages:
                msg["time_stamp"] = timestamp

            # lightmem.py와 동일하게 항상 compress 먼저 수행 (rate=0.8)
            turn_messages = compressor.compress(turn_messages, segmenter.tokenizer)
            # 압축 후에도 max_tokens 초과 시 청크 분할 압축
            user_msg = turn_messages[0]
            if len(segmenter.tokenizer.encode(user_msg["content"])) >= buffer.max_tokens:
                user_msg["content"] = compress_with_chunking(
                    compressor, segmenter.tokenizer, user_msg["content"], buffer.max_tokens
                )

            # is → 인덱스 비교로 수정
            is_last = (sess_idx == len(sessions) - 1 and turn_idx == num_turns - 1)

            segs = buffer.add_messages(turn_messages, segmenter, embedder)
            all_segments.extend(segs)

            if is_last and buffer.buffer:
                segs = buffer.cut_with_segmenter(segmenter, embedder, force_segment=True)
                all_segments.extend(segs)

    return all_segments


def print_sample_result(question_id, sessions, all_segments):
    print("=" * 60)
    print(f"question_id : {question_id}")
    print(f"sessions    : {len(sessions)}개  →  segments: {len(all_segments)}개")
    print()

    for seg_idx, seg in enumerate(all_segments):
        num_turns = len(seg) // 2
        print(f"  [Segment {seg_idx + 1}] ({num_turns} turns)")
        for msg in seg:
            role    = msg.get("role", "?")
            content = msg.get("content", "")[:80]
            print(f"    {role:<10}: {content}")
        print()


def save_results(all_results, stats):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jsonl_path = os.path.join(RESULTS_DIR, f"chunks_{ts}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for qid, segments in all_results.items():
            record = {
                "question_id": qid,
                "num_segments": len(segments),
                "segments": [
                    {
                        "num_turns": len(seg) // 2,
                        "messages": [
                            {
                                "role":       m.get("role"),
                                "content":    m.get("content", ""),
                                "time_stamp": m.get("time_stamp"),
                            }
                            for m in seg
                        ],
                    }
                    for seg in segments
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = os.path.join(RESULTS_DIR, f"summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료:")
    print(f"  JSONL  : {jsonl_path}")
    print(f"  Summary: {summary_path}")


def main():
    print("컴포넌트 초기화 중...")
    compressor, segmenter, embedder = init_components()

    print("데이터 로드 중...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # sanity check: 앞 N개 샘플만 처리. 전체 실행 시 아래 줄 제거
    # data = data[:1]

    all_results          = {}
    total_segments       = 0
    total_turns_in_segs  = 0
    start_time           = time.time()

    for item in data:
        qid      = item["question_id"]
        sessions = item["haystack_sessions"]

        t0           = time.time()
        all_segments = process_sample(item, compressor, segmenter, embedder)
        elapsed      = time.time() - t0

        print_sample_result(qid, sessions, all_segments)
        print(f"  처리 시간: {elapsed:.2f}s")

        all_results[qid]  = all_segments
        total_segments   += len(all_segments)
        total_turns_in_segs += sum(len(seg) // 2 for seg in all_segments)

    total_time = time.time() - start_time
    avg_turns  = total_turns_in_segs / total_segments if total_segments > 0 else 0

    stats = {
        "num_samples":            len(data),
        "total_segments":         total_segments,
        "avg_turns_per_segment":  round(avg_turns, 2),
        "total_elapsed_sec":      round(total_time, 2),
    }

    print("\n===== 요약 통계 =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    save_results(all_results, stats)


if __name__ == "__main__":
    main()
