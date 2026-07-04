import json
import os
import time
from datetime import datetime

from chunker import init_components, process_sample, print_sample_result


DATA_PATH = "/data/delta9043/repos/A-mem/chunker_inputs/amem_locomo_chunk.json"
RESULTS_DIR = "/data/delta9043/repos/A-mem/chunker_outputs/amem_locomo_chunker_v1"


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
                                "role": m.get("role"),
                                "content": m.get("content", ""),
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

    print("\n결과 저장 완료:")
    print(f"  JSONL  : {jsonl_path}")
    print(f"  Summary: {summary_path}")


def main():
    print("A-Mem용 chunker 실행")
    print(f"DATA_PATH   : {DATA_PATH}")
    print(f"RESULTS_DIR : {RESULTS_DIR}")

    print("컴포넌트 초기화 중...")
    compressor, segmenter, embedder = init_components()

    print("데이터 로드 중...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = {}
    total_segments = 0
    total_turns_in_segs = 0
    start_time = time.time()

    for item in data:
        qid = item["question_id"]
        sessions = item["haystack_sessions"]

        t0 = time.time()
        all_segments = process_sample(item, compressor, segmenter, embedder)
        elapsed = time.time() - t0

        print_sample_result(qid, sessions, all_segments)
        print(f"  처리 시간: {elapsed:.2f}s")

        all_results[qid] = all_segments
        total_segments += len(all_segments)
        total_turns_in_segs += sum(len(seg) // 2 for seg in all_segments)

    total_time = time.time() - start_time
    avg_turns = total_turns_in_segs / total_segments if total_segments > 0 else 0

    stats = {
        "num_samples": len(data),
        "total_segments": total_segments,
        "avg_turns_per_segment": round(avg_turns, 2),
        "total_elapsed_sec": round(total_time, 2),
    }

    print("\n===== 요약 통계 =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    save_results(all_results, stats)


if __name__ == "__main__":
    main()