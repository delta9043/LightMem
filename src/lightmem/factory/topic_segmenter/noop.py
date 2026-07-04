from typing import Dict, List, Optional


class _WhitespaceTokenizer:
    """버퍼 토큰 카운트용 더미 토크나이저.

    정확도는 불필요하다 — buffer_len을 매우 크게 두어 토큰 한도에 의한 자동 컷이
    일어나지 않게 하므로, encode 길이는 실제로 컷 판단에 쓰이지 않는다.
    """

    def encode(self, text: str) -> List[str]:
        return text.split()


class NoOpSegmenter:
    """Topic segmentation 무력화용 segmenter.

    propose_cut이 항상 빈 리스트를 반환한다. SenMemBufferManager.cut_with_segmenter는
    coarse boundary가 비면 버퍼 전체를 세그먼트 1개로 반환하고 fine(유사도) 단계도
    건너뛰므로, 외부에서 force_segment로 강제 컷을 줄 때마다 '주어진 버퍼 = 1 세그먼트'가
    된다. 외부(MemoryAgnostic)가 미리 만든 chunk 경계를 메모리 단위로 보존하면서
    LightMem 자체 청킹만 끄기 위해 사용한다.
    """

    def __init__(self, config: Optional[Dict] = None, shared: bool = False, compressor=None):
        cfg = config or {}
        # 토큰 한도 자동 컷 방지 → force_segment에서만 컷
        self.buffer_len = cfg.get("buffer_len", 10 ** 9)
        self.tokenizer = _WhitespaceTokenizer()

    def propose_cut(self, buffer_texts: List[str]) -> List[int]:
        return []
