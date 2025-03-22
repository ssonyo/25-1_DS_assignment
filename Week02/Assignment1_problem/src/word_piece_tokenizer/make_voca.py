import math 
from collections import defaultdict, Counter
from pathlib import Path
from utils import whitespace_tokenize
from itertools import pairwise
from typing import Counter, DefaultDict, Set, List, Tuple

def get_corpus(file_path: str) -> List[str]:
    """파일에서 모든 텍스트 라인을 읽어 리스트로 반환합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def get_initial_vocab(corpus: List[str]) -> Counter[str]:
    """
    초기 어휘 구성:
    각 단어를 문자 단위로 분해한 후 단어 경계를 나타내는 </w>를 붙여 vocabulary를 구성합니다.
    예: "hello" -> "h e l l o </w>"
    """
    vocab: Counter[str] = Counter()
    for line in corpus:
        words = whitespace_tokenize(line)
        for word in words:
            tokens = list(word) + ["</w>"]
            token_seq = " ".join(tokens)
            vocab[token_seq] += 1
    return vocab

def get_pair_stats(vocab: Counter[str]) -> DefaultDict[Tuple[str, str], int]:
    """
    현재 vocabulary에서 각 단어(토큰 시퀀스) 내 인접한 토큰 쌍의 빈도를 계산합니다.
    """
    pairs: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    # TODO: 각 token_seq에서 인접한 토큰 쌍을 추출하여, 전체 vocabulary에서의 빈도를 계산하는 코드를 작성하세요.
    # 예: for token_seq, freq in vocab.items(): ... 
    for token_seq, freq in vocab.items():
        tokens = token_seq.split()
        for pair in pairwise(tokens):
            pairs[pair] += freq
        
    return pairs

def get_unigram_counts(vocab: Counter[str]) -> Counter[str]:
    """
    vocabulary 내의 모든 토큰(단일 토큰)의 빈도를 계산합니다.
    """
    unigram_counts: Counter[str] = Counter()
    for token_seq, freq in vocab.items():
        tokens = token_seq.split()
        for token in tokens:
            unigram_counts[token] += freq
    return unigram_counts

def merge_vocab(pair: Tuple[str, str], vocab: Counter[str]) -> Counter[str]:
    """
    주어진 pair(예: ("a", "b"))를 vocabulary의 모든 항목에서 병합합니다.
    "a b" 형태의 bigram을 "ab"로 치환합니다.
    """
    merged_vocab: Counter[str] = Counter()

    # TODO: 주어진 pair를 활용하여 vocab 내의 모든 token sequence에서 "a b"를 "ab"로 병합하는 코드를 작성하세요.
    for token_seq, freq in vocab.items():
        new_token_seq = token_seq.replace(" ".join(pair), "".join(pair))
        merged_vocab[new_token_seq] += freq

    return merged_vocab

def compute_likelihood_score(pair: Tuple[str, str], pair_freq: float, unigram_counts: Counter, total_tokens: int) -> float:
    """
    얼마나 정보적으로 의미있는 병합인가? likelihood를 얼마나 올릴수있는가? 측정.

    likelihood 점수 계산:
    - 관측 빈도: pair_freq
    - 기대 빈도: (freq(token1) * freq(token2)) / total_tokens
    점수는 observed * log(observed/expected)로 계산합니다.
    """
    # TODO: likelihood 점수를 계산하는 코드를 작성하세요.
    token1, token2 = pair
    obs = pair_freq
    exp = (unigram_counts[token1] * unigram_counts[token2] / total_tokens)

    likelihood = obs * math.log(obs/exp)
    return likelihood

def learn_wordpiece_vocab(file_path, num_merges=1000, target_vocab_size=1000) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """
    test.txt 파일을 기반으로 likelihood 기반의 점수를 사용해 vocabulary를 학습합니다.
    
    num_merges: 최대 병합 횟수
    target_vocab_size: 원하는 최종 vocabulary 크기 (특수 토큰 포함)
    """
    corpus = get_corpus(file_path)
    vocab: Counter[str] = get_initial_vocab(corpus)
    merges: List[Tuple[str, str]] = []

    for i in range(num_merges):
        pair_stats = get_pair_stats(vocab)
        unigram_counts = get_unigram_counts(vocab)
        total_tokens = sum(unigram_counts.values())
        
        # 각 후보 쌍에 대해 likelihood 점수를 계산하는 부분
        scores = {}

        # TODO: pair_stats를 순회하면서 각 pair에 대한 likelihood 점수를 계산하고 scores 딕셔너리에 저장하는 코드를 작성하세요.
        for pair, freq in pair_stats.items():
            scores[pair] = compute_likelihood_score(pair, freq, unigram_counts, total_tokens)

        if not scores:
            break
        
        # TODO: scores 딕셔너리에서 가장 높은 점수를 가진 pair(best_pair)와 그 점수(best_score)를 결정하는 코드를 작성하세요.
        best_pair, best_score = max(scores.items(), key= lambda x: x[1])
        
        # 점수가 음수이거나 변화가 없으면 종료
        if best_score <= 0:
            break
        
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)
        
        # 현재 vocabulary 크기 확인 (특수 토큰 제외)
        current_vocab = set()
        for token_seq in vocab.keys():
            tokens = token_seq.split()
            for token in tokens:
                token = token.replace("</w>", "")
                current_vocab.add(token)
        if len(current_vocab) >= target_vocab_size:
            break

    return current_vocab, merges

def save_vocab(vocab_set, output_path="vocab.txt"):
    """
    생성된 vocabulary를 파일로 저장합니다.
    특수 토큰([UNK], [CLS], [SEP])을 우선 추가합니다.
    """
    special_tokens = ["[UNK]", "[CLS]", "[SEP]"]
    with open(output_path, "w", encoding="utf-8") as f:
        for token in special_tokens:
            f.write(token + "\n")
        for token in sorted(vocab_set):
            f.write(token + "\n")

if __name__ == "__main__":
    # test.txt 파일을 기반으로 vocabulary 생성
    file_path = Path(__file__).resolve().parent.parent.parent / "tests" / "tests.txt"
    vocab_set, merges = learn_wordpiece_vocab(str(file_path), num_merges=1000, target_vocab_size=1000)
    save_vocab(vocab_set, output_path="src/word_piece_tokenizer/vocab.txt")
    print("Vocabulary 생성 완료. 총 토큰 수:", len(vocab_set))
    print("병합 기록 (일부):", merges[:10])
