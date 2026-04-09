import torch

from block_utils import (
    bd3_train_mask_special_cases_ok,
    block_causal_equals_causal_when_block_len_is_one,
    make_bd3_train_mask,
    make_block_causal_mask,
)


def upstream_block_diff_mask(seq_len: int, block_len: int) -> torch.Tensor:
    """Reference the mask logic used in kuleshov-group/bd3lms/models/dit.py."""
    idx = torch.arange(2 * seq_len)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]

    q_is_x0 = q_idx >= seq_len
    kv_is_x0 = kv_idx >= seq_len

    q_block = torch.where(q_is_x0, (q_idx - seq_len) // block_len, q_idx // block_len)
    kv_block = torch.where(kv_is_x0, (kv_idx - seq_len) // block_len, kv_idx // block_len)

    block_diagonal = (q_block == kv_block) & (q_is_x0 == kv_is_x0)
    offset_block_causal = (q_block > kv_block) & kv_is_x0 & (~q_is_x0)
    block_causal = (q_block >= kv_block) & kv_is_x0 & q_is_x0
    return block_diagonal | offset_block_causal | block_causal


def test_block_causal_len_one_is_standard_causal():
    assert block_causal_equals_causal_when_block_len_is_one(8)


def test_bd3_special_cases():
    assert bd3_train_mask_special_cases_ok(seq_len=8, block_len=1)
    assert bd3_train_mask_special_cases_ok(seq_len=8, block_len=8)


def test_training_mask_matches_upstream_formula():
    for seq_len, block_len in [(8, 1), (8, 2), (8, 4), (8, 8), (12, 3)]:
        local = make_bd3_train_mask(seq_len, block_len)[0, 0]
        upstream = upstream_block_diff_mask(seq_len, block_len)
        assert torch.equal(local, upstream)


def test_sampling_mask_matches_x0_stream_restriction():
    seq_len = 12
    block_len = 3
    train_mask = make_bd3_train_mask(seq_len, block_len)[0, 0]
    sample_mask = make_block_causal_mask(seq_len, block_len)[0, 0]
    assert torch.equal(train_mask[seq_len:, seq_len:], sample_mask)


def test_training_mask_structure():
    seq_len = 8
    block_len = 2
    mask = make_bd3_train_mask(seq_len, block_len)[0, 0]

    # noisy token 0 can only see noisy block 0 and no clean prefix
    assert mask[0, 0]
    assert mask[0, 1]
    assert not mask[0, 2]
    assert not mask[0, seq_len + 0]

    # noisy token in block 2 can see clean blocks 0 and 1, but not block 2 clean side
    q = 4
    assert mask[q, seq_len + 0]
    assert mask[q, seq_len + 3]
    assert not mask[q, seq_len + 4]

    # clean x0 stream is block-causal
    assert mask[seq_len + 4, seq_len + 0]
    assert mask[seq_len + 4, seq_len + 4]
    assert not mask[seq_len + 4, seq_len + 6]
