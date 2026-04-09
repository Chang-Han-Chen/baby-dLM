"""
block_utils.py — BD3-LM attention masks and basic block helpers.

Used by backbone.py for constructing the dual-stream training mask
and the block-causal sampling mask.
"""

import torch


# ------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------


def validate_block_len(block_size: int, block_len: int) -> None:
    if block_len <= 0:
        raise ValueError(f"block_len must be positive, got {block_len}")
    if block_size % block_len != 0:
        raise ValueError(
            f"block_len={block_len} must divide block_size={block_size}"
        )


def num_blocks(block_size: int, block_len: int) -> int:
    validate_block_len(block_size, block_len)
    return block_size // block_len


# ------------------------------------------------------------------
# BD3-LM masks
# ------------------------------------------------------------------


def make_bd3_train_mask(seq_len: int, block_len: int, device=None) -> torch.Tensor:
    """
    Full 2L x 2L BD3-LM training mask for x_t ⊕ x_0.

    This implements the paper/repo mask

        M_full = [[M_BD, M_OBC],
                  [   0,  M_BC]]

    with
      - M_BD  : block-diagonal attention within noisy blocks x_t
      - M_OBC : attention from noisy block b to clean blocks < b
      - M_BC  : block-causal attention on the clean x_0 stream

    Returned mask is boolean and shaped (1, 1, 2L, 2L), ready for SDPA.
    True means "allowed to attend".
    """
    validate_block_len(seq_len, block_len)
    idx = torch.arange(2 * seq_len, device=device)
    q_idx = idx[:, None]
    k_idx = idx[None, :]

    q_is_x0 = q_idx >= seq_len
    k_is_x0 = k_idx >= seq_len

    q_block = torch.where(q_is_x0, (q_idx - seq_len) // block_len, q_idx // block_len)
    k_block = torch.where(k_is_x0, (k_idx - seq_len) // block_len, k_idx // block_len)

    m_bd = (q_block == k_block) & (~q_is_x0) & (~k_is_x0)
    m_obc = (~q_is_x0) & k_is_x0 & (k_block < q_block)
    m_bc = q_is_x0 & k_is_x0 & (k_block <= q_block)

    mask = m_bd | m_obc | m_bc
    return mask[None, None, :, :]


def make_block_causal_mask(seq_len: int, block_len: int, device=None) -> torch.Tensor:
    """
    One-stream block-causal mask used at sampling time.

    Token i may attend to token j iff block(j) <= block(i), i.e.
    previous blocks are fully visible and the current block is
    bidirectional within itself.
    """
    validate_block_len(seq_len, block_len)
    pos = torch.arange(seq_len, device=device)
    q_block = pos[:, None] // block_len
    k_block = pos[None, :] // block_len
    mask = k_block <= q_block
    return mask[None, None, :, :]


# ------------------------------------------------------------------
# Agreement checks (used by tests)
# ------------------------------------------------------------------


def block_causal_equals_causal_when_block_len_is_one(seq_len: int, device=None) -> bool:
    mask = make_block_causal_mask(seq_len, 1, device=device)[0, 0]
    tri = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return bool(torch.equal(mask, tri))


def bd3_train_mask_special_cases_ok(seq_len: int, block_len: int, device=None) -> bool:
    mask = make_bd3_train_mask(seq_len, block_len, device=device)[0, 0]
    xt_to_xt = mask[:seq_len, :seq_len]
    xt_to_x0 = mask[:seq_len, seq_len:]
    x0_to_xt = mask[seq_len:, :seq_len]
    x0_to_x0 = mask[seq_len:, seq_len:]

    if block_len == seq_len:
        cond_1 = bool(xt_to_xt.all())
        cond_2 = bool((~xt_to_x0).all())
        cond_3 = bool((~x0_to_xt).all())
        cond_4 = bool(x0_to_x0.all())
        return cond_1 and cond_2 and cond_3 and cond_4

    if block_len == 1:
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        tri = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        strict_lower = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
        return (
            bool(torch.equal(xt_to_xt, eye))
            and bool(torch.equal(xt_to_x0, strict_lower))
            and bool((~x0_to_xt).all())
            and bool(torch.equal(x0_to_x0, tri))
        )

    return True
