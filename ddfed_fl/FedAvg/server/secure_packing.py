import math
from typing import Iterable, List, Optional, Sequence, Tuple


def pack_digits(digits: List[int], base: int) -> int:
    """Pack non-negative digits into one integer with little-endian base expansion."""
    value = 0
    multiplier = 1
    for digit in digits:
        value += int(digit) * multiplier
        multiplier *= int(base)
    return int(value)


def unpack_digits(value: int, base: int, count: int) -> List[int]:
    """Unpack little-endian base digits from one integer."""
    out = []
    x = int(value)
    for _ in range(int(count)):
        out.append(int(x % int(base)))
        x //= int(base)
    return out


def compute_safe_base(sum_weights: int, max_abs_digit: int, margin: int = 1) -> int:
    """
    Choose a carry-safe base B for weighted sums.

    Need B > max_j |sum_i w_i * x_i[j]|.
    """
    return max(2, int(sum_weights) * int(max_abs_digit) + int(margin))


def compute_max_pack_len_by_bits(base: int, max_plain_bits: int) -> int:
    """Upper bound on pack length from plaintext bit-size budget."""
    bits_per_digit = max(1.0, math.log2(max(2, int(base))))
    return max(1, int(int(max_plain_bits) // bits_per_digit))


def compute_max_pack_len_by_dlog(base: int, max_dlog: int) -> int:
    """Upper bound on pack length to keep packed integer <= max_dlog."""
    if int(max_dlog) < 1:
        return 1
    if int(base) <= 1:
        return 1
    val = 1
    count = 0
    while val <= int(max_dlog):
        count += 1
        if val > int(max_dlog) // int(base):
            break
        val *= int(base)
    return max(1, count)


def choose_block_len(
    requested_pack_size: int,
    remaining: int,
    base: int,
    max_plain_bits: Optional[int] = None,
    max_dlog: Optional[int] = None,
) -> int:
    """Pick effective pack length under safety constraints."""
    block_len = min(max(1, int(requested_pack_size)), int(remaining))
    if max_plain_bits is not None:
        block_len = min(block_len, compute_max_pack_len_by_bits(base, int(max_plain_bits)))
    if max_dlog is not None:
        block_len = min(block_len, compute_max_pack_len_by_dlog(base, int(max_dlog)))
    return max(1, int(block_len))


def estimate_block_qmax(vectors: Iterable, start: int, length: int) -> int:
    """Estimate max absolute digit in a block from a list of 1D tensors."""
    qmax = 0
    s = int(start)
    e = int(start + length)
    for vec in vectors:
        local = vec[s:e]
        if int(local.numel()) == 0:
            continue
        vmax = int(local.max().item())
        if vmax > qmax:
            qmax = vmax
    return int(qmax)


def compute_slot_bits(
    num_clients: int,
    quantization_scale: int,
    max_abs_update: Optional[float] = None,
    precision_bits: Optional[int] = None,
    packing_value_bits: int = 32,
) -> Tuple[int, int]:
    """
    Compute per-slot bit width for plaintext packing.

    slot_bits = value_bits + padding_bits + sign_shift_bits
    where sign_shift_bits is fixed to 2 in this implementation.
    """
    del quantization_scale, max_abs_update  # currently optional hooks for future adaptive sizing
    value_bits = int(precision_bits) if precision_bits is not None else int(packing_value_bits)
    padding_bits = int(math.ceil(math.sqrt(max(1, int(num_clients))))) + 1
    sign_shift_bits = 2
    slot_bits = int(value_bits + padding_bits + sign_shift_bits)
    return slot_bits, padding_bits


def encode_signed_to_slot(value: int, slot_bits: int, offset: int) -> int:
    """Shift signed integer into non-negative slot range [0, 2^slot_bits)."""
    encoded = int(value) + int(offset)
    upper = 1 << int(slot_bits)
    if encoded < 0 or encoded >= upper:
        raise OverflowError(
            f"Encoded value out of range: value={value}, encoded={encoded}, slot_bits={slot_bits}"
        )
    return int(encoded)


def decode_slot_to_signed(value: int, slot_bits: int, offset: int) -> int:
    """Recover signed integer from encoded slot value."""
    upper = 1 << int(slot_bits)
    val = int(value)
    if val < 0 or val >= upper:
        raise OverflowError(f"Slot value out of range before decode: value={value}, slot_bits={slot_bits}")
    return int(val - int(offset))


def pack_plaintexts(values: Sequence[int], slot_bits: int) -> int:
    """Pack slot values with fixed-width bit layout (little-endian slots)."""
    packed = 0
    shift = 0
    upper = 1 << int(slot_bits)
    for val in values:
        v = int(val)
        if v < 0 or v >= upper:
            raise OverflowError(f"Slot overflow during pack: value={v}, slot_bits={slot_bits}")
        packed |= v << shift
        shift += int(slot_bits)
    return int(packed)


def unpack_plaintext(packed_value: int, pack_size: int, slot_bits: int) -> List[int]:
    """Unpack fixed-width slots from packed integer (little-endian)."""
    out = []
    mask = (1 << int(slot_bits)) - 1
    x = int(packed_value)
    for _ in range(int(pack_size)):
        out.append(int(x & mask))
        x >>= int(slot_bits)
    return out


def pack_client_update_vector(
    q_vector,
    pack_size: int,
    slot_bits: int,
    offset: int,
) -> Tuple[List[int], bool]:
    """Encode signed quantized vector and pack into plaintext blocks."""
    vals = [int(v.item()) for v in q_vector]
    encoded = []
    overflow = False
    for q in vals:
        try:
            encoded.append(encode_signed_to_slot(q, slot_bits, offset))
        except OverflowError:
            overflow = True
            encoded.append(0)

    blocks = []
    idx = 0
    n = len(encoded)
    k = max(1, int(pack_size))
    while idx < n:
        chunk = encoded[idx: idx + k]
        if len(chunk) < k:
            chunk = chunk + [int(offset)] * (k - len(chunk))
        blocks.append(pack_plaintexts(chunk, slot_bits))
        idx += k
    return blocks, overflow


def unpack_aggregated_vector(
    packed_values: Sequence[int],
    original_length: int,
    pack_size: int,
    slot_bits: int,
    offset: int,
) -> List[int]:
    """Unpack aggregated encoded slots and decode signed values."""
    out = []
    for packed in packed_values:
        slots = unpack_plaintext(int(packed), int(pack_size), int(slot_bits))
        for val in slots:
            out.append(decode_slot_to_signed(val, slot_bits, offset))
            if len(out) >= int(original_length):
                return out
    return out[: int(original_length)]
