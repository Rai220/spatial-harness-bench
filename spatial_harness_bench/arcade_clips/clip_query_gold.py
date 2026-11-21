"""Reference archival query client. Input and output are newline-delimited JSON."""
import json


def exchange(message):
    print(json.dumps(message), flush=True)
    return json.loads(input())


def locate(query):
    n, left, right, k, zeros = query
    pages = {}
    levels = []

    def rank(level, i):
        if i == n:
            return n - zeros[level]
        page, bit = divmod(i, 128)
        key = (level, page)
        if key not in pages:
            prefix, bits = exchange(["page", level, page])
            pages[key] = prefix, int(bits, 16)
        prefix, mask = pages[key]
        return prefix + (mask & ((1 << bit) - 1)).bit_count()

    gain = 0
    for level, z in enumerate(zeros):
        a, b = rank(level, left), rank(level, right)
        ones = b - a
        high = k < ones
        levels.append((high, left, right))
        gain = 2 * gain + high
        if high:
            left, right = z + a, z + b
        else:
            k -= ones
            left, right = left - a, right - b
    pos = left + k
    for level in reversed(range(8)):
        high, lo, hi = levels[level]
        order = pos - zeros[level] if high else pos
        while lo < hi:
            mid = (lo + hi) // 2
            ones = rank(level, mid + 1)
            count = ones if high else mid + 1 - ones
            if count <= order:
                lo = mid + 1
            else:
                hi = mid
        pos = lo
    return ["answer", pos, gain]


if __name__ == "__main__":
    print(json.dumps(locate(json.loads(input()))), flush=True)
