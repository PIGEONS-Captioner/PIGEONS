"""CapRiCorn-1K evaluation and bundled annotations."""


def duration_bin(duration):
    for low,high in [(0,120),(120,300),(300,480),(480,601)]:
        if low<duration<=high: return f'({low},{high}]'
    return None
