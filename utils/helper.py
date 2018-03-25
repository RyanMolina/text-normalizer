def csv_to_dict(file):
    """Convert csv to python dictionary."""
    d = {}
    with open(file, 'r') as f:
        rows = f.read().splitlines()
        for row in rows:
            k, v = row.split(',')
            d.setdefault(k, []).append(v)
    return d