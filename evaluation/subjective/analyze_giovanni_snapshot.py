#!/usr/bin/env python3
"""Adapt the web app's ratings.csv to the prespecified analysis, without exporting individual records."""
import argparse
import csv
import tempfile
from pathlib import Path
from analyze_subjective_results import load_key, read_responses, summarize, write_outputs, PROFILE, SLOTS


def run(ratings, key_path, out):
    key = load_key(key_path)
    aliases = {**{p: p for p in PROFILE}, 'age_group': 'age_range', **{s: 'rating_'+s for s in SLOTS}}
    fields = ['participant_id', 'external_id', *SLOTS, *PROFILE]
    with tempfile.TemporaryDirectory() as tmp:
        adapted = Path(tmp) / 'adapted.csv'
        with Path(ratings).open(encoding='utf-8-sig', newline='') as src, adapted.open('w', encoding='utf-8', newline='') as dst:
            reader = csv.DictReader(src)
            required = {'participant_id','external_id','collection_order_v2', *aliases.values()}
            if not required <= set(reader.fieldnames or []):
                raise ValueError('Missing required columns in app export')
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            for i, row in enumerate(reader, 2):
                sid = row['external_id']
                if sid not in key or int(row['collection_order_v2']) != key[sid]['order']:
                    raise ValueError(f'Invalid scenario/order on row {i}')
                if row['age_range'] == 'under_18':
                    raise ValueError(f'Under-18 entry on row {i}; resolve eligibility before analysis')
                writer.writerow({f: row[aliases.get(f, f)] for f in fields})
        rows, profiles = read_responses(adapted, key)
        result = summarize(rows, profiles)
        write_outputs(result, out)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ratings', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run(a.ratings, a.key, a.out)
