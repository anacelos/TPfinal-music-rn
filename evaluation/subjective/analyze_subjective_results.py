#!/usr/bin/env python3
"""TP2 PICO-8: blinded listening study analysis; Python 3 standard library only.

Input: UTF-8 CSV, one row per participant/scenario (see --help).
No identifiers or individual ratings are written to the aggregate outputs.
The private mapping and response files must stay outside the public repository.
"""
import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SLOTS = tuple('ABCD')
MUSIC = ('real', 'markov', 'neural')
CLASSES = set(MUSIC) | {'attention'}
PROFILE = ('age_group', 'musical_experience', 'gaming_frequency', 'retro_familiarity', 'region', 'headphones')
REQUIRED = ('participant_id', 'external_id', *SLOTS)


def normalize_mapping(mapping):
    if not isinstance(mapping, dict):
        raise ValueError('mapping must be an object')
    m = {str(k).strip().lower(): str(v).strip().lower() for k, v in mapping.items()}
    slots = set('abcd')
    if set(m) == CLASSES and set(m.values()) == slots:
        m = {v: k for k, v in m.items()}
    if set(m) != slots or set(m.values()) != CLASSES:
        raise ValueError('mapping must be a bijection between A/B/C/D and real/markov/neural/attention')
    return {k.upper(): v for k, v in m.items()}


def load_key(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = d['scenarios']
    if not isinstance(rows, list) or len(rows) != 240:
        raise ValueError('Expected 240 scenarios in the definitive key')
    key, orders = {}, set()
    for r in rows:
        sid = r['external_id']
        order = r['collection_order_v2']
        if not isinstance(sid, str) or not sid or sid in key or type(order) is not int or order in orders:
            raise ValueError('Duplicate or invalid key identifier/order')
        m = normalize_mapping(r['mapping'])
        if m['A'] == 'attention' or m['D'] == 'attention':
            raise ValueError('Unexpected attention position in key')
        key[sid] = {'mapping': m, 'order': order, 'tier': r.get('context_tier')}
        orders.add(order)
    if orders != set(range(240)):
        raise ValueError('Invalid collection order in key')
    balance = Counter(tuple(key[s]['mapping'][p] for p in SLOTS) for s in key)
    if len(balance) != 12 or set(balance.values()) != {20}:
        raise ValueError('Key does not contain the expected 12 x 20 permutations')
    return key


def read_responses(path, key):
    rows, seen, owners = [], set(), {}
    profiles = defaultdict(dict)
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not set(REQUIRED) <= set(reader.fieldnames):
            raise ValueError('CSV must contain unique headers: ' + ', '.join(REQUIRED))
        for line, r in enumerate(reader, 2):
            if None in r or any(r.get(k) is None for k in REQUIRED):
                raise ValueError(f'Line {line}: malformed CSV row')
            pid, sid = (r[k].strip() for k in REQUIRED[:2])
            if not pid or sid not in key:
                raise ValueError(f'Line {line}: missing participant or unknown scenario')
            if (pid, sid) in seen:
                raise ValueError(f'Line {line}: duplicate participant/scenario; resolve export before analysis')
            seen.add((pid, sid))
            values = {}
            for slot in SLOTS:
                val = r[slot].strip()
                if val and val not in {'1','2','3','4','5'}:
                    raise ValueError(f'Line {line}: ratings must be integers 1–5 or empty')
                values[key[sid]['mapping'][slot]] = int(val) if val else None
            complete = all(v is not None for v in values.values())
            if complete:
                if sid in owners:
                    raise ValueError(f'Line {line}: scenario completed more than once; resolve export')
                owners[sid] = pid
            for name in PROFILE:
                val = (r.get(name) or '').strip()
                if val:
                    if name in profiles[pid] and profiles[pid][name] != val:
                        raise ValueError(f'Line {line}: inconsistent participant profile ({name})')
                    profiles[pid][name] = val
            rows.append({'pid': pid, 'sid': sid, 'order': key[sid]['order'], 'mapping': key[sid]['mapping'], 'scores': values, 'complete': complete})
    per_pid = Counter(r['pid'] for r in rows if r['complete'])
    if any(n > 2 for n in per_pid.values()):
        raise ValueError('A participant has more than two completed scenarios; resolve export')
    return rows, profiles


def describe(vals):
    return {'n': len(vals), 'mean': st.mean(vals) if vals else None,
            'median': st.median(vals) if vals else None,
            'sd_sample': st.stdev(vals) if len(vals)>1 else None,
            'counts_1_to_5': {str(i): vals.count(i) for i in range(1,6)}}


def exact_sign_p(pos, neg):
    n = pos + neg
    if not n:
        return None
    return min(1.0, 2 * sum(math.comb(n,i) for i in range(min(pos,neg)+1)) / 2**n)


def paired_tests(per_pid):
    output = []
    for a,b in combinations(MUSIC, 2):
        # Scores are integer; signs are based on exact integer sums, not float roundoff.
        sums = [sum(r['scores'][a]-r['scores'][b] for r in rr) for rr in per_pid.values()]
        diffs = [d/len(rr) for d,rr in zip(sums,per_pid.values())]
        pos,neg = sum(d>0 for d in sums), sum(d<0 for d in sums)
        output.append({'first':a,'second':b,'participants':len(sums),'positive':pos,'negative':neg,
                       'ties':len(sums)-pos-neg,'mean_difference':st.mean(diffs) if diffs else None,
                       'median_difference':st.median(diffs) if diffs else None,
                       'p_exact_sign':exact_sign_p(pos,neg),'p_holm':None})
    # Family of three pre-specified comparisons. Untestable all-tie comparisons
    # are treated as p=1 for the correction, but reported as not tested.
    ordered = sorted(output,key=lambda x: x['p_exact_sign'] if x['p_exact_sign'] is not None else 1.)
    running = 0.
    for i,x in enumerate(ordered):
        p = x['p_exact_sign'] if x['p_exact_sign'] is not None else 1.
        running = max(running,min(1.,(len(ordered)-i)*p))
        if x['p_exact_sign'] is not None:
            x['p_holm'] = running
    return output


def analyze(rows):
    per_pid = defaultdict(list)
    for r in rows:
        per_pid[r['pid']].append(r)
    desc = {}
    for c in MUSIC:
        vals = [r['scores'][c] for r in rows]
        by_person = [st.mean(r['scores'][c] for r in rr) for rr in per_pid.values()]
        desc[c] = {**describe(vals), 'participant_weighted_mean': st.mean(by_person) if by_person else None}
    balance = {c:{p:sum(r['mapping'][p]==c for r in rows) for p in SLOTS} for c in (*MUSIC,'attention')}
    return {'n_scenarios':len(rows),'n_participants':len(per_pid),
            'participants_by_scenario_count':dict(sorted(Counter(map(len,per_pid.values())).items())),
            'conditions':desc,'paired_comparisons':paired_tests(per_pid),'position_counts':balance}


def subgroups(rows, profiles, minimum=5):
    out = []
    for name in PROFILE:
        groups = defaultdict(list)
        for r in rows:
            val = profiles.get(r['pid'],{}).get(name)
            if val:
                groups[val].append(r)
        for val, rr in sorted(groups.items()):
            pids = {r['pid'] for r in rr}
            if len(pids)<minimum:
                continue
            per_pid = defaultdict(list)
            for r in rr:
                per_pid[r['pid']].append(r)
            out.append({'variable':name,'category':val,'participants':len(pids),'scenarios':len(rr),
                        **{c:st.mean(st.mean(r['scores'][c] for r in pr) for pr in per_pid.values()) for c in MUSIC}})
    return out


def summarize(rows, profiles):
    complete = [r for r in rows if r['complete']]
    passed = [r for r in complete if r['scores']['attention']==1]
    all_pids = {r['pid'] for r in rows}
    completed_pids = {r['pid'] for r in complete}
    passed_pids = {r['pid'] for r in passed}
    checks = Counter(r['scores']['attention'] for r in complete)
    return {'protocol':{'main':'completed scenarios with attention=1','sensitivity':'all completed scenarios',
                        'inference':'two-sided exact sign test of participant-level average paired differences; Holm adjustment for 3 comparisons per analysis',
                        'subgroups':'exploratory participant-weighted means; at least 5 distinct participants per category; no subgroup significance tests'},
            'flow':{'participants_in_export':len(all_pids),'rows_in_export':len(rows),
                    'incomplete_rows':len(rows)-len(complete),'participants_with_completed_scenario':len(completed_pids),
                    'participants_with_two_completed_scenarios':sum(n==2 for n in Counter(r['pid'] for r in complete).values()),
                    'completed_scenarios':len(complete),'attention_passed':len(passed),'attention_failed':len(complete)-len(passed),
                    'attention_pass_rate':len(passed)/len(complete) if complete else None,
                    'attention_counts_1_to_5':{str(i):checks[i] for i in range(1,6)},
                    'participants_retained':len(passed_pids),'participants_with_no_passed_scenario':len(completed_pids-passed_pids)},
            'main':analyze(passed),'sensitivity':analyze(complete),'exploratory_profiles':subgroups(passed,profiles)}


def write_outputs(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def csv_out(name, fields, rows):
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    desc = []
    tests = []
    for label in ('main','sensitivity'):
        for c,v in result[label]['conditions'].items():
            desc.append({'analysis':label,'condition':c,**{k:v[k] for k in ('n','mean','median','sd_sample','participant_weighted_mean')},
                         **{f'rating_{i}':v['counts_1_to_5'][str(i)] for i in range(1,6)}})
        tests.extend({'analysis':label,**t} for t in result[label]['paired_comparisons'])
    csv_out('descriptive.csv',list(desc[0]),desc)
    csv_out('paired_comparisons.csv',list(tests[0]),tests)
    csv_out('exploratory_profiles.csv',['variable','category','participants','scenarios',*MUSIC],result['exploratory_profiles'])
    print(json.dumps(result['flow'],ensure_ascii=False,indent=2))
    print('Aggregate output:',out)


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--key',required=True,help='Private scenario_key.json')
    p.add_argument('--responses',required=True,help='CSV: participant_id,external_id,A,B,C,D; optional: '+','.join(PROFILE))
    p.add_argument('--out',required=True,help='Private output directory for aggregate analysis')
    args=p.parse_args()
    try:
        key=load_key(args.key)
        rows,profiles=read_responses(args.responses,key)
        result=summarize(rows,profiles)
        write_outputs(result,args.out)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        p.exit(2,f'Analysis stopped: {exc}\n')

if __name__=='__main__':
    main()
