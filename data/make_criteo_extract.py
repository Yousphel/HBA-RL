"""
Recreate the 200,000-row Criteo extract used in the paper (D4).

Selection rule: the FIRST N rows of Criteo's train.txt, in file order (no sampling, no
shuffling). Split by row order: first 80% = training rows, last 20% = evaluation rows.
The script prints the exact row count, the click rate of each split and the SHA-256 of
the extract, and writes data/criteo_extract.sha256 / criteo_extract.json.

    python data/make_criteo_extract.py --src /path/to/train.txt --out criteo
    python data/make_criteo_extract.py --check criteo        # verify an existing extract

Do NOT commit train.txt or the extract: check the Criteo licence before redistributing rows.
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def click_rate(labels):
    return 100.0 * sum(labels) / max(len(labels), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', help='Path to the original Criteo train.txt')
    ap.add_argument('--out', default='criteo', help='Output folder for the extract (default: criteo)')
    ap.add_argument('--rows', type=int, default=200_000)
    ap.add_argument('--check', metavar='FOLDER', help='Only verify FOLDER/train.txt against data/criteo_extract.sha256')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sha_file = os.path.join(here, 'criteo_extract.sha256')

    if args.check:
        path = os.path.join(args.check, 'train.txt')
        digest = sha256_file(path)
        print(f"{path}\n  sha256 {digest}")
        if os.path.exists(sha_file):
            expected = open(sha_file).read().split()[0]
            print('  MATCHES the recorded checksum' if digest == expected else f'  DIFFERENT from recorded {expected}')
            sys.exit(0 if digest == expected else 1)
        print('  no recorded checksum yet (run without --check first)')
        return

    if not args.src:
        ap.error('--src is required')
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, 'train.txt')
    labels = []
    with open(args.src, 'rb') as fin, open(dst, 'wb') as fout:
        for i, line in enumerate(fin):
            if i >= args.rows:
                break
            fout.write(line)
            labels.append(1 if line.split(b'\t', 1)[0] == b'1' else 0)
    n = len(labels)
    cut = int(n * 0.8)
    info = {
        'source_file': os.path.basename(args.src), 'rule': f'first {args.rows} rows of train.txt, file order',
        'rows': n, 'train_rows': cut, 'eval_rows': n - cut,
        'click_rate_train_pct': round(click_rate(labels[:cut]), 3),
        'click_rate_eval_pct': round(click_rate(labels[cut:]), 3),
        'sha256': sha256_file(dst),
    }
    print(json.dumps(info, indent=2))
    if n != args.rows:
        print(f"WARNING: only {n} rows available (wanted {args.rows}).")
    with open(sha_file, 'w') as f:
        f.write(f"{info['sha256']}  train.txt  ({n} rows)\n")
    with open(os.path.join(here, 'criteo_extract.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {dst}, {sha_file}")


if __name__ == '__main__':
    main()
