"""Build the plain 0.958 fork + R7 at 0.4 of the compact slot (goodpjw2008's version-26 recipe).

R7's setup and run cells are copied verbatim from heliosli/0-957-latest-dataset-score-in-one-click
and run in their own namespace, with their own temp dir and output names, so they cannot touch R6's
globals or files. Any R7 failure, a late start, or under 90 % coverage leaves the blend exactly as
the plain fork.

usage: build_r7.py <out_dir> <slug> [--fork NB] [--a-r7 0.4] [--a224 0.35] [--w-raptor 0.5]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("slug")
ap.add_argument("--fork", default=r"D:\kaggle_c\fork958\stage_rsna-knee-fork-958-r6\rsna-knee-fork-958-r6.ipynb",
                help="base notebook: the plain fork, or a fork that already carries our arm")
ap.add_argument("--a-r7", type=float, default=0.4)
ap.add_argument("--a224", type=float, default=0.35, help="share of the Raptor slot given to the 224 model")
ap.add_argument("--w-raptor", type=float, default=0.5, help="Raptor weight for findings without a per-finding one")
args = ap.parse_args()
FORK = Path(args.fork)
R7NB = Path(r"D:\kaggle_c\pub_scan\r7\0-957-latest-dataset-score-in-one-click.ipynb")
out_dir, slug = Path(args.out_dir), args.slug

A_R7 = args.a_r7         # share of the compact slot given to R7 (goodpjw2008 version 26: 0.4)
R7_START_LIMIT_H = 5.5   # do not start R7 later than this after the notebook start
R7_DEADLINE_H = 7.0      # R7's workers stop by then, leaving 2 h for Part B and the 9 h limit
R7_MIN_COVER = 0.90      # below this R7 is dropped rather than mixed into part of the rows


def src(cell):
    return "".join(cell["source"])


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def sub1(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)


fork = json.loads(FORK.read_text(encoding="utf-8"))
r7nb = json.loads(R7NB.read_text(encoding="utf-8"))
cells = fork["cells"]

# Locate cells by content, not index.
i_r6run = next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
               and "OURS_STAGE_COMPLETE" in src(c))
i_blend = next(i for i, c in enumerate(cells) if c["cell_type"] == "code"
               and src(c).startswith("# Stage 3: per-finding rank blend"))
r7_setup = next(src(c) for c in r7nb["cells"] if src(c).startswith("ARM = 'E_lrprior'"))
r7_run = next(src(c) for c in r7nb["cells"] if "OURS_STAGE_COMPLETE" in src(c))
assert "BUNDLE_ID = '8c7b666e27dc0e8f'" in r7_setup and "rsna-knee-compact-r7cnxfsss" in r7_setup

# R7 setup: own temp dir, earlier deadline.
r7_setup = sub1(r7_setup, "DEADLINE_HOURS = 8.3", f"DEADLINE_HOURS = {R7_DEADLINE_H}   # [r7] was 8.3")
r7_setup = sub1(r7_setup, "TMP = TMP / 'probe'\n", "TMP = TMP / 'probe_r7'   # [r7] apart from R6's /probe\n")

# R7 run: own output names.
r7_run = r7_run.replace("WORK / 'ours_sub.csv'", "WORK / 'r7_sub.csv'")
assert "ours_sub" not in r7_run
r7_run = sub1(r7_run, "(WORK / 'submission_receipt.json')", "(WORK / 'r7_receipt.json')")
r7_run = sub1(r7_run, "WORK / (log.parent.name + '_log.txt')", "WORK / ('r7_' + log.parent.name + '_log.txt')")
r7_run = sub1(r7_run, "print('OURS_STAGE_COMPLETE'", "print('R7_STAGE_COMPLETE'")

driver = f'''# Part A2b [r7]: heliosli's R7 (bundle 8c7b666e, 4 folds, no flip TTA), its setup and run cells unchanged apart
# from the marked [r7] lines, run in a namespace of their own so R6's globals and files are untouched.
# Any failure, a start later than {R7_START_LIMIT_H} h, or under {R7_MIN_COVER:.0%} coverage leaves the blend as the plain fork.
import csv as _r7_csv, time as _r7_time, traceback as _r7_tb
R7_OK, R7_PRED = False, None
_r7_t = _r7_time.time()
if _r7_t - NB_T0 > {R7_START_LIMIT_H} * 3600:
    print('R7_SKIPPED started too late: %.2f h' % ((_r7_t - NB_T0) / 3600), flush=True)
else:
    _r7_ns = {{'__name__': '__r7__', 'NB_T0': NB_T0, 'display': display}}
    try:
        for _f in ('/tmp/r7_setup.py', '/tmp/r7_run.py'):
            exec(compile(open(_f, encoding='utf-8').read(), _f, 'exec'), _r7_ns)
        R7_PRED = _r7_ns['pred']
        R7_OK = (WORK / 'r7_sub.csv').is_file()
    except BaseException as _e:   # SystemExit included: nothing in R7 may stop the notebook
        _r7_tb.print_exc()
        print('R7_FAILED', repr(_e)[:300], flush=True)
print('R7_STAGE ok=%s seconds %.0f elapsed %.2f h' % (R7_OK, _r7_time.time() - _r7_t, (_r7_time.time() - NB_T0) / 3600), flush=True)
'''

new = [
    md("---\n# Part A2b · heliosli's R7 as a second compact model (0.4 of the compact slot)\n\n"
       "goodpjw2008's version 26 recipe: R7 next to R6 in the compact slot at 0.4, R6 keeping 0.6. "
       "R7's cells are heliosli's, unchanged apart from the marked lines (own temp dir and output names, "
       f"workers stop at {R7_DEADLINE_H} h)."),
    code("%%writefile /tmp/r7_setup.py\n" + r7_setup),
    code("%%writefile /tmp/r7_run.py\n" + r7_run),
    code(driver),
]

# Blend: R7 at A_R7 of the compact slot, over the studies both compact models scored.
b = src(cells[i_blend])
b = sub1(b, "n = len(test_ids)\n", f'''n = len(test_ids)
A_R7 = {A_R7}   # [r7] share of the compact slot given to R7 (R6 keeps the rest); fixed before any run
r7, r7_fail = None, np.ones(n, dtype=bool)
if R7_OK:
    try:
        _r = pd.read_csv(WORK / 'r7_sub.csv', dtype={{'StudyInstanceUID': str}})
        assert sorted(_r['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_r.columns)
        _r = _r.set_index('StudyInstanceUID').loc[test_ids, labels]
        assert np.isfinite(_r.values).all()
        _s7 = set()
        for f in R7_PRED.glob('worker*/predictions.csv'):
            with open(f, encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    try:
                        v = [float(r[L]) for L in labels]
                    except (KeyError, ValueError, TypeError):
                        continue
                    if all(0. <= x <= 1. for x in v):
                        _s7.add(r['StudyInstanceUID'])
        r7_fail = np.array([u not in _s7 for u in test_ids])
        if (~r7_fail).mean() < {R7_MIN_COVER}:
            print('R7_DROPPED coverage %d of %d' % ((~r7_fail).sum(), n), flush=True)
        else:
            r7 = _r
    except Exception as _e:
        print('R7_UNUSABLE', repr(_e), flush=True); r7 = None
''')
b = sub1(b, "    if r224 is not None:   # [s5]", '''    if r7 is not None:   # [r7] mix R7 into the compact slot over the studies both compact models scored
        _both = ~our_fail & ~r7_fail
        ro = ro.copy()
        ro[_both] = (1.0 - A_R7) * pct_on(ours[c].values, _both)[_both] + A_R7 * pct_on(r7[c].values, _both)[_both]
    if r224 is not None:   # [s5]''')
b = sub1(b, "         'rank_basis': 'each model over its own non-failed rows'}",
         "         'rank_basis': 'each model over its own non-failed rows',\n"
         "         'r7_used': r7 is not None, 'r7_share': A_R7, 'r7_failed': int(r7_fail.sum())}   # [r7]")
if args.a224 != 0.35:
    b = sub1(b, "A224 = 0.35   #", f"A224 = {args.a224}   # [probe] was 0.35;")
if args.w_raptor != 0.5:
    b = sub1(b, "W_RAPTOR, W_OURS = 0.5, 0.5", f"W_RAPTOR, W_OURS = {args.w_raptor}, {round(1 - args.w_raptor, 4)}   # [probe] was 0.5, 0.5")
cells[i_blend] = code(b)

cells[i_r6run + 1:i_r6run + 1] = new

# Attribution header.
assert cells[0]["cell_type"] == "markdown" and "Attribution" in src(cells[0])
cells.insert(1, md("### Change in this version\n\n"
                   "heliosli's R7 compact model added at 0.4 of the compact slot next to R6 (goodpjw2008's "
                   "version 26 recipe; Part A2b and the marked `[r7]` lines of the blend). Everything else is "
                   "the plain fork. R7: heliosli/rsna-knee-compact-r7cnxfsss, code from heliosli's "
                   "0-957-latest-dataset-score-in-one-click."))

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{slug}.ipynb").write_text(json.dumps(fork, ensure_ascii=False, indent=1), encoding="utf-8")
meta = json.loads((FORK.parent / "kernel-metadata.json").read_text())
meta.update(id=f"shivpratap0007/{slug}", title=slug, code_file=f"{slug}.ipynb")
assert "heliosli/rsna-knee-compact-r7cnxfsss" in meta["dataset_sources"]
(out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print("built", out_dir / f"{slug}.ipynb", len(cells), "cells")
