"""Cross-surface consistency: the public pages must say what the result files say.

A research artefact fails quietly in a particular way -- a number is corrected in the code, the
result file follows, and the README, the docs site or the issue register keeps the old value. These
tests close that gap mechanically. They read the machine-readable outputs in `analysis/` and the
prose in `README.md`, `SECURITY.md`, `LIMITATIONS.md`, `CHANGELOG.md`, `CITATION.cff` and `docs/`,
and fail if the two disagree.

What is deliberately *not* tested here: the paper. `paper/paper.tex` is the authors' copy and this
repository does not edit it; the one thing asserted about it is that every open issue it cites
exists in the public register.

Run:  .venv/bin/python -m pytest -q tests/test_consistency.py
"""
import math
import hashlib
import re
from math import comb
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, result

PAGES = {
    'README.md': ROOT / 'README.md',
    'SECURITY.md': ROOT / 'SECURITY.md',
    'LIMITATIONS.md': ROOT / 'LIMITATIONS.md',
    'CHANGELOG.md': ROOT / 'CHANGELOG.md',
    'CITATION.cff': ROOT / 'CITATION.cff',
    'DATA.md': ROOT / 'DATA.md',
    'docs/index.md': ROOT / 'docs' / 'index.md',
    'docs/findings.md': ROOT / 'docs' / 'findings.md',
    'docs/method.md': ROOT / 'docs' / 'method.md',
    'docs/limitations.md': ROOT / 'docs' / 'limitations.md',
    'docs/reproduce.md': ROOT / 'docs' / 'reproduce.md',
    'docs/faq.md': ROOT / 'docs' / 'faq.md',
    'docs/llms.txt': ROOT / 'docs' / 'llms.txt',
}


def norm(s: str) -> str:
    """ASCII-ise the typography so one regex matches prose written with real dashes."""
    return (s.replace('−', '-').replace('–', '-').replace('—', '-')
             .replace('±', '+/-').replace(' ', ' ').replace('’', "'")
             .replace('≥', '>=').replace('≤', '<=').replace('×', 'x'))


def page(name: str) -> str:
    return norm(PAGES[name].read_text())


def only(pattern: str, name: str, group: int = 1) -> str:
    """The single match of `pattern` on `name`; fails loudly if it is absent or ambiguous."""
    hits = re.findall(pattern, page(name))
    assert hits, f'{name} no longer contains a number matching /{pattern}/'
    vals = {h if isinstance(h, str) else h[group - 1] for h in hits}
    assert len(vals) == 1, f'{name} gives conflicting values for /{pattern}/: {sorted(vals)}'
    return vals.pop()


def num(pattern: str, name: str, group: int = 1) -> float:
    return float(only(pattern, name, group))


# --------------------------------------------------------------- the open-issue register
def test_every_open_issue_the_paper_cites_exists_in_the_public_register():
    """LIMITATIONS.md owns the OI-n ids. A dangling reference in the paper is a broken promise."""
    tex = (ROOT / 'paper' / 'paper.tex').read_text()
    cited = {int(m) for m in re.findall(r'OI-(\d+)', tex)}
    assert cited, 'the manuscript cites no open issue at all -- is paper.tex the right file?'
    defined = set()
    for line in page('LIMITATIONS.md').split('\n'):
        if line.startswith('## OI-'):
            defined |= {int(m) for m in re.findall(r'OI-(\d+)', line.split('-', 1)[0] + line)}
    missing = sorted(cited - defined)
    assert not missing, f'paper.tex cites OI-{missing} with no entry in LIMITATIONS.md'


def test_open_issue_ids_are_unique():
    ids = [int(m) for line in page('LIMITATIONS.md').split('\n') if line.startswith('## OI-')
           for m in re.findall(r'OI-(\d+)', line)]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f'LIMITATIONS.md defines OI-{dupes} more than once'


def test_every_open_issue_referenced_by_a_public_page_exists():
    defined = {int(m) for line in page('LIMITATIONS.md').split('\n') if line.startswith('## OI-')
               for m in re.findall(r'OI-(\d+)', line)}
    for name in ('README.md', 'SECURITY.md', 'CHANGELOG.md', 'docs/index.md', 'docs/findings.md',
                 'docs/limitations.md', 'docs/faq.md', 'docs/llms.txt'):
        used = {int(m) for m in re.findall(r'OI-(\d+)', page(name))}
        missing = sorted(used - defined)
        assert not missing, f'{name} points at OI-{missing}, which LIMITATIONS.md does not define'


# --------------------------------------------------------------- headline numbers, page vs result
def test_fleet_and_pattern_counts():
    maps = np.load(ROOT / 'analysis' / 'harvest_maps_full.npy')
    sockets, distinct = maps.shape[0], len({tuple(r) for r in maps})
    assert sockets == 1962 and distinct == 443          # guards the files themselves
    assert comb(12, 4) == 495
    for name in ('README.md', 'CITATION.cff', 'docs/llms.txt', 'docs/findings.md'):
        assert f'{sockets:,}' in page(name), f'{name} does not carry the {sockets:,}-socket count'
    assert num(r'(\d+) of 495 possible patterns', 'README.md') == distinct
    assert num(r'\*\*(\d+)\*\* distinct patterns appear among the 495', 'docs/findings.md') == distinct


def test_clustering_null_numbers():
    cb = result('model_curveball')['main']
    assert num(r'z = (-\d+\.\d) against 100,000 exchangeable', 'README.md') == round(cb['gap_z'], 1)
    assert num(r'(\d+) of 66 pairs family-wise significant', 'README.md') == cb['n_sig_maxT']
    assert num(r'(\d+) of the 66 pairs\s+are family-wise significant', 'docs/findings.md') \
        == cb['n_sig_maxT']
    # 8 of the 9 positive excesses sit at |dk| <= 2; the ninth is the far pair 7&10
    assert len(cb['near_pos_maxT']) == 8 and len(cb['far_pos_maxT']) == 1
    assert cb['n_sig_maxT_pos'] == 9
    assert num(r'(\d+) of the 9 positive excesses', 'README.md') == len(cb['near_pos_maxT'])


def test_thermal_quad_numbers():
    q = result('thermal_socket_stats')['quad_decomposition']
    ibm = result('thermal_ibm_layout')['months']['22-08']['pooled']
    r = page('README.md')
    assert f"+{q['samequad']['pooled']:.2f}" in r            # +0.48 same quad, other slice
    assert f"+{q['within']['pooled']:.2f}" in r              # +0.37 same slice
    assert f"+{q['crossquad']['pooled']:.2f}" in r           # +0.10 across a quad boundary
    # the docs quote the layout scores to three decimals; agree to within that rounding
    for pattern, val in ((r'1x12 linear \(identity index order\) \| (-\d\.\d+)', ibm['linear']),
                         (r"documented layout, at the diagram's proportions \| (-\d\.\d+)", ibm['ibm']),
                         (r'exact optimum over all 12!/2 orderings[^|]*\| (-\d\.\d+)', ibm['opt'])):
        assert num(pattern, 'docs/findings.md') == pytest.approx(val, abs=0.001)
    # the ordering claim the paper now makes: IBM wins overall, the index order wins across quads
    assert ibm['ibm'] < ibm['linear'] and ibm['linear_noquad'] < ibm['ibm_noquad']
    assert num(r'the index order fits better \((-\d\.\d+) vs', 'docs/findings.md') \
        == pytest.approx(ibm['linear_noquad'], abs=0.001)


def test_lot_boundary_numbers():
    cb = result('cohort_boundary')
    null = np.load(ROOT / 'analysis' / 'changepoint_null.npy')
    prof = np.load(ROOT / 'analysis' / 'changepoint_profile.npy')
    obs = float(np.abs(prof[1]).max())
    p_mc = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    assert cb['section1_boundary_uncertainty']['observed_cut'] == 22
    assert f'{p_mc:.1e}'.replace('e-04', 'e-4') == '2.5e-4'
    for name in ('README.md', 'docs/index.md'):
        assert '2.5e-4' in page(name), f'{name} does not carry the calibrated boundary p'
    worst = min(v['share_at_observed']
                for v in cb['section1_boundary_uncertainty']['resampling'].values())
    assert num(r'rack 22 in >= (\d+\.\d)% of rack-level bootstrap', 'README.md') \
        == pytest.approx(100 * worst, abs=0.05)
    acc = cb['section3_nested_classification']['welch']['cb']
    assert acc['racks_correct_vs_full_label'] == 47
    assert num(r'(\d+)/49 held-out racks', 'README.md') == 47


def test_reboot_and_guard_numbers():
    rb, xc = result('lifecycle_reboot'), result('lifecycle_os_crosscheck')
    r = page('README.md')
    assert f"{rb['dayflag']['transitions_with_reboot']}/{rb['dayflag']['transitions_covered']}" in r
    assert f"{rb['epoch']['t_boot']} of {rb['epoch']['transitions']}" in r
    assert rb['epoch']['t_no_boot'] == 0, 'a transition without a boot would break the claim'
    assert xc['os_to_bmc']['n'] == xc['os_to_bmc']['matched'] == 9
    assert re.search(r'matched by a BMC core loss \(9 of 9\)', r)
    assert 'nine in steady state' in r


def test_replacement_rate_numbers():
    t = result('lifecycle_taxonomy_errors')
    rate, rb = t['rate']['rate'], t['rate']['rack_bootstrap']
    r = page('README.md')
    assert f'{100 * rate:.1f}% per socket-year' in r
    assert f'[{100 * rb["lo"]:.1f}, {100 * rb["hi"]:.1f}]' in r
    assert f"{t['FP4_die_moves']['traced_7d']} of the {t['rate']['dies']} changes" in r
    lo, hi = (100 * x for x in t['rate_new_dies']['range_rate'])
    assert f'{lo:.1f}-{hi:.1f}%' in r, f'README does not carry the new-processor rate {lo:.1f}-{hi:.1f}%'


def test_survival_null_numbers():
    c = result('survival_cindex')
    r = page('README.md')
    assert f"concordance {c['cindex']:.2f}" in r
    assert f"[{c['cindex_ci95'][0]:.2f}, {c['cindex_ci95'][1]:.2f}]" in r
    assert f"null {c['cindex_null_mean']:.2f} +/- {c['cindex_null_sd']:.2f}" in r
    assert f"p = {c['cindex_p']:.2f}" in r
    assert c['cindex_ci95'][1] < 0.7, 'the "not useful prediction" framing needs an upper bound < 0.7'


def test_power_equivalence_numbers():
    fit = result('equivalence_fit')
    head = fit['headline']
    rows = {r['feature']: r for r in fit['rows']
            if r['design'] == head['design'] and r['model'] == head['model']}
    assert set(rows) >= {'meanidx', 'nadj', 'lowhalf'}, sorted(rows)
    r = page('README.md')
    for key in ('meanidx', 'lowhalf', 'nadj'):
        e = rows[key]['range']['eff']
        assert f'{e:+.1f} W' in r or f'{e:+.1f},' in r or f'{e:+.1f} ' in r, \
            f'README does not carry the {e:+.1f} W effect of {key}'
    bound = max(rows[k]['eq_bound'] for k in ('meanidx', 'nadj', 'lowhalf'))
    assert f'+/-{bound:.1f} W' in r, f'README does not carry the {bound:.1f} W smallest cleared margin'
    # The robustness envelope must be quoted ROUNDED UP. Its measured maximum is 7.3192 W, so
    # "+/-7.3 W" would state a bound that does not hold -- by 0.02 W, in the unsafe direction.
    env = math.ceil(fit['robustness_max_bound'] * 10) / 10
    assert f'+/-{env:.1f} W' in r, f'README does not carry the {env:.1f} W robustness envelope'
    assert f"+/-{fit['robustness_max_bound']:.1f} W" not in r, \
        'the robustness envelope is quoted rounded down, which understates it'
    assert 'a few watts' in r
    assert rows['meanidx']['n'] == 636_026 and rows['meanidx']['nodes'] == 980


def test_side_channel_numbers():
    raw, sc = result('side_channel_raw'), result('side_channel_results')
    one_day = sc['iid']['windows']['1']['p_exact'][sc['iid']['f'].index(0.999)]
    one_month = sc['iid']['windows']['31']['p_exact'][sc['iid']['f'].index(0.9999)]
    for name in ('README.md', 'SECURITY.md'):
        assert f"{100 * raw['single_interval']['measured']:.2f}%" in page(name), \
            f'{name} does not carry the measured single-interval recovery'
    s = page('SECURITY.md')
    assert f'{one_day:.2f}' in s, f'SECURITY.md does not carry one-day recovery {one_day:.2f}'
    assert f'{one_month:.5f}' in s, f'SECURITY.md does not carry one-month recovery {one_month:.5f}'
    uniq = sc['structural']['pseudonymous_expected_unique_share']
    assert f'{100 * uniq:.1f}%' in s


def test_identifiability_numbers():
    d = result('cohort_diversity')
    s, faq = page('SECURITY.md'), page('docs/faq.md')
    h = d['D1_entropy']
    for txt in (s, faq):
        assert f"{h['plugin']['H_bits']:.2f} bits" in txt
        assert f"{h['miller_madow']['H_bits']:.2f}" in txt and f"{h['chao_shen']['H_bits']:.2f}" in txt
    ident = d['D3_identifiability']
    assert f"{100 * ident['map alone (fleet)']['unique_fraction']:.1f}%" in s
    assert str(round(100 * ident['map + rack']['unique_fraction'])) + '%' in s
    assert str(round(100 * ident['node: ordered (p0, p1) pair']['unique_fraction'])) + '%' in s


# --------------------------------------------------------------- claims that were withdrawn
WITHDRAWN = [
    # (substring, pages it must not appear on as a live claim)
    ('recovers the sensor index as the physical',
     ('README.md', 'docs/index.md', 'docs/findings.md', 'CITATION.cff', 'docs/llms.txt')),
    ('about two minutes',
     ('README.md', 'SECURITY.md', 'docs/index.md', 'docs/findings.md', 'docs/faq.md',
      'CITATION.cff', 'docs/llms.txt')),
    ('invisible to the BMC',
     ('README.md', 'docs/index.md', 'docs/findings.md', 'docs/llms.txt')),
    ('95% CI [22, 22]',
     ('README.md', 'docs/index.md', 'docs/findings.md', 'LIMITATIONS.md', 'docs/llms.txt')),
    ('fused off',
     ('README.md', 'SECURITY.md', 'docs/index.md', 'docs/findings.md', 'docs/method.md',
      'CITATION.cff', 'docs/llms.txt')),
]


@pytest.mark.parametrize('claim,pages', WITHDRAWN)
def test_withdrawn_claims_are_gone(claim, pages):
    for name in pages:
        assert claim not in page(name), f'{name} still carries the withdrawn claim "{claim}"'


def test_operationally_invisible_only_appears_as_a_withdrawal():
    """The phrase may be quoted while saying it was withdrawn, never asserted.

    LIMITATIONS.md and CHANGELOG.md are registers of corrections: they quote the old claim in the
    entry that retracts it, and the retraction may be a paragraph away, so they are checked as a
    whole file rather than paragraph by paragraph."""
    for name in ('LIMITATIONS.md', 'CHANGELOG.md'):
        t = page(name)
        if 'operationally invisible' in t:
            assert 'withdraw' in t.lower(), f'{name} quotes the claim without retracting it'
    for name, p in PAGES.items():
        if name in ('LIMITATIONS.md', 'CHANGELOG.md'):
            continue
        for para in norm(p.read_text()).split('\n\n'):
            if 'operationally invisible' in para:
                assert 'withdraw' in para.lower(), \
                    f'{name} uses "operationally invisible" without saying it was withdrawn'


def test_the_ci_badge_is_not_described_as_validating_everything():
    """AR-01: a green badge means the committed tables still give the asserted numbers."""
    r, d = page('README.md'), page('docs/reproduce.md')
    assert 'does **not** re-derive' in r or 'does not re-derive' in r
    assert 'not that every number in' in d
    for name in ('README.md', 'docs/reproduce.md'):
        assert 'validates every' not in page(name)


# --------------------------------------------------------------- the data manifest
def test_data_manifest_matches_the_committed_tables():
    lines = [l for l in (ROOT / 'DATA.sha256').read_text().split('\n') if l.strip()]
    assert len(lines) > 150, 'DATA.sha256 looks truncated'
    for line in lines:
        digest, rel = line.split('  ', 1)
        f = ROOT / rel
        assert f.exists(), f'DATA.sha256 lists {rel}, which is not in the tree'
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        assert got == digest, f'{rel} does not match its recorded sha256'


def test_data_manifest_covers_every_committed_derived_table():
    listed = {l.split('  ', 1)[1] for l in (ROOT / 'DATA.sha256').read_text().split('\n') if l.strip()}
    for d in ('daily', 'cov_all', 'os_all', 'power_all', 'power'):
        on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / d).glob('*.parquet')}
        missing = sorted(on_disk - listed)
        assert not missing, f'DATA.sha256 does not cover {missing}'


def test_data_md_describes_the_schemas_it_claims():
    """Spot-check the documented dtypes against the files, one per table family."""
    pd = pytest.importorskip('pandas')
    cases = {
        'daily/daily_20-04.parquet': {'node': 'int32', 'socket': 'int8', 'core': 'int8', 'n': 'int32'},
        'power_all/pw_20-04.parquet': {'node': 'int32', 'socket': 'int8', 'mean': 'float64',
                                       'count': 'int64'},
        'os_all/os_20-06.parquet': {'node': 'int32', 'value': 'int32', 'n': 'int64'},
    }
    for rel, dtypes in cases.items():
        df = pd.read_parquet(ROOT / rel)
        for col, want in dtypes.items():
            assert str(df[col].dtype) == want, f'{rel}:{col} is {df[col].dtype}, DATA.md says {want}'
        assert str(df['day'].dtype).startswith('datetime64') and 'UTC' in str(df['day'].dtype), \
            f'{rel}: day is not a UTC timestamp, contradicting DATA.md'
        assert (df['day'] == df['day'].dt.floor('D')).all(), f'{rel}: day is not a UTC midnight'


# --------------------------------------------------------------- environment and CI claims
def _floor_pins(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().split('\n'):
        line = line.split('#')[0].strip()
        m = re.match(r'^([A-Za-z0-9_.\-]+)\s*(>=|==)\s*([0-9][^\s]*)$', line)
        if m:
            out[m.group(1).lower()] = (m.group(2), m.group(3))
    return out


def _ver(s: str) -> tuple:
    return tuple(int(x) for x in re.findall(r'\d+', s)[:3])


def test_requirements_floors_are_not_above_the_locked_versions():
    floors = _floor_pins(ROOT / 'requirements.txt')
    lock = _floor_pins(ROOT / 'requirements-lock.txt')
    assert floors and lock
    for pkg, (op, v) in floors.items():
        assert op == '>=', f'requirements.txt should carry floors, not {op} for {pkg}'
        assert pkg in lock, f'{pkg} is a floor but is not pinned in requirements-lock.txt'
        assert _ver(lock[pkg][1]) >= _ver(v), \
            f'requirements-lock.txt pins {pkg} {lock[pkg][1]} below the floor {v}'
    for pkg, (op, _) in lock.items():
        assert op == '==', f'requirements-lock.txt must pin exactly, not {op} for {pkg}'
    dev = _floor_pins(ROOT / 'requirements-dev.txt')
    assert dev.get('pytest', ('', ''))[0] == '==', 'requirements-dev.txt must pin pytest exactly'


def test_the_changelog_does_not_promise_files_that_are_absent():
    """P7: the Unreleased entry names repository artefacts; they have to be there."""
    unreleased = page('CHANGELOG.md').split('## v1.0.0')[0]
    for rel in re.findall(r'`([A-Za-z0-9_./\-]+\.(?:txt|md|sha256|yml|py))`', unreleased):
        if rel.startswith(('analysis/', 'tests/', '.github/')) or '/' not in rel:
            cand = ROOT / rel
            if not cand.exists() and '/' not in rel:
                cand = next(iter(ROOT.glob(f'**/{rel}')), None)
            assert cand is not None and cand.exists(), \
                f'CHANGELOG Unreleased names {rel}, which does not exist'


def test_the_workflow_runs_every_script_the_docs_say_it_runs():
    wf = (ROOT / '.github' / 'workflows' / 'reproduce.yml').read_text()
    claimed = set(re.findall(r'`([a-z0-9_]+\.py)`', page('docs/reproduce.md')
                             .split('## What CI checks')[1].split('## Figures')[0]))
    # the same section also names what CI does *not* run; those appear after "does **not**"
    not_run = set(re.findall(r'`([a-z0-9_]+\.py)`',
                             page('docs/reproduce.md').split('Each step asserts')[1]
                             .split('## Figures')[0]))
    for s in sorted(claimed - not_run):
        assert s in wf, f'docs/reproduce.md says CI runs {s}, but the workflow does not'
    for s in sorted(not_run - claimed):
        assert f'analysis/{s}' not in wf, \
            f'docs/reproduce.md says CI does not run {s}, but the workflow does'
    assert 'sha256sum -c DATA.sha256' in wf
    assert 'pytest' in wf
    for trigger in ('push:', 'pull_request:', 'schedule:'):
        assert trigger in wf, f'the workflow lost its {trigger} trigger'
