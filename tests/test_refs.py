"""REFS cluster: bibliography integrity, verified metadata, and the field-core-override check.

The metadata pinned here was verified on 2026-09-11 against Crossref/DataCite DOI records, the
arXiv API, or the primary document (see the REFS response in the review package).
"""
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / 'paper' / 'refs.bib'
TEX = ROOT / 'paper' / 'paper.tex'


# ------------------------------------------------------------------ a small BibTeX reader
def _value(body, i):
    """Parse one field value starting at body[i]; return (value, next_index)."""
    while body[i].isspace():
        i += 1
    if body[i] == '{':
        depth, j = 0, i
        while True:
            if body[j] == '{':
                depth += 1
            elif body[j] == '}':
                depth -= 1
                if depth == 0:
                    return body[i + 1:j], j + 1
            j += 1
    if body[i] == '"':
        j = body.index('"', i + 1)
        return body[i + 1:j], j + 1
    m = re.match(r'[A-Za-z0-9_.:-]+', body[i:])
    return m.group(0), i + m.end()


def parse_bib(text):
    entries = {}
    for m in re.finditer(r'@(\w+)\s*\{', text):
        kind = m.group(1).lower()
        start = m.end()
        depth, j = 1, start
        while depth:
            depth += {'{': 1, '}': -1}.get(text[j], 0)
            j += 1
        body = text[start:j - 1]
        if kind in ('preamble', 'comment', 'string'):
            continue
        key, rest = body.split(',', 1)
        fields, i = {}, 0
        while True:
            fm = re.compile(r'\s*,?\s*([A-Za-z]+)\s*=\s*').match(rest, i)
            if not fm:
                break
            val, i = _value(rest, fm.end())
            fields[fm.group(1).lower()] = ' '.join(val.split())
        entries[key.strip()] = dict(kind=kind, **fields)
    return entries


@pytest.fixture(scope='module')
def bib():
    return parse_bib(BIB.read_text())


def cited_keys(tex):
    keys = set()
    for m in re.finditer(r'\\cite(?:\[[^\]]*\])?\{([^}]*)\}', tex):
        keys.update(k.strip() for k in m.group(1).split(','))
    return keys


# ------------------------------------------------------------------ integrity
def test_parser_on_a_known_entry():
    e = parse_bib('@article{k1,\n author = {A and B}, title = {{X}: y},\n month = may, year = {2020}}')
    assert e == {'k1': dict(kind='article', author='A and B', title='{X}: y', month='may', year='2020')}


def test_every_citation_in_the_paper_resolves(bib):
    missing = cited_keys(TEX.read_text()) - set(bib)
    assert not missing, f'cited but not in refs.bib: {sorted(missing)}'


def test_keys_used_by_the_refs_revision_exist(bib):
    needed = {'ibm_p9um', 'gonzalez2017power9', 'hostboot_p9', 'skiboot_dt', 'top500_m100',
              'm100_zenodo', 'mccalpin2021knl', 'mccalpin2021capid', 'powell2009salvaging',
              'shivakumar2003redundancy',
              'deneckere1996damaged', 'kim2022thermalbleed', 'masti2015thermal',
              'ostrouchov2020titan', 'antici2023pm100'}
    assert needed <= set(bib)


def test_dois_are_well_formed_and_rendered(bib):
    for key, e in bib.items():
        if 'doi' in e:
            assert re.fullmatch(r'10\.\d{4,9}/\S+', e['doi']), key
            # IEEEtran.bst 1.14 ignores `doi`; the DOI must also be printed via \bibdoi
            assert f"\\bibdoi{{{e['doi']}}}" in e.get('note', ''), key


def test_journal_and_conference_entries_carry_a_doi(bib):
    no_doi_ok = {'masti2015thermal'}          # USENIX registers no DOIs
    for key, e in bib.items():
        if e['kind'] in ('article', 'inproceedings', 'incollection') and key not in no_doi_ok:
            assert 'doi' in e, key


def test_notes_carry_metadata_only(bib):
    allowed = re.compile(r'^(\\bibdoi\{[^}]+\}'
                         r'|\{Accessed\}: [A-Z][a-z]{2}\.~\d{1,2}, \d{4}'
                         r'|First record \\bibdoi\{[^}]+\})$')
    for key, e in bib.items():
        if 'note' in e:
            assert allowed.match(e['note']), f'{key}: prose in note: {e["note"]!r}'


def test_arxiv_year_matches_identifier(bib):
    for key, e in bib.items():
        m = re.search(r'arXiv:(\d{2})(\d{2})\.\d{4,5}', e.get('howpublished', ''))
        if m:
            assert int(e['year']) == 2000 + int(m.group(1)), key
            assert 'author' in e, f'{key}: arXiv entry without authors'


def test_no_anonymous_et_al_where_the_list_is_short(bib):
    # 'and others' is kept only for the 12-creator Zenodo dataset entry
    for key, e in bib.items():
        if 'and others' in e.get('author', ''):
            assert key == 'm100_zenodo', key


# ------------------------------------------------------------------ metadata verified 2026-09-11
@pytest.mark.parametrize('key,field,value', [
    ('koren1993negbinom', 'pages', '724--734'),          # next article in TC 42(6) starts p. 735
    ('m100_setdiff', 'year', '2026'),
    ('m100_setdiff', 'doi', '10.1145/3801487.3806064'),
    ('m100_setdiff', 'pages', '109--112'),
    ('hardharvest2025', 'pages', '708--722'),
    ('badnodes2020', 'pages', '123--130'),
    ('nodesentry2025', 'pages', '1180--1194'),
    ('inadomi2015variation', 'doi', '10.1145/2807591.2807638'),
    ('sinha2022gpuvariability', 'doi', '10.1109/SC41404.2022.00070'),
    ('jetta2025provenance', 'doi', '10.1007/s10836-025-06161-x'),
    ('powell2009salvaging', 'doi', '10.1145/1555754.1555769'),   # ISCA '09 proceedings DOI
    ('kim2022thermalbleed', 'pages', '25718--25731'),
    ('mccalpin2021capid', 'doi', '10.26153/tsw/13119'),
    ('mccalpin2021knl', 'number', 'TR-2021-04'),          # no DOI registered; handle is persistent
    ('mccalpin2021knl', 'url', 'https://hdl.handle.net/2152/89580'),
    ('thompto2016power9', 'doi', '10.1109/HOTCHIPS.2016.7936223'),
])
def test_verified_metadata(bib, key, field, value):
    assert bib[key][field] == value


@pytest.mark.parametrize('key,needle', [
    ('borghesi2023m100', 'Mauri, Alessio'),
    ('borghesi2023m100', 'Seyedkazemi Ardebili, Mohsen'),
    ('bae2007waferyield', 'Hwang, Jung Yoon'),
    ('m100_setdiff', 'Esposito, Giovanni B.'),
    ('m100_energy', 'Cornelius, Melanie'),
    ('m100_energy', 'Lan, Zhiling'),
    ('m100_ontology', 'Khan, Junaid Ahmed'),
    ('hardharvest2025', 'Torrellas, Josep'),
    ('jetta2025provenance', 'Asadizanjani, Navid'),
    ('nodesentry2025', 'Feng, Jinghua'),
])
def test_verified_authors(bib, key, needle):
    assert needle in bib[key]['author']


def test_seiz_chapter_has_editors(bib):
    assert bib['badnodes2020']['editor'].startswith('Resch, Michael M.')


# ------------------------------------------------------------------ field-core-override signature
def disabled_index_stats(M):
    """Per-slice disable rate and mean disabled slice index for a (sockets x 12) 0/1 matrix."""
    M = np.asarray(M, dtype=float)
    rate = M.mean(axis=0)
    mean_idx = (M * np.arange(M.shape[1])).sum() / M.sum()
    return rate, mean_idx


def test_disabled_index_stats_on_constructed_maps():
    # FCO on a fully good die keeps core pairs 0..7, i.e. disables slices 8..11
    fco = np.tile(np.r_[np.zeros(8), np.ones(4)], (5, 1))
    rate, mean_idx = disabled_index_stats(fco)
    assert mean_idx == pytest.approx(9.5)
    assert rate.argmax() >= 8
    uniform = np.array([[1 if k in s else 0 for k in range(12)]
                        for s in ((0, 1, 2, 3), (8, 9, 10, 11), (4, 5, 6, 7))])
    assert disabled_index_stats(uniform)[1] == pytest.approx(5.5)


def rack_bootstrap_mean_idx(M, rack, B=4000, seed=20260911):
    """Percentile 95% interval for the mean disabled index, resampling whole racks (P6)."""
    M = np.asarray(M, dtype=float)
    groups = [np.flatnonzero(rack == r) for r in np.unique(rack)]
    rng = np.random.default_rng(seed)
    k = np.arange(M.shape[1])
    rows_idx = (M * k).sum(axis=1)                # per-socket sum of disabled indices
    rows_n = M.sum(axis=1)
    g_idx = np.array([rows_idx[g].sum() for g in groups])
    g_n = np.array([rows_n[g].sum() for g in groups])
    draws = rng.integers(0, len(groups), size=(B, len(groups)))
    est = g_idx[draws].sum(axis=1) / g_n[draws].sum(axis=1)
    return np.percentile(est, [2.5, 97.5])


def test_rack_bootstrap_degenerates_when_racks_are_identical():
    one = np.r_[np.ones(4), np.zeros(8)]
    M = np.tile(one, (6, 1))
    lo, hi = rack_bootstrap_mean_idx(M, np.repeat([0, 1, 2], 2), B=200)
    assert lo == hi == pytest.approx(1.5)


def test_observed_maps_run_opposite_to_the_override_signature():
    M = np.load(ROOT / 'analysis' / 'harvest_maps_full.npy').astype(int)
    K = np.load(ROOT / 'analysis' / 'harvest_keys_full.npy')
    assert M.shape == (1962, 12) and (M.sum(axis=1) == 4).all()
    lot_a = (K[:, 0] // 20) <= 21
    for sel, rate0, mean_idx in ((lot_a, 0.882, 3.85), (~lot_a, 0.393, 5.31)):
        rate, m = disabled_index_stats(M[sel])
        assert rate.argmax() == 0                 # slice 0 is the most-disabled slice in each lot
        assert rate[0] == pytest.approx(rate0, abs=5e-4)
        assert m == pytest.approx(mean_idx, abs=5e-3)
        assert m < 5.5                            # below uniform, far below the override's 9.5
    top4 = np.r_[np.zeros(8), np.ones(4)].astype(int)
    assert int((M == top4).all(axis=1).sum()) == 9


def test_top_slices_are_not_over_harvested_in_either_lot():
    # The override keeps the lowest-numbered functional pairs, so its footprint is an excess at
    # slices 8-11. Each lot's mean rate over slices 8-11 is at or below the 4/12 average.
    M = np.load(ROOT / 'analysis' / 'harvest_maps_full.npy').astype(int)
    K = np.load(ROOT / 'analysis' / 'harvest_keys_full.npy')
    lot_a = (K[:, 0] // 20) <= 21
    for sel, top_mean in ((lot_a, 0.220), (~lot_a, 0.324)):
        rate, _ = disabled_index_stats(M[sel])
        assert rate[8:].mean() == pytest.approx(top_mean, abs=5e-4)
        assert rate[8:].mean() <= 4 / 12
        assert rate[0] >= rate.max()              # slice 0 at least as often as any other slice


def test_rack_clustered_interval_stays_below_uniform():
    M = np.load(ROOT / 'analysis' / 'harvest_maps_full.npy').astype(int)
    K = np.load(ROOT / 'analysis' / 'harvest_keys_full.npy')
    rack = K[:, 0] // 20
    lot_a = rack <= 21
    lo_a, hi_a = rack_bootstrap_mean_idx(M[lot_a], rack[lot_a])
    lo_b, hi_b = rack_bootstrap_mean_idx(M[~lot_a], rack[~lot_a])
    # seeded, so exact; other seeds move the endpoints by about 0.01
    assert (round(lo_a, 2), round(hi_a, 2)) == (3.74, 3.94)
    assert (round(lo_b, 2), round(hi_b, 2)) == (5.16, 5.44)
    assert hi_a < 5.5 and hi_b < 5.5
