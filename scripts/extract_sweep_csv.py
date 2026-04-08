#!/usr/bin/env python3
"""
Comprehensive CSV extraction from 4D sweep best_model_metrics.json files.

Extracts ~96 columns per config into a flat CSV: identity, training metadata,
eval/train/full-data reconstruction + 4-quadrant clustering, centroid-based
metrics, KL diagnostics, baseline ModKMeans, model config, and composite score.

Python 3.6 stdlib only — runs on IBM Power (ppc64le) with Open-CE.

Usage:
    # Single participant:
    python3 extract_sweep_csv.py \\
        --participant 010012 \\
        --sweep-dir /path/to/outputs/010012/sweep_4d_010012 \\
        --output sweep_010012.csv

    # Merge multiple CSVs:
    python3 extract_sweep_csv.py \\
        --merge sweep_010012.csv sweep_010016.csv \\
        --output sweep_merged.csv
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import sys


# ---- Helpers ----------------------------------------------------------------

def safe_num(d, *keys):
    """Navigate nested dict by keys, return '' on any miss/None/exception."""
    try:
        cur = d
        for k in keys:
            if isinstance(cur, dict):
                cur = cur[k]
            elif isinstance(cur, (list, tuple)):
                cur = cur[int(k)]
            else:
                return ''
        if cur is None:
            return ''
        return cur
    except (KeyError, IndexError, TypeError, ValueError):
        return ''


def num_or_blank(val):
    """Return numeric value or '' for missing/None/NaN."""
    if val == '' or val is None:
        return ''
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return ''
        return f
    except (ValueError, TypeError):
        return ''


def parse_folder_name(name):
    """Extract K, batch, latent, depth, ndf from cluster_K_B_LD_D_NDF."""
    m = re.match(r'cluster_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$', name)
    if not m:
        return None
    return {
        'K': int(m.group(1)),
        'batch_size': int(m.group(2)),
        'latent_dim': int(m.group(3)),
        'depth': int(m.group(4)),
        'ndf': int(m.group(5)),
    }


def compute_composite(eval_q1_sil, train_gev):
    """Composite = sqrt(((sil+1)/2) * gev). Matches trainer formula."""
    s = num_or_blank(eval_q1_sil)
    g = num_or_blank(train_gev)
    if s == '' or g == '':
        return ''
    sil_norm = (float(s) + 1.0) / 2.0
    gev = float(g)
    if sil_norm <= 0 or gev <= 0:
        return 0.0
    return math.sqrt(sil_norm * gev)


# ---- Extraction helpers -----------------------------------------------------

# Quadrant JSON sub-key mapping
QUADRANT_MAP = {
    'q1': 'latent_sklearn',
    'q2': 'latent_pycrostates',
    'q3': 'topomap_sklearn',
    'q4': 'topomap_pycrostates',
}
QUAD_METRICS = {
    'sil': 'silhouette',
    'db': 'db',
    'ch': 'ch',
    'dunn': 'dunn',
}


def extract_recon(section, prefix):
    """Extract 5 reconstruction metrics from a quadrant_metrics section."""
    recon = safe_num(section, 'quadrant_metrics', 'reconstruction')
    if not isinstance(recon, dict):
        recon = {}
    row = {}
    row[prefix + '_mse'] = num_or_blank(recon.get('mse'))
    row[prefix + '_mse_volts'] = num_or_blank(recon.get('mse_volts'))
    row[prefix + '_kld'] = num_or_blank(recon.get('kld'))
    row[prefix + '_ssim'] = num_or_blank(recon.get('ssim'))
    row[prefix + '_spatial_corr'] = num_or_blank(recon.get('spatial_corr'))
    return row


def extract_4q(section, prefix):
    """Extract 16 quadrant clustering metrics from a section."""
    clust = safe_num(section, 'quadrant_metrics', 'clustering')
    if not isinstance(clust, dict):
        clust = {}
    row = {}
    for q_short, q_json in QUADRANT_MAP.items():
        q_data = clust.get(q_json, {})
        if not isinstance(q_data, dict):
            q_data = {}
        for m_short, m_json in QUAD_METRICS.items():
            col = '%s_%s_%s' % (prefix, q_short, m_short)
            row[col] = num_or_blank(q_data.get(m_json))
    return row


def extract_centroid(cb, prefix_lat, prefix_raw):
    """Extract 16 centroid-based metrics from centroid_based section."""
    row = {}
    for space_key, prefix in [('latent_space', prefix_lat),
                               ('raw_space', prefix_raw)]:
        space = {}
        raw_space = safe_num(cb, space_key, 'centroid_based_metrics')
        if isinstance(raw_space, dict):
            space = raw_space
        for dist_key, dist_suffix in [('sklearn', 'eucl'),
                                       ('correlation_based', 'corr')]:
            dist = space.get(dist_key, {})
            if not isinstance(dist, dict):
                dist = {}
            col_base = '%s_%s' % (prefix, dist_suffix)
            row[col_base + '_sil'] = num_or_blank(dist.get('silhouette'))
            row[col_base + '_db'] = num_or_blank(dist.get('davies_bouldin'))
            row[col_base + '_ch'] = num_or_blank(dist.get('calinski_harabasz'))
            row[col_base + '_dunn'] = num_or_blank(dist.get('dunn'))
    return row


def find_epoch_entry(d, best_epoch):
    """Find the epoch entry matching best_epoch in training_history.epochs."""
    epochs = safe_num(d, 'training_history', 'epochs')
    if not isinstance(epochs, list):
        return {}
    be = num_or_blank(best_epoch)
    if be == '':
        return {}
    target = int(float(be))
    for entry in epochs:
        if isinstance(entry, dict) and entry.get('epoch') == target:
            return entry
    # Fallback: last entry
    if epochs:
        last = epochs[-1]
        if isinstance(last, dict):
            return last
    return {}


# ---- Main extraction --------------------------------------------------------

# Canonical column order (96 columns)
COLUMNS = [
    # Identity (7)
    'subject', 'config', 'K', 'batch_size', 'latent_dim', 'depth', 'ndf',
    # Training metadata (7)
    'best_epoch', 'total_epochs', 'train_gev', 'full_data_gev',
    'electrode_gev', 'best_train_loss', 'loss_at_best_epoch',
    # Eval reconstruction (5)
    'eval_mse', 'eval_mse_volts', 'eval_kld', 'eval_ssim',
    'eval_spatial_corr',
    # Eval 4-quadrant clustering (16)
    'eval_q1_sil', 'eval_q1_db', 'eval_q1_ch', 'eval_q1_dunn',
    'eval_q2_sil', 'eval_q2_db', 'eval_q2_ch', 'eval_q2_dunn',
    'eval_q3_sil', 'eval_q3_db', 'eval_q3_ch', 'eval_q3_dunn',
    'eval_q4_sil', 'eval_q4_db', 'eval_q4_ch', 'eval_q4_dunn',
    # Train reconstruction (5)
    'train_mse', 'train_mse_volts', 'train_kld', 'train_ssim',
    'train_spatial_corr',
    # Train 4-quadrant clustering (16)
    'train_q1_sil', 'train_q1_db', 'train_q1_ch', 'train_q1_dunn',
    'train_q2_sil', 'train_q2_db', 'train_q2_ch', 'train_q2_dunn',
    'train_q3_sil', 'train_q3_db', 'train_q3_ch', 'train_q3_dunn',
    'train_q4_sil', 'train_q4_db', 'train_q4_ch', 'train_q4_dunn',
    # Full-data metrics (22)
    'full_n_samples', 'full_gev',
    'full_mse', 'full_mse_volts', 'full_kld', 'full_ssim',
    'full_spatial_corr',
    'full_q1_sil', 'full_q1_db', 'full_q1_ch', 'full_q1_dunn',
    'full_q2_sil', 'full_q2_db', 'full_q2_ch', 'full_q2_dunn',
    'full_q3_sil', 'full_q3_db', 'full_q3_ch', 'full_q3_dunn',
    'full_q4_sil', 'full_q4_db', 'full_q4_ch', 'full_q4_dunn',
    # Eval centroid-based (16)
    'cent_lat_eucl_sil', 'cent_lat_eucl_db', 'cent_lat_eucl_ch',
    'cent_lat_eucl_dunn',
    'cent_lat_corr_sil', 'cent_lat_corr_db', 'cent_lat_corr_ch',
    'cent_lat_corr_dunn',
    'cent_raw_eucl_sil', 'cent_raw_eucl_db', 'cent_raw_eucl_ch',
    'cent_raw_eucl_dunn',
    'cent_raw_corr_sil', 'cent_raw_corr_db', 'cent_raw_corr_ch',
    'cent_raw_corr_dunn',
    # KL diagnostics (3)
    'kl_active_dims', 'kl_collapsed_dims', 'kl_collapse_ratio',
    # Baseline ModKMeans (7)
    'baseline_available', 'baseline_gev', 'baseline_sil', 'baseline_db',
    'baseline_ch', 'baseline_dunn', 'baseline_composite',
    # Model config (4)
    'lr', 'config_patience', 'pretrain_epochs', 'config_total_epochs',
    # Computed (1)
    'composite',
]


def extract_one_config(json_path, subject, folder_name, params):
    """Extract all metrics from one best_model_metrics.json into a flat row."""
    try:
        with open(str(json_path), 'r') as f:
            d = json.load(f)
    except (ValueError, IOError, OSError):
        return None

    row = {}

    # ---- Identity ----
    row['subject'] = subject
    row['config'] = folder_name
    row['K'] = params['K']
    row['batch_size'] = params['batch_size']
    row['latent_dim'] = params['latent_dim']
    row['depth'] = params['depth']
    row['ndf'] = params['ndf']

    # ---- Training metadata ----
    meta = d.get('metadata', {})
    if not isinstance(meta, dict):
        meta = {}
    row['best_epoch'] = num_or_blank(meta.get('best_epoch'))
    row['total_epochs'] = num_or_blank(meta.get('total_epochs_trained'))
    row['train_gev'] = num_or_blank(meta.get('best_gev'))
    row['best_train_loss'] = num_or_blank(meta.get('best_train_loss'))

    bm = d.get('best_model_metrics', {})
    if not isinstance(bm, dict):
        bm = {}

    # GEV variants
    gev_section = bm.get('gev', {})
    if not isinstance(gev_section, dict):
        gev_section = {}
    row['full_data_gev'] = num_or_blank(gev_section.get('pixel_space_full_data'))
    row['electrode_gev'] = num_or_blank(gev_section.get('electrode_space'))

    # loss_at_best_epoch — try train section
    train_sec = bm.get('train', {})
    if not isinstance(train_sec, dict):
        train_sec = {}
    row['loss_at_best_epoch'] = num_or_blank(train_sec.get('loss_at_best_epoch'))

    # ---- Eval section ----
    eval_sec = bm.get('eval', {})
    if not isinstance(eval_sec, dict):
        eval_sec = {}
    # Fall back to train if eval missing
    use_eval = eval_sec if eval_sec else train_sec

    row.update(extract_recon(use_eval, 'eval'))
    row.update(extract_4q(use_eval, 'eval'))

    # ---- Train section ----
    row.update(extract_recon(train_sec, 'train'))
    row.update(extract_4q(train_sec, 'train'))

    # ---- Full-data section ----
    full_sec = bm.get('full_data', {})
    if not isinstance(full_sec, dict):
        full_sec = {}
    row['full_n_samples'] = num_or_blank(full_sec.get('n_samples'))
    row['full_gev'] = num_or_blank(full_sec.get('gev'))
    row.update(extract_recon(full_sec, 'full'))
    row.update(extract_4q(full_sec, 'full'))

    # ---- Centroid-based (prefer eval, fall back to train) ----
    cb_eval = safe_num(use_eval, 'centroid_based')
    if not isinstance(cb_eval, dict):
        cb_eval = {}
    row.update(extract_centroid(cb_eval, 'cent_lat', 'cent_raw'))

    # ---- KL diagnostics at best epoch ----
    epoch_entry = find_epoch_entry(d, row['best_epoch'])
    row['kl_active_dims'] = num_or_blank(epoch_entry.get('kl_active_dims'))
    row['kl_collapsed_dims'] = num_or_blank(epoch_entry.get('kl_collapsed_dims'))
    row['kl_collapse_ratio'] = num_or_blank(epoch_entry.get('kl_collapse_ratio'))

    # ---- Baseline ModKMeans ----
    bl = d.get('baseline_modkmeans', {})
    if not isinstance(bl, dict):
        bl = {}
    row['baseline_available'] = 1 if bl.get('available') else 0
    bl_cvm = bl.get('cluster_validation_metrics', {})
    if not isinstance(bl_cvm, dict):
        bl_cvm = {}
    row['baseline_gev'] = num_or_blank(bl_cvm.get('gev'))
    row['baseline_sil'] = num_or_blank(bl_cvm.get('silhouette'))
    row['baseline_db'] = num_or_blank(bl_cvm.get('davies_bouldin'))
    row['baseline_ch'] = num_or_blank(bl_cvm.get('calinski_harabasz'))
    row['baseline_dunn'] = num_or_blank(bl_cvm.get('dunn'))
    # Composite: try both nesting styles
    bl_comp = bl_cvm.get('composite_scores', {})
    if not isinstance(bl_comp, dict):
        bl_comp = {}
    bl_gm = bl_comp.get('geometric_mean')
    if bl_gm is None:
        # Try nested sklearn path
        bl_sk = bl_comp.get('sklearn', {})
        if isinstance(bl_sk, dict):
            bl_gm = bl_sk.get('geometric_mean')
    row['baseline_composite'] = num_or_blank(bl_gm)

    # ---- Model config ----
    mc = d.get('model_config', {})
    if not isinstance(mc, dict):
        mc = {}
    row['lr'] = num_or_blank(mc.get('learning_rate'))
    row['config_patience'] = num_or_blank(mc.get('patience'))
    row['pretrain_epochs'] = num_or_blank(mc.get('pretrain_epochs'))
    row['config_total_epochs'] = num_or_blank(mc.get('total_epochs'))

    # ---- Computed composite ----
    row['composite'] = compute_composite(row.get('eval_q1_sil'),
                                         row.get('train_gev'))

    return row


# ---- Sweep directory parsing ------------------------------------------------

def parse_sweep_dir(participant, sweep_dir):
    """Parse all cluster_* dirs in sweep_dir, return (rows, missing, skipped)."""
    rows = []
    missing = 0
    skipped = 0

    sweep_path = sweep_dir
    if not os.path.isdir(sweep_path):
        sys.stderr.write("ERROR: Sweep dir does not exist: %s\n" % sweep_path)
        return rows, missing, skipped

    entries = sorted(os.listdir(sweep_path))
    for name in entries:
        full = os.path.join(sweep_path, name)
        if not os.path.isdir(full) or not name.startswith('cluster_'):
            continue

        params = parse_folder_name(name)
        if params is None:
            skipped += 1
            continue

        json_path = os.path.join(full, 'best_model_metrics.json')
        if not os.path.isfile(json_path):
            missing += 1
            continue

        row = extract_one_config(json_path, participant, name, params)
        if row is None:
            missing += 1
            continue

        rows.append(row)

    return rows, missing, skipped


def sort_rows(rows):
    """Sort by K ascending, then composite descending."""
    def sort_key(r):
        k = r.get('K', 0)
        c = r.get('composite', '')
        if c == '':
            c = -999.0
        else:
            c = float(c)
        return (k, -c)
    rows.sort(key=sort_key)


def write_csv(rows, output_path, columns):
    """Write rows to CSV with given column order."""
    with open(str(output_path), 'w') as f:
        # Use newline='' on Python 3 to avoid blank lines on Windows
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore',
                                lineterminator='\n')
        writer.writeheader()
        for row in rows:
            # Ensure all columns exist
            out = {}
            for col in columns:
                val = row.get(col, '')
                out[col] = val
            writer.writerow(out)


def print_summary(rows, missing, skipped, output_path, participant):
    """Print extraction summary to stderr."""
    sys.stderr.write("Parsed: %d configs | Missing JSON: %d | Skipped: %d\n"
                     % (len(rows), missing, skipped))

    # Top 3 by composite
    scored = []
    for r in rows:
        c = r.get('composite', '')
        if c != '':
            scored.append((float(c), r))
    scored.sort(key=lambda x: -x[0])

    if scored:
        sys.stderr.write("Top 3 by composite:\n")
        for comp, r in scored[:3]:
            sys.stderr.write(
                "  K=%-2s  ld=%-2s d=%-1s ndf=%-3s  composite=%.3f  "
                "train_gev=%s  eval_q1_sil=%s\n"
                % (r.get('K', '?'), r.get('latent_dim', '?'),
                   r.get('depth', '?'), r.get('ndf', '?'),
                   comp,
                   r.get('train_gev', 'N/A'),
                   r.get('eval_q1_sil', 'N/A'))
            )

    sys.stderr.write("Output: %s (%d rows, %d cols)\n"
                     % (output_path, len(rows), len(COLUMNS)))


# ---- Merge mode -------------------------------------------------------------

def merge_csvs(csv_paths, output_path):
    """Merge multiple CSVs, union-ing fieldnames, filling missing with ''."""
    all_rows = []
    all_fields_ordered = []
    seen_fields = set()

    for csv_path in csv_paths:
        try:
            with open(str(csv_path), 'r') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for fn in reader.fieldnames:
                        if fn not in seen_fields:
                            all_fields_ordered.append(fn)
                            seen_fields.add(fn)
                for row in reader:
                    all_rows.append(row)
        except (IOError, OSError) as e:
            sys.stderr.write("WARNING: Could not read %s: %s\n"
                             % (csv_path, e))

    if not all_rows:
        sys.stderr.write("ERROR: No rows found in any input CSV\n")
        sys.exit(1)

    # Fill missing fields with ''
    with open(str(output_path), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields_ordered,
                                extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in all_rows:
            out = {}
            for fn in all_fields_ordered:
                out[fn] = row.get(fn, '')
            writer.writerow(out)

    sys.stderr.write("Merged %d rows from %d files -> %s (%d cols)\n"
                     % (len(all_rows), len(csv_paths), output_path,
                        len(all_fields_ordered)))


# ---- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract comprehensive CSV from 4D sweep results')

    # Single-participant mode
    parser.add_argument('--participant', type=str,
                        help='Participant/subject ID (e.g. 010012)')
    parser.add_argument('--sweep-dir', type=str,
                        help='Path to sweep directory (e.g. outputs/010012/sweep_4d_010012)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: {participant}_sweep.csv)')

    # Merge mode
    parser.add_argument('--merge', nargs='+', type=str, default=None,
                        help='Merge multiple CSVs into one')

    args = parser.parse_args()

    # ---- Merge mode ----
    if args.merge:
        if not args.output:
            sys.stderr.write("ERROR: --output required with --merge\n")
            sys.exit(1)
        merge_csvs(args.merge, args.output)
        return

    # ---- Single-participant mode ----
    if not args.participant or not args.sweep_dir:
        parser.error("--participant and --sweep-dir required (or use --merge)")

    output = args.output
    if output is None:
        output = '%s_sweep.csv' % args.participant

    rows, missing, skipped = parse_sweep_dir(args.participant, args.sweep_dir)

    if not rows:
        sys.stderr.write("ERROR: No valid configs found in %s\n"
                         % args.sweep_dir)
        sys.exit(1)

    sort_rows(rows)
    write_csv(rows, output, COLUMNS)
    print_summary(rows, missing, skipped, output, args.participant)


if __name__ == '__main__':
    main()
