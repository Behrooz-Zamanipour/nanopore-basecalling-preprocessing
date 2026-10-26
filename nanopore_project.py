"""Reconstructed pipeline; not the lost historical Google Cloud source."""
import argparse
import csv
import hashlib
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pywt
import torch
from scipy.signal import lfilter, savgol_filter
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

METHODS = ['raw', 'robust', 'wavelet', 'robust_wavelet',
           'robust_kalman', 'robust_savgol']
LETTERS = '_ACGT'
LENGTH = 2000
TIME_STEPS = 500


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                          encoding='utf-8')


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def extract_read(path, group):
    with h5py.File(path, 'r') as f:
        reads = list(f['Raw/Reads'].values())
        if len(reads) != 1:
            raise ValueError('Expected a single-read FAST5')
        raw = reads[0]
        read_id = text(raw.attrs['read_id'])
        channel = f['UniqueGlobalKey/channel_id'].attrs
        scale = float(channel['range']) / float(channel['digitisation'])
        signal = (raw['Signal'][:].astype(np.float64) +
                  float(channel['offset'])) * scale
        corrected = f['Analyses/' + group]
        if text(corrected.attrs.get('status', '')) != 'success':
            raise ValueError('Tombo alignment was not successful')
        if bool(corrected.attrs.get('rna', False)):
            raise ValueError('This reconstruction supports DNA only')
        event_slot = corrected['Events']
        events = event_slot[:]
        start = events['start'].astype(np.int64)
        start += int(event_slot.attrs['read_start_rel_to_raw'])
        lengths = events['length'].astype(np.int64)
        bases = b''.join(events['base'].tolist()).decode().upper()
        if not len(start) or len(bases) != len(start):
            raise ValueError('Empty or inconsistent alignment')
        if (np.any(lengths <= 0) or np.any(np.diff(start) <= 0) or
                start[0] < 0 or np.max(start + lengths) > len(signal)):
            raise ValueError('Invalid event coordinates')
        if set(bases) - set('ACGT') or not np.isfinite(signal).all():
            raise ValueError('Noncanonical base or invalid signal')
        labels = np.array([LETTERS.index(c) for c in bases], dtype=np.int64)
        # Tombo Events are already in read/signal order, including minus strand.
        # Do not reverse-complement them a second time.
        return read_id, signal.astype(np.float32), start, lengths, labels


def valid_chunks(signal, starts, lengths, labels):
    rows = []
    aligned_end = int(starts[-1] + lengths[-1])
    for s in range(0, len(signal) - LENGTH + 1, LENGTH):
        # Reconstruction choice: exclude partly unaligned adapter/end windows.
        if s < starts[0] or s + LENGTH > aligned_end:
            continue
        lo, hi = np.searchsorted(starts, [s, s + LENGTH], side='left')
        target = labels[lo:hi]
        repeats = np.count_nonzero(target[1:] == target[:-1])
        if len(target) and len(target) + repeats <= TIME_STEPS:
            rows.append((s, int(lo), int(hi)))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def prepare(fast5_dir, work, group):
    work.mkdir(parents=True, exist_ok=False)
    read_dir = work / 'reads'
    read_dir.mkdir()
    records, seen, reasons = [], set(), Counter()
    with (work / 'qc.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file', 'status', 'reason'])
        for path in sorted(Path(fast5_dir).rglob('*.fast5')):
            try:
                rid, signal, starts, lengths, labels = extract_read(path, group)
                if rid in seen:
                    raise ValueError('Duplicate read ID')
                seen.add(rid)
                chunks = valid_chunks(signal, starts, lengths, labels)
                if not len(chunks):
                    raise ValueError('No valid full chunks')
                name = hashlib.sha256(rid.encode()).hexdigest() + '.npz'
                np.savez(read_dir / name, signal=signal, labels=labels,
                         chunks=chunks)
                records.append(dict(id=rid, file='reads/' + name,
                                    chunks=len(chunks)))
                writer.writerow([str(path), 'accepted', ''])
            except (OSError, KeyError, ValueError, UnicodeError) as error:
                reason = str(error)
                reasons[reason] += 1
                writer.writerow([str(path), 'rejected', reason])
    if len(records) < 20:
        raise ValueError('Need at least 20 accepted reads for three splits')
    records.sort(key=lambda r: r['id'])
    order = np.random.default_rng(42).permutation(len(records))
    ntrain, nval = int(.70 * len(order)), int(.15 * len(order))
    split = dict(train=order[:ntrain].tolist(),
                 validation=order[ntrain:ntrain+nval].tolist(),
                 test=order[ntrain+nval:].tolist())
    fingerprint = hashlib.sha256('\n'.join(r['id'] for r in records).encode())
    save_json(work / 'manifest.json', dict(
        records=records, split=split, fingerprint=fingerprint.hexdigest(),
        provenance='Reconstructed code; actual inputs; no historical outputs',
        choices=dict(chunk_length=LENGTH, target_boundary='event start in chunk',
                     exclude_partial_alignment=True, seed=42)))
    save_json(work / 'qc_summary.json', dict(
        accepted=len(records), chunks=sum(r['chunks'] for r in records),
        rejected=dict(reasons)))
    print('Prepared reads:', len(records), 'chunks:', sum(r['chunks'] for r in records))


def preprocess(x, method, q=.20, window=5):
    x = np.asarray(x, dtype=np.float64)
    if method.startswith('robust'):
        middle = np.median(x)
        mad = np.median(np.abs(x - middle))
        x = (x - middle) / max(float(mad), 1e-8)
    if method in ('wavelet', 'robust_wavelet'):
        parts = pywt.wavedec(x, 'db4', level=3, mode='symmetric')
        sigma = np.median(np.abs(parts[-1])) / .6745
        threshold = sigma * np.sqrt(2 * np.log(len(x)))
        parts = [parts[0]] + [pywt.threshold(p, threshold, mode='soft')
                               for p in parts[1:]]
        x = pywt.waverec(parts, 'db4', mode='symmetric')[:len(x)]
    elif method == 'robust_kalman':
        # Scalar random-walk Kalman filter initialized at steady covariance.
        # Initialization was not specified in the thesis; this choice is explicit.
        gain = (np.sqrt(q*q + 4*q) - q) / 2
        x, _ = lfilter([gain], [1, -(1-gain)], x,
                       zi=[(1-gain)*x[0]])
    elif method == 'robust_savgol':
        x = savgol_filter(x, window, 2, mode='interp')
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite preprocessed signal')
    return x.astype(np.float32)


class ReadChunks(IterableDataset):
    def __init__(self, work, records, indices, method, q, window,
                 training=False, epoch=0):
        self.work, self.records, self.indices = work, records, list(indices)
        self.method, self.q, self.window = method, q, window
        self.training, self.epoch = training, epoch

    def __iter__(self):
        indices = self.indices.copy()
        rng = np.random.default_rng(42 + self.epoch)
        if self.training:
            rng.shuffle(indices)
        worker = get_worker_info()
        if worker:
            indices = indices[worker.id::worker.num_workers]
        for idx in indices:
            record = self.records[idx]
            with np.load(self.work / record['file'], allow_pickle=False) as z:
                signal = preprocess(z['signal'], self.method, self.q, self.window)
                labels, chunks = z['labels'], z['chunks']
            rows = np.arange(len(chunks))
            if self.training:
                rng.shuffle(rows)
            for j in rows:
                start, lo, hi = chunks[j]
                yield (signal[start:start+LENGTH].copy(), labels[lo:hi].copy(),
                       record['id'], int(start))


def collate(batch):
    x, targets, ids, starts = zip(*batch)
    return (torch.from_numpy(np.stack(x))[:, None, :],
            torch.from_numpy(np.concatenate(targets)),
            torch.tensor([len(t) for t in targets], dtype=torch.long),
            ids, starts)


def loader(work, manifest, indices, method, q, window, args,
           training=False, epoch=0):
    data = ReadChunks(work, manifest['records'], indices, method, q, window,
                      training, epoch)
    # Evaluation uses zero workers to keep read/chunk order unambiguous.
    return DataLoader(data, batch_size=args.batch,
                      num_workers=args.workers if training else 0,
                      collate_fn=collate, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(42 + epoch))


class Basecaller(nn.Module):
    def __init__(self):
        super().__init__()
        # ReLU is a reconstruction choice; activation name was not specified.
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 11, 2, 5), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 7, 2, 3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 192, 5, 1, 2), nn.BatchNorm1d(192), nn.ReLU(),
            nn.Dropout(.1), nn.Conv1d(192, 256, 5, 1, 2), nn.ReLU(),
            nn.Conv1d(256, 5, 1))

    def forward(self, x):
        return self.net(x).permute(2, 0, 1).log_softmax(-1)


def run_epoch(model, batches, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=not training)
    total, count = 0., 0
    with torch.set_grad_enabled(training):
        for x, target, lengths, _, _ in batches:
            x, target = x.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(x)
            in_lengths = torch.full((len(x),), output.shape[0], dtype=torch.long)
            loss = loss_fn(output, target, in_lengths, lengths)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss; inspect data and CTC targets')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
    if not count:
        raise ValueError('Empty data loader')
    return total / count


def fit(work, manifest, method, q, window, args, output, epochs=None,
        train_ids=None, val_ids=None):
    seed_all()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Basecaller().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, stale, history = float('inf'), 0, []
    train_ids = manifest['split']['train'] if train_ids is None else train_ids
    val_ids = manifest['split']['validation'] if val_ids is None else val_ids
    output.mkdir(parents=True, exist_ok=False)
    for epoch in range(epochs or args.epochs):
        train_loss = run_epoch(model, loader(work, manifest, train_ids, method,
                               q, window, args, True, epoch), device, optimizer)
        val_loss = run_epoch(model, loader(work, manifest, val_ids, method,
                             q, window, args), device)
        row = dict(epoch=epoch+1, train_loss=train_loss, validation_loss=val_loss)
        history.append(row)
        print(method, row, flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(dict(model=model.state_dict(), epoch=epoch+1,
                            method=method, q=q, window=window,
                            fingerprint=manifest['fingerprint']), output / 'best.pt')
        else:
            stale += 1
        save_json(output / 'history.json', history)
        if stale >= 3:
            break
    return best


def tune(work, manifest, args, root):
    # Fixed read subsets; test split is never used here.
    train = manifest['split']['train'][:args.pilot_train_reads]
    val = manifest['split']['validation'][:args.pilot_val_reads]
    rows = []
    selected = {}
    for method, values in [('robust_kalman', [.01, .05, .20, .50, 1.]),
                           ('robust_savgol', [5, 9, 15])]:
        candidates = []
        for value in values:
            q = value if method == 'robust_kalman' else .20
            window = value if method == 'robust_savgol' else 5
            loss = fit(work, manifest, method, q, window, args,
                       root / f'{method}_{value}', args.pilot_epochs, train, val)
            candidates.append((loss, value))
            rows.append(dict(method=method, parameter=value, validation_loss=loss))
        selected[method] = min(candidates)[1]
    save_json(root / 'selection.json', dict(results=rows, selected=selected,
              note='Pilot subset sizes are reconstruction choices'))
    return selected['robust_kalman'], selected['robust_savgol']


def decode(path):
    return ''.join(LETTERS[int(k)] for k, _ in itertools.groupby(path) if k != 0)


def align_counts(reference, prediction):
    if not prediction:
        return np.array([0, 0, 0, len(reference)], dtype=np.int64)
    if not reference:
        return np.array([0, 0, len(prediction), 0], dtype=np.int64)
    import edlib
    result = edlib.align(prediction, reference, mode='NW', task='path')
    cigar = result['cigar']
    if cigar is None:
        raise ValueError('Global alignment failed')
    counts = dict.fromkeys('=XID', 0)
    for n, operation in re.findall(r'(\d+)([=XID])', cigar):
        counts[operation] += int(n)
    return np.array([counts[c] for c in '=XID'], dtype=np.int64)


def percentages(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        raise ValueError('Empty alignment counts')
    return dict(zip(['identity', 'substitution', 'insertion', 'deletion'],
                    (100 * counts / counts.sum()).tolist()))


def evaluate(work, manifest, method, args, output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    if ckpt['fingerprint'] != manifest['fingerprint'] or ckpt['method'] != method:
        raise ValueError('Checkpoint and dataset/method mismatch')
    model = Basecaller().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ids, counts, current, refs, preds = [], [], None, [], []
    totals = np.zeros(4, dtype=np.int64)
    expected = {manifest['records'][i]['id']: manifest['records'][i]['chunks']
                for i in manifest['split']['test']}
    observed = Counter()
    fasta = (output / 'predictions.fasta').open('w', encoding='utf-8')

    def finish():
        nonlocal totals
        if current is not None:
            ref, pred = ''.join(refs), ''.join(preds)
            c = align_counts(ref, pred)
            ids.append(current)
            counts.append(c)
            totals += c
            fasta.write(f'>{current}\n{pred}\n')

    try:
        with torch.no_grad():
            batches = loader(work, manifest, manifest['split']['test'], method,
                             ckpt['q'], ckpt['window'], args)
            for x, y, lengths, read_ids, starts in batches:
                paths = model(x.to(device)).argmax(-1).cpu().numpy().T
                offset = 0
                for path, length, rid, start in zip(paths, lengths.tolist(),
                                                    read_ids, starts):
                    if rid != current:
                        finish()
                        current, refs, preds = rid, [], []
                        previous_start = -1
                    if start <= previous_start:
                        raise ValueError('Out-of-order read chunks')
                    previous_start = start
                    observed[rid] += 1
                    refs.append(''.join(LETTERS[k] for k in y[offset:offset+length]))
                    preds.append(decode(path))
                    offset += length
        finish()
    finally:
        fasta.close()
    if dict(observed) != expected:
        raise ValueError('Missing or duplicated test chunks')
    np.savez(output / 'read_counts.npz', ids=np.array(ids), counts=np.stack(counts))
    stats = percentages(totals)
    stats['error'] = 100 - stats['identity']
    stats.update(reads=len(ids), chunks=sum(observed.values()))
    save_json(output / 'metrics.json', stats)
    print(method, stats, flush=True)
    return stats


def bootstrap(root, repeats=5000):
    arrays, ids = [], None
    for method in METHODS:
        with np.load(root / method / 'read_counts.npz', allow_pickle=False) as z:
            order = np.argsort(z['ids'])
            current_ids = z['ids'][order]
            if ids is not None and not np.array_equal(ids, current_ids):
                raise ValueError('Methods must have identical read IDs')
            ids = current_ids
            arrays.append(z['counts'][order])
    data = np.stack(arrays)
    rng = np.random.default_rng(20260908)
    samples = np.empty((repeats, len(METHODS)))
    for b in range(repeats):
        selected = rng.integers(0, len(ids), len(ids))
        total = data[:, selected, :].sum(axis=1)
        samples[b] = 100 * total[:, 0] / total.sum(axis=1)
    rows = {}
    for i, method in enumerate(METHODS):
        rows[method] = dict(
            identity_ci95=np.percentile(samples[:, i], [2.5, 97.5]).tolist(),
            difference_from_robust_ci95=np.percentile(
                samples[:, i] - samples[:, 1], [2.5, 97.5]).tolist())
    save_json(root / 'bootstrap.json', dict(repeats=repeats, seed=20260908,
              unit='read', difference_unit='percentage points', results=rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'bootstrap'])
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--fast5', type=Path)
    p.add_argument('--group', default='RawGenomeCorrected_000/BaseCalled_template')
    p.add_argument('--run-name', default='experiment_01')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--tune', action='store_true')
    p.add_argument('--pilot-train-reads', type=int, default=1000)
    p.add_argument('--pilot-val-reads', type=int, default=200)
    p.add_argument('--pilot-epochs', type=int, default=1)
    p.add_argument('--bootstrap-repeats', type=int, default=5000)
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        p.error('Invalid training settings')
    if min(args.pilot_epochs, args.pilot_train_reads, args.pilot_val_reads,
           args.bootstrap_repeats) < 1:
        p.error('Pilot sizes and bootstrap repeats must be positive')
    if Path(args.run_name).name != args.run_name:
        p.error('run-name must be a single directory name')
    if args.command == 'prepare':
        if not args.fast5 or not args.fast5.is_dir():
            p.error('Provide an existing --fast5 directory')
        prepare(args.fast5, args.work, args.group)
        return
    manifest = json.loads((args.work / 'manifest.json').read_text(encoding='utf-8'))
    root = args.work / args.run_name
    if args.command == 'bootstrap':
        bootstrap(root, args.bootstrap_repeats)
        return
    root.mkdir(exist_ok=False)
    save_json(root / 'settings.json', dict(
        arguments={k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        torch=torch.__version__, numpy=np.__version__,
        cuda=torch.cuda.is_available(), fingerprint=manifest['fingerprint'],
        kalman_initialization='steady covariance', activation='ReLU'))
    q, window = .20, 5
    if args.tune:
        q, window = tune(args.work, manifest, args, root / 'pilot')
    results = {}
    # All model selection/training finishes before the test set is evaluated.
    for method in METHODS:
        fit(args.work, manifest, method, q, window, args, root / method)
    for method in METHODS:
        results[method] = evaluate(args.work, manifest, method, args, root / method)
    save_json(root / 'summary.json', results)
    bootstrap(root, args.bootstrap_repeats)


if __name__ == '__main__':
    main()
