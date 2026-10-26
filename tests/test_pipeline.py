"""Synthetic checks; no private data or thesis outputs are required."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import nanopore_project as p


def synthetic_fast5(path, read_id, offset=0):
    """Write DNA FAST5 with mock successful Tombo annotations."""
    with h5py.File(path, 'w') as f:
        raw = f.create_group('Raw/Reads/Read_0')
        raw.attrs['read_id'] = read_id
        bases = np.resize(np.array(list('ACGT')), 200)
        signal = np.repeat(np.resize([60, 75, 90, 105], 200), 20)
        if offset:
            signal = np.pad(signal, (offset, offset))
        raw.create_dataset('Signal', data=signal.astype('i2'))
        channel = f.create_group('UniqueGlobalKey/channel_id')
        channel.attrs.update(offset=0., range=1., digitisation=1.)
        corrected = f.create_group('Analyses/RawGenomeCorrected_000/BaseCalled_template')
        corrected.attrs.update(status='success', rna=False)
        events = np.zeros(200, dtype=[('start', 'u4'), ('length', 'u4'), ('base', 'S1')])
        events['start'] = np.arange(200) * 20
        events['length'] = 20
        events['base'] = bases.astype('S1')
        slot = corrected.create_dataset('Events', data=events)
        slot.attrs['read_start_rel_to_raw'] = offset


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ctc_repeats_and_blank(self):
        self.assertEqual(p.decode([1, 1, 0, 1]), 'AA')
        self.assertEqual(p.decode([0, 0]), '')
        self.assertEqual(p.decode([1, 1, 0, 2, 2, 0, 4]), 'ACT')

    def test_filters_preserve_length_and_finiteness(self):
        x = np.random.default_rng(7).normal(80, 4, 4000)
        for method in p.METHODS:
            with self.subTest(method=method):
                y = p.preprocess(x, method)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(np.isfinite(y).all())
        np.testing.assert_array_equal(p.preprocess(np.ones(4000), 'robust'), 0)

    def test_event_offset_and_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'read.fast5'
            synthetic_fast5(path, 'offset_read', offset=40)
            rid, signal, starts, lengths, labels = p.extract_read(
                path, 'RawGenomeCorrected_000/BaseCalled_template')
            self.assertEqual(rid, 'offset_read')
            self.assertEqual(starts[0], 40)
            self.assertEqual(signal[40], 60)
            chunks = p.valid_chunks(signal, starts, lengths, labels)
            self.assertEqual(chunks[:, 0].tolist(), [2000])

    def test_ctc_impossible_target_is_excluded(self):
        signal = np.ones(2000)
        starts = np.arange(0, 2000, 5)
        lengths = np.full(400, 5)
        # 400 identical letters need 799 CTC steps, more than 500.
        labels = np.ones(400, dtype=np.int64)
        self.assertEqual(len(p.valid_chunks(signal, starts, lengths, labels)), 0)

    def test_model_shape_and_finite_loss_gradient(self):
        p.seed_all()
        model = p.Basecaller()
        output = model(torch.randn(2, 1, 2000))
        self.assertEqual(tuple(output.shape), (500, 2, 5))
        targets = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        loss = torch.nn.CTCLoss()(output, targets, torch.tensor([500, 500]),
                                  torch.tensor([4, 4]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(v.grad).all()
                            for v in model.parameters() if v.grad is not None))

    def test_read_split_and_six_training_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fast5 = root / 'fast5'
            fast5.mkdir()
            for i in range(20):
                synthetic_fast5(fast5 / f'{i}.fast5', f'synthetic_{i:03d}')
            work = root / 'dataset'
            p.prepare(fast5, work, 'RawGenomeCorrected_000/BaseCalled_template')
            manifest = json.loads((work / 'manifest.json').read_text())
            splits = [set(manifest['split'][s])
                      for s in ['train', 'validation', 'test']]
            self.assertEqual([len(s) for s in splits], [14, 3, 3])
            self.assertFalse(splits[0] & splits[1])
            self.assertFalse(splits[0] & splits[2])
            self.assertFalse(splits[1] & splits[2])
            args = SimpleNamespace(batch=8, workers=0, epochs=1)
            for method in p.METHODS:
                with self.subTest(method=method):
                    output = root / method
                    loss = p.fit(work, manifest, method, .2, 5, args, output)
                    self.assertTrue(np.isfinite(loss))
                    self.assertTrue((output / 'best.pt').is_file())
                    if importlib.util.find_spec('edlib') is not None:
                        stats = p.evaluate(work, manifest, method, args, output)
                        self.assertEqual(stats['reads'], 3)
                        self.assertEqual(stats['chunks'], 6)
                        self.assertAlmostEqual(stats['identity'] + stats['error'], 100)

    @unittest.skipUnless(importlib.util.find_spec('edlib'), 'edlib is not installed')
    def test_edlib_insertion_deletion_direction(self):
        self.assertEqual(p.align_counts('ACGT', 'AGT').tolist(), [3, 0, 0, 1])
        self.assertEqual(p.align_counts('ACGT', 'ACGGT').tolist(), [4, 0, 1, 0])
        self.assertEqual(p.align_counts('ACGT', 'ACAT').tolist(), [3, 1, 0, 0])
        self.assertEqual(p.align_counts('ACGT', '').tolist(), [0, 0, 0, 4])

    def test_bootstrap_pairs_ids_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, method in enumerate(p.METHODS):
                folder = root / method
                folder.mkdir()
                ids = np.array(['r3', 'r1', 'r2'])
                counts = np.array([[70+i, 10, 5, 15-i],
                                   [80+i, 5, 5, 10-i], [75+i, 7, 8, 10-i]])
                if i % 2:
                    ids, counts = ids[::-1], counts[::-1]
                np.savez(folder / 'read_counts.npz', ids=ids, counts=counts)
            p.bootstrap(root, 100)
            report = json.loads((root / 'bootstrap.json').read_text())
            interval = report['results']['robust_kalman']['difference_from_robust_ci95']
            np.testing.assert_allclose(interval, [3, 3])
            np.savez(root / 'raw/read_counts.npz',
                     ids=np.array(['r1']), counts=np.array([[1, 0, 0, 0]]))
            with self.assertRaises(ValueError):
                p.bootstrap(root, 10)


if __name__ == '__main__':
    unittest.main()
