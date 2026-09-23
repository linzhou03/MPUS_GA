"""Paired supervised mechanism study; not a new UDA method."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .pcdiag import Audit, cpu, oracle_policy, save_json
from ..scripts.probe_r2_replay import tree_differences

MODES = ('reference', 'R2', 'P', 'C', 'PC', 'C_truth', 'PC_truth')
FOCUS = {'B': 2, 'C': 1, 'E': 1}
PROTOCOL = {
    'version': 'r2_paired_oracle_20260915_v1', 'directions': ['B', 'C', 'E'],
    'subjects': [1, 2, 3, 4, 5], 'seed': 42, 'iterations': 1000, 'fork_step': 300,
    'core_conditions': ['R2', 'P', 'C', 'PC'], 'extra_E_conditions': ['C_truth', 'PC_truth'],
    'coverage_budget_per_batch': 4, 'added_effective_weight': .25,
    'P': 'Correct original accepted labels with truth; retain original mask and effective weights.',
    'C': 'Keep original accepted labels; add up to 4 unique originally-correct rejected trials, rotating classes.',
    'PC': 'Apply P and exactly the same additions as C.',
    'C_truth': 'E only: add up to 4 unique rejected true-Neutral trials, even if predicted wrong; assign truth.',
    'PC_truth': 'Apply P and exactly the same additions as C_truth.',
    'reference': 'New deterministic original-R2 run; fresh optimizer checkpoint for every fold including E.',
    'R2': 'Replay reference steps 301-1000 exactly; gate every prediction/loss/final state before Oracle.',
    'selection': 'No target metric selects thresholds, checkpoints, subjects, or branches.',
    'limitations': ['Target truth is used by Oracle branches; these are not UDA results.',
                   'P fixes count globally, but changes class allocation and correct coverage; factors are not orthogonal.',
                   'One seed and five paired subjects per direction are exploratory, not confirmatory.'],
}


def configure_determinism():
    import os
    if torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def truth_coverage_policy(reference, truth, mode, iteration, focus=1, budget=4):
    if mode not in ('C_truth', 'PC_truth'):
        raise ValueError(mode)
    labels, accepted, weights, _ = oracle_policy(reference, truth, 'P' if mode == 'PC_truth' else 'R2', iteration)
    candidates = ~reference['valid_mask'] & (truth == focus)
    ids = reference['target_ids'].tolist()
    # Rank by probability of the true focus class, not max confidence in a wrong class.
    rows = sorted(candidates.nonzero().flatten().tolist(),
                  key=lambda i: (-float(reference['probability'][i, focus]), ids[i], i))
    added, seen = torch.zeros_like(accepted), set()
    for i in rows:
        key = tuple(ids[i])
        if key in seen:
            continue
        seen.add(key); added[i] = True
        if len(seen) >= budget:
            break
    accepted |= added
    labels[added], weights[added] = truth[added], .25
    return labels, accepted, weights, added


class StudyAudit(Audit):
    checkpoint_optimizer = True

    def __init__(self, directory, condition, baseline_directory=None, **kwargs):
        if condition not in MODES:
            raise ValueError(condition)
        self.condition = condition
        super().__init__(directory, 'baseline' if condition == 'reference' else 'replay' if condition == 'R2' else condition,
                         baseline_directory, **kwargs)

    def ready(self):
        super().ready()
        if self.start_iteration:
            self.snapshot(self.start_iteration)
            checkpoint = self.baseline_directory / f'step_{self.start_iteration:04d}.pt'
            save_json(self.directory / 'fork.json', {
                'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'step': self.start_iteration, 'condition': self.condition,
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()})

    def policy(self, reference, truth, iteration):
        if self.condition in ('C_truth', 'PC_truth'):
            return truth_coverage_policy(reference, truth, self.condition, iteration)
        return oracle_policy(reference, truth, self.condition, iteration)

    def compare_evidence(self, iteration, key, actual, expected):
        # A new matched control must be bitwise equal. Do not widen the old tolerance.
        try:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        except AssertionError as exc:
            save_json(self.directory / 'gate_failure.json', {'step': iteration, 'key': key, 'error': str(exc)})
            raise AssertionError(f'Exact R2 replay failed at {iteration}, {key}: {exc}') from exc

    def capture(self, iteration, source_batches, target_batch, output, consensus, active, source_labels, threshold):
        live = cpu(asdict(consensus))
        result = super().capture(iteration, source_batches, target_batch, output, consensus, active, source_labels, threshold)
        record = self.pending[-1]
        record['condition'] = self.condition
        record['live_pseudo_label'] = live['pseudo_label']
        record['live_valid_mask'] = live['valid_mask']
        if self.condition != 'reference':
            ref = self.reference(iteration)
            record['baseline_effective_weight'] = ref['effective_weight']
            # These checks concern policy invariants only, never target performance.
            if self.condition in ('P', 'R2'):
                assert torch.equal(record['valid_mask'], ref['valid_mask'])
                assert torch.equal(record['effective_weight'], ref['effective_weight'])
        return result

    def after_step(self, iteration, record):
        if self.condition == 'R2':
            expected = self.reference(iteration)['losses']
            for key in expected:
                if record[key] != expected[key]:
                    save_json(self.directory / 'gate_failure.json', {'step': iteration, 'key': 'loss.' + key,
                              'actual': record[key], 'expected': expected[key]})
                    raise AssertionError(f'Exact replay loss failed {iteration} {key}: {record[key]} != {expected[key]}')
        super().after_step(iteration, record)

    def finish(self, result, final_evidence):
        if self.condition == 'R2':
            filename = f'step_{self.nodes[-1]:04d}.pt'
            a = torch.load(self.directory / filename, map_location='cpu', weights_only=False)
            b = torch.load(self.baseline_directory / filename, map_location='cpu', weights_only=False)
            keys = ('model', 'objects', 'optimizer', 'scheduler', 'rng')
            diffs = tree_differences({k: a[k] for k in keys}, {k: b[k] for k in keys})
            reference_result = json.loads(self.baseline_directory.with_suffix('.json').read_text())
            if diffs or result['evaluation'] != reference_result['evaluation']:
                save_json(self.directory / 'gate_failure.json', {'final_state_differences': diffs,
                    'evaluation_equal': result['evaluation'] == reference_result['evaluation']})
                raise AssertionError('Final R2 state/evaluation replay failed')
        super().finish(result, final_evidence)
        result['method'] = 'original_r2_paired_control' if self.condition in ('reference', 'R2') else 'supervised_mechanism_oracle_' + self.condition
        result['study_protocol'] = PROTOCOL
        result['diagnostic']['condition'] = self.condition
        result['diagnostic']['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
        save_json(self.directory / 'complete.json', result['diagnostic'])
        if self.condition == 'R2':
            save_json(self.directory / 'exact_replay_passed.json', {
                'bitwise_equal': True, 'steps_checked': self.nodes[-1] - self.start_iteration,
                'checked': ['all consensus fields', 'four training losses', *keys, 'final evaluation'],
                'reference': str(self.baseline_directory)})
