from pathlib import Path
import importlib.util

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('prototype_audit', Path(__file__).parents[1] / 'scripts/diagnose_r2_prototypes.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_source_support_is_disjoint_and_balanced_by_subject_class():
    keys = [(s, 1, c * 4 + j) for s in (1, 2) for c in range(3) for j in range(4)]
    labels = [c for s in (1, 2) for c in range(3) for j in range(4)]
    a, b = audit.source_design(keys, labels)
    assert (a, b) == audit.source_design(keys, labels)
    assert not set(a) & set(b)
    assert len(a) == 12 and len(b) == 6
    for s in (1, 2):
        for c in range(3):
            assert sum(keys[i][0] == s and labels[i] == c for i in a) == 2
            assert sum(keys[i][0] == s and labels[i] == c for i in b) == 1


def test_matched_refresh_keeps_assignments_and_empty_slot_reference():
    memory = torch.tensor([[[[1., 0.], [0., 1.]]]])
    initialized = torch.ones(1, 1, 2, dtype=torch.bool)
    zold = torch.tensor([[[1., .1]], [[1., .2]]])
    labels = torch.zeros(2, dtype=torch.long)
    assignment = audit.assign_slots(zold, labels, memory, initialized)
    znew = zold.flip(-1)
    old, old_counts = audit.rebuild(zold, labels, assignment, memory)
    new, new_counts = audit.rebuild(znew, labels, assignment, memory)
    assert torch.equal(old_counts, new_counts)
    assert old_counts.tolist() == [[[2, 0]]]
    assert torch.equal(old[0, 0, 1], memory[0, 0, 1])
    assert torch.equal(new[0, 0, 1], memory[0, 0, 1])
    assert new[0, 0, 0, 1] > new[0, 0, 0, 0]
    # The refreshed vectors would change slots if assignments were recomputed;
    # keeping them frozen is what isolates feature-space drift.
    assert not torch.equal(assignment, audit.assign_slots(znew, labels, memory, initialized))


def test_class_metrics_count_regressions_as_well_as_corrections():
    y, old, new = [0, 1, 2, 2], [1, 1, 2, 0], [0, 2, 2, 2]
    flips = audit.flips(new, old, y)
    assert (flips['wrong_to_correct'], flips['correct_to_wrong'], flips['net_corrected']) == (2, 1, 1)
    m = audit.metrics(new, y)
    assert m['accuracy'] == .75 and m['recall']['neutral'] == 0
    assert m['precision']['neutral'] is None
    assert np.isfinite(m['macro_f1'])


def test_geometry_queries_all_classes_without_target_routing():
    memory = torch.eye(3).reshape(1, 3, 1, 3)
    mask = torch.ones(1, 3, 1, dtype=torch.bool)
    z = torch.tensor([[[0., 1., .1]], [[.1, 0., 1.]]])
    result = audit.geometry(z, memory, mask)
    assert result['prediction'].tolist() == [1, 2]
    checked = audit.geometry(z, memory, mask, [0, 0])
    torch.testing.assert_close(result['scores'], checked['scores'])
    assert checked['metrics']['accuracy'] == 0
