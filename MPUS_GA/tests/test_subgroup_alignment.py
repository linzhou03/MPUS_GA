"""CPU regression tests for selective alignment and the R2 integration."""

from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MPUS_GA.scripts.run_r2_subgroup_suite import GPU_EXPERIMENTS, command
from MPUS_GA.tests.test_training import _small_model
from MPUS_GA.trial_temporal.subgroup_alignment import (
    SelectiveSubgroupAlignment, SubgroupBank, SubgroupConfig, mutual_matches,
    selective_prototype_contrast, spherical_clusters, teacher_evidence,
)
from MPUS_GA.trial_temporal.train import (
    EXPERIMENTS, PrototypeBank, SourceMultiPrototypeMemory, build_parser,
    subgroup_config, subgroup_experiment, train_step, validate_args,
)


def test_six_directions_inherit_only_r2_and_roundtrip_config():
    baseline = asdict(EXPERIMENTS["A_R2"])
    changed = {"name", "description", "transfer_direction", "ablation", "source_domains",
               "target_dataset", "target_subject_count", "target_trials",
               "use_subgroup_alignment", "prototype_weight"}
    folds = 0
    for direction in "ABCDEF":
        spec = subgroup_experiment(direction)
        assert spec.source_domains == EXPERIMENTS[direction].source_domains
        assert spec.target_dataset == EXPERIMENTS[direction].target_dataset
        assert spec.use_subgroup_alignment and spec.prototype_weight == 0
        assert all(value == baseline[key] for key, value in asdict(spec).items() if key not in changed)
        config = replace(SubgroupConfig(), k=3, weight=0.07)
        cmd = command("python", direction, "/tmp/data", "/tmp/r2_subgroup_results",
                      [43, 42], "all", config)
        args = build_parser().parse_args(cmd[4:])
        validate_args(args, spec)
        assert args.random_seeds == [43, 42] and args.method == "r2_subgroup"
        assert subgroup_config(args) == config
        folds += len(args.target_subjects) * len(args.random_seeds)
    assert folds == 204
    assert sorted(sum(GPU_EXPERIMENTS, ())) == list("ABCDEF")
    assert not EXPERIMENTS["A_R2"].use_subgroup_alignment
    assert EXPERIMENTS["A_R2"].prototype_weight > 0


def test_teacher_consensus_rejects_uncertain_or_disagreeing_scales():
    logits = torch.tensor([
        [[10., 0., 0.], [10., 0., 0.], [10., 0., 0.]],
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]],
    ], requires_grad=True)
    labels, confidence, valid = teacher_evidence(logits, SubgroupConfig())
    assert valid[0].all() and not valid[1:].any()
    assert labels[0] == 0 and not confidence.requires_grad


def test_clusters_need_unique_support_and_do_not_force_k():
    config = SubgroupConfig()
    vectors = torch.eye(3).repeat_interleave(3, 0)
    prototypes, support, _ = spherical_clusters(vectors, torch.ones(9), config)
    assert len(prototypes) == 3 and support.tolist() == [3, 3, 3]
    assert len(spherical_clusters(vectors[:2], torch.ones(2), config)[0]) == 0
    assert len(spherical_clusters(torch.ones(30, 3), torch.ones(30), config)[0]) == 1
    bank = SubgroupBank(1, 3, 3, config, torch.device("cpu"))
    bank.observe(1, [(0, 1, 1, 1)] * 9, vectors[:, None], torch.zeros(9).long(),
                 torch.ones(9), torch.ones(9, 1).bool(), 299)
    bank.refresh(300)
    assert len(bank.memory[1]) == 1 and not bank.support.any()
    bank.observe(1, [(0, 1, 1, 1)], vectors[:1, None], torch.tensor([2]),
                 torch.ones(1), torch.ones(1, 1).bool(), 301)
    assert bank.memory[1][(0, 1, 1, 1)][1] == 2
    bank.observe(1, [(0, 1, 1, 1)], vectors[:1, None], torch.tensor([2]),
                 torch.ones(1), torch.zeros(1, 1).bool(), 302)
    assert not bank.memory[1]


def test_periodic_snapshot_is_frozen_and_stale_memory_expires():
    config = SubgroupConfig()
    bank = SubgroupBank(1, 3, 3, config, torch.device("cpu"))
    keys = [(0, 1, 1, i) for i in range(3)]
    for domain in (0, 1):
        bank.observe(domain, keys, torch.eye(3)[:1].repeat(3, 1)[:, None],
                     torch.zeros(3).long(), torch.ones(3), torch.ones(3, 1).bool(), 299)
    assert not bank.refresh(299)
    assert bank.refresh(300) and bank.matches.any()
    old = bank.prototypes.clone()
    bank.observe(0, keys, torch.eye(3)[1:2].repeat(3, 1)[:, None],
                 torch.zeros(3).long(), torch.ones(3), torch.ones(3, 1).bool(), 301)
    assert not bank.refresh(301)
    torch.testing.assert_close(old, bank.prototypes)
    bank.refresh(350)
    assert not bank.matches.any()
    bank.refresh(550)
    assert not bank.support.any() and not bank.memory[0] and not bank.memory[1]


def test_mutual_matching_allows_missing_and_rejects_ambiguity():
    config = SubgroupConfig()
    edges, _ = mutual_matches(torch.eye(3), torch.eye(3)[:2], config)
    assert edges.tolist() == [[True, False], [False, True], [False, False]]
    edges, _ = mutual_matches(torch.eye(3)[:1], torch.eye(3)[:1].repeat(2, 1), config)
    assert not edges.any()
    edges, _ = mutual_matches(torch.eye(3), -torch.eye(3), config)
    assert not edges.any()


def _manual_bank():
    bank = SubgroupBank(1, 3, 3, SubgroupConfig(), torch.device("cpu"))
    for domain in (0, 1):
        bank.prototypes[domain, 0, :, 0] = torch.eye(3)
        bank.support[domain, 0, :, 0] = 3
        bank.quality[domain, 0, :, 0] = 1
    bank.matches[0, :, 0, 0] = True
    bank.similarity[0, :, 0, 0] = 1
    return bank


def test_unmatched_same_class_is_excluded_and_student_has_gradient():
    bank = _manual_bank()
    features = [torch.tensor([[[.8, .6, 0.]]], requires_grad=True) for _ in range(2)]
    teachers = [torch.tensor([[[1., 0., 0.]]]) for _ in range(2)]
    labels, confidence, valid = [torch.tensor([0])] * 2, [torch.ones(1)] * 2, [torch.ones(1, 1).bool()] * 2
    loss, record = bank.loss(features, teachers, labels, confidence, valid)
    assert record["positive_pairs"] == 2 and record["negative_pairs"] == 4
    # Add a same-class target prototype with no matching edge: denominator must be unchanged.
    bank.prototypes[1, 0, 0, 1] = torch.tensor([-1., 0., 0.])
    bank.support[1, 0, 0, 1] = 3
    bank.quality[1, 0, 0, 1] = 1
    other, record = bank.loss(features, teachers, labels, confidence, valid)
    torch.testing.assert_close(loss, other)
    assert record["ignored_same_class_pairs"] == 1
    other.backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in features)
    assert bank.prototypes.grad is None
    bank.matches.zero_()
    zero, record = bank.loss(features, teachers, labels, confidence, valid)
    assert zero.requires_grad and zero == 0 and record["positive_pairs"] == 0


def test_only_other_class_negatives_detached_and_correct_temperature():
    anchors = torch.tensor([[.8, .6, 0.]], requires_grad=True)
    positive = torch.tensor([[1., 0., 0.]], requires_grad=True)
    negatives = torch.eye(3)[1:].requires_grad_(True)
    loss = selective_prototype_contrast(anchors, positive, negatives, torch.ones(1), 0.2)
    expected = F.cross_entropy(torch.tensor([[4., 3., 0.]]), torch.tensor([0]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert anchors.grad is not None and positive.grad is None and negatives.grad is None


def test_integrated_r2_step_uses_previous_snapshot_preserves_frozen_source_memory():
    torch.manual_seed(43)
    device = torch.device("cpu")
    model = _small_model(
        scales=(1., 2., 4.), num_domains=2, pyramid_gate_mode="sample_class",
        multiview_fusion_mode="class_query_low_rank", use_multiview_uncertainty=True,
        use_sign_aware_pyramid_guard=True, multiview_source_anchor_mix=0.5,
        pyramid_gate_shrinkage=0.5, pyramid_gate_ceiling=0.25,
        use_temporal_msad=True, use_source_prototype_memory=True, multiview_low_rank=8,
    )
    config = replace(SubgroupConfig(), confidence=0.0, jsd_threshold=10., minimum_votes=1,
                     min_support=1, match_threshold=-0.9, match_margin=0., assignment_threshold=-1.)
    alignment = SelectiveSubgroupAlignment(model, config, device)

    def batch(labeled):
        result = {"x": {key: torch.randn(3, 3, 62, 5) for key in ("1s", "2s", "4s")},
                  "mask": {key: torch.ones(3, 3).bool() for key in ("1s", "2s", "4s")},
                  "subject_id": torch.ones(3).long(), "session_id": torch.ones(3).long(),
                  "trial_id": torch.arange(3), "domain_id": torch.full((3,), 0 if labeled else 1)}
        if labeled:
            result["y"] = torch.arange(3)
        return result

    source, target = batch(True), batch(False)
    # Controlled prototype snapshot guarantees a nonempty contrastive path independently of
    # untrained pseudo-label quality. Production target labels come only from teacher_evidence.
    sf, _ = alignment._predict(source)
    tf, tl = alignment._predict(target)
    target_labels, _, _ = teacher_evidence(tl, config)
    for scale in range(3):
        for label in range(3):
            feature = F.normalize(sf[label, scale], dim=-1)
            alignment.bank.prototypes[:, scale, label, 0] = feature
            alignment.bank.support[:, scale, label, 0] = 3
            alignment.bank.quality[:, scale, label, 0] = 1
            alignment.bank.matches[scale, label, 0, 0] = True
            alignment.bank.similarity[scale, label, 0, 0] = 1
    alignment.bank.last_refresh = 300
    memory = SourceMultiPrototypeMemory(3, 3, 2, 32, .9, device)
    memory.update_source(torch.randn(6, 3, 32), torch.arange(3).repeat_interleave(2))
    before = memory.updates.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    # Disallow accidental use of the old whole-class attraction.
    with patch("MPUS_GA.trial_temporal.train.class_conditional_prototype_alignment_loss",
               side_effect=AssertionError("Old centroid loss must be disabled")):
        record = train_step(
            model, [source], target, optimizer, scheduler, device, 350, subgroup_experiment("A"),
            torch.tensor([[.2, .2, .6]]), PrototypeBank(1, 3, 3, 32, .9, .15, .1, device),
            .1, 5., 300, 600, .6, source_prototype_memory=memory, subgroup_alignment=alignment,
        )
    assert math.isfinite(record["total"]) and record["subgroup_contrast"] > 0
    assert record["subgroup_alignment"]["positive_pairs"] >= 9
    assert record["subgroup_alignment"]["snapshot_iteration"] == 300
    assert alignment.bank.last_refresh == 350 and alignment.pending is None
    assert alignment.teacher_updates == 1 and not alignment.teacher.training
    assert all(p.grad is None for p in alignment.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    torch.testing.assert_close(before, memory.updates)
    for teacher, student in zip(alignment.teacher.parameters(), model.parameters()):
        torch.testing.assert_close(teacher, student)  # First EMA update copies the student.


def test_target_labels_are_rejected_before_predict():
    controller = object.__new__(SelectiveSubgroupAlignment)
    try:
        controller.loss([], [], {}, {"y": torch.tensor([0])}, 1)
    except RuntimeError as error:
        assert "must not contain labels" in str(error)
    else:
        raise AssertionError("Target labels were accepted")


if __name__ == "__main__":
    torch.set_num_threads(2)
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    print(f"{len(tests)} tests passed", flush=True)
