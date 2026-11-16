"""Fixture-free, CPU-only training safety checks for the public Gable mirror."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import train


class FakeParameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self


class FakeModel:
    def __init__(self, named):
        self.named = named

    def named_parameters(self):
        return iter(self.named)


class FakeTrainer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.optimizer = None

    def create_optimizer(self):
        self.optimizer = SimpleNamespace(param_groups=[{"params": self.parameters}])


class TrainSafetyTests(unittest.TestCase):
    def test_exact_sft_assistant_loss_mask(self):
        ids = [2, 101, 102, 201, 202, 203, 204]
        prefix = ids[:3]
        train.prove_sft_prompt_mask(ids, [-100, -100, -100, *ids[3:]], prefix)
        for labels in (
            [-100, -100, -100, -100, 202, 203, 204],
            [-100, -100, -100, 201, 202, -100, 204],
            [-100, -100, -100, 201, 202, 203, -100],
            [-100, -100, 102, 201, 202, 203, 204],
            [-100, -100, -100, 201, 222, 203, 204],
        ):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(RuntimeError, "loss-mask invariants"):
                    train.prove_sft_prompt_mask(ids, labels, prefix)

    def test_collator_cannot_reintroduce_prompt_loss_or_mask_assistant(self):
        row = {"input_ids": [2, 101, 201, 202],
               "labels": [-100, -100, 201, 202]}
        for ids, labels in (
            ([2, 101, 201, 202], [-100, -100, 201, 202]),
            ([2, 101, 201, 202, 0], [-100, -100, 201, 202, -100]),
            ([0, 2, 101, 201, 202], [-100, -100, -100, 201, 202]),
        ):
            train.prove_sft_collator_keeps_mask(
                row, {"input_ids": [ids], "labels": [labels]},
            )
        for ids, labels in (
            ([2, 101, 201, 202], [2, 101, 201, 202]),
            ([2, 101, 201, 202], [-100, -100, 201, -100]),
            ([2, 101, 201, 203], [-100, -100, 201, 202]),
            ([0, 2, 101, 201, 202], [0, -100, -100, 201, 202]),
        ):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(RuntimeError, "collator altered"):
                    train.prove_sft_collator_keeps_mask(
                        row, {"input_ids": [ids], "labels": [labels]},
                    )

    def test_sft_rejects_backbone_and_empty_adapter_gradients(self):
        adapter = FakeParameter()
        backbone = FakeParameter()
        train.prove_only_lora_trainable(FakeModel([
            ("model.layers.0.lora_A.default.weight", adapter),
            ("model.layers.0.weight", FakeParameter(False)),
        ]))
        with self.assertRaisesRegex(RuntimeError, "backbone"):
            train.prove_only_lora_trainable(FakeModel([
                ("model.layers.0.lora_A.default.weight", adapter),
                ("model.layers.0.weight", backbone),
            ]))
        with self.assertRaisesRegex(RuntimeError, "No trainable"):
            train.prove_only_lora_trainable(FakeModel([
                ("model.layers.0.lora_A.default.weight", FakeParameter(False)),
            ]))

    def test_dpo_optimizer_updates_only_train_adapter(self):
        reference = FakeParameter()
        policy = FakeParameter()
        model = FakeModel([
            ("layers.0.lora_A.reference.weight", reference),
            ("layers.0.lora_A.train.weight", policy),
        ])
        trainer = FakeTrainer([policy])
        refs = train.prove_reference_excluded_from_optimizer(model, trainer)
        self.assertIs(refs[0], reference)
        self.assertFalse(reference.requires_grad)
        self.assertTrue(policy.requires_grad)

        reference.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "reference adapter entered"):
            train.prove_reference_excluded_from_optimizer(model, FakeTrainer([policy, reference]))
        with self.assertRaisesRegex(RuntimeError, "exclusively"):
            train.prove_reference_excluded_from_optimizer(model, FakeTrainer([]))
        with self.assertRaisesRegex(RuntimeError, "Non-policy"):
            train.prove_reference_excluded_from_optimizer(
                FakeModel(model.named + [("model.layers.0.weight", FakeParameter())]),
                FakeTrainer([policy]),
            )

    def test_dpo_fails_if_only_reference_is_trainable(self):
        model = FakeModel([
            ("layers.0.lora_B.reference.weight", FakeParameter()),
            ("layers.0.lora_B.train.weight", FakeParameter(False)),
        ])
        with self.assertRaisesRegex(RuntimeError, "No trainable policy"):
            train.prove_reference_excluded_from_optimizer(model, FakeTrainer([]))


if __name__ == "__main__":
    unittest.main()
