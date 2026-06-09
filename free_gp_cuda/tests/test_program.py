from __future__ import annotations

import unittest

from alpha_gen.free_gp_cuda.program import (
    BinaryNode,
    ConstNode,
    FieldNode,
    GateNode,
    Program,
    UnaryNode,
    field_names,
    is_valid_program,
    node_from_dict,
    operator_names,
    program_expression,
    program_key,
    validate_program,
)


class ProgramTests(unittest.TestCase):
    def test_valid_gate_program_roundtrip(self) -> None:
        program = Program(
            root=GateNode(
                op="gate_nan",
                signal=BinaryNode(
                    op="sub",
                    left=FieldNode("roe"),
                    right=UnaryNode("ts_mean_20", FieldNode("ret_20")),
                ),
                mask=UnaryNode("mask_rank_high_80", FieldNode("turnover")),
            )
        )

        self.assertTrue(is_valid_program(program, available_fields={"roe", "ret_20", "turnover"}))
        self.assertEqual(program.output_type, "numeric")
        self.assertEqual(program.size, 7)
        self.assertEqual(program.depth, 4)
        self.assertGreater(program.complexity_cost, 0.0)
        self.assertEqual(field_names(program.root), ("ret_20", "roe", "turnover"))
        self.assertEqual(operator_names(program.root), ("gate_nan", "sub", "ts_mean_20", "mask_rank_high_80"))
        self.assertEqual(
            program_expression(program),
            "gate_nan(sub(roe, ts_mean_20(ret_20)), mask_rank_high_80(turnover))",
        )

        restored = Program.from_json(program.to_json())
        self.assertEqual(program_key(restored), program_key(program))
        self.assertEqual(restored.to_dict(), program.to_dict())

    def test_commutative_key_sorts_add_and_mul_children(self) -> None:
        left = Program(BinaryNode("add", FieldNode("a"), FieldNode("b")))
        right = Program(BinaryNode("add", FieldNode("b"), FieldNode("a")))
        self.assertEqual(program_key(left), program_key(right))

        ordered_left = Program(BinaryNode("sub", FieldNode("a"), FieldNode("b")))
        ordered_right = Program(BinaryNode("sub", FieldNode("b"), FieldNode("a")))
        self.assertNotEqual(program_key(ordered_left), program_key(ordered_right))

    def test_validation_rejects_mask_in_numeric_binary(self) -> None:
        program = Program(
            BinaryNode(
                op="add",
                left=FieldNode("signal"),
                right=UnaryNode("mask_rank_high_50", FieldNode("state")),
            )
        )
        errors = validate_program(program, available_fields={"signal", "state"})
        self.assertTrue(any("right child" in item and "mask" in item for item in errors))

    def test_validation_rejects_gate_with_numeric_mask(self) -> None:
        program = Program(GateNode("gate_nan", signal=FieldNode("signal"), mask=FieldNode("state")))
        errors = validate_program(program, available_fields={"signal", "state"})
        self.assertTrue(any("gate mask must be mask" in item for item in errors))

    def test_validation_rejects_mask_root(self) -> None:
        program = Program(UnaryNode("mask_sign_pos", FieldNode("state")))
        errors = validate_program(program, available_fields={"state"})
        self.assertTrue(any("program root must be numeric" in item for item in errors))

    def test_validation_rejects_unknown_or_removed_operators(self) -> None:
        self.assertTrue(validate_program(Program(UnaryNode("identity", FieldNode("x"))), available_fields={"x"}))
        self.assertTrue(validate_program(Program(BinaryNode("where", FieldNode("x"), FieldNode("y"))), available_fields={"x", "y"}))
        self.assertTrue(validate_program(Program(BinaryNode("not_an_op", FieldNode("x"), FieldNode("y"))), available_fields={"x", "y"}))

    def test_validation_rejects_unknown_fields_constants_and_limits(self) -> None:
        program = Program(BinaryNode("add", FieldNode("a"), ConstNode(1.5)))
        self.assertTrue(is_valid_program(program, available_fields={"a"}))
        self.assertTrue(validate_program(program, available_fields={"b"}))

        bad_const = Program(ConstNode(float("nan")))
        self.assertTrue(validate_program(bad_const))

        deep = Program(UnaryNode("ts_mean_20", UnaryNode("slog", FieldNode("a"))))
        self.assertTrue(validate_program(deep, available_fields={"a"}, max_depth=2))
        self.assertTrue(validate_program(deep, available_fields={"a"}, max_size=2))

    def test_node_from_dict(self) -> None:
        raw = {
            "node": "gate",
            "op": "gate_zero",
            "signal": {"node": "field", "field": "signal"},
            "mask": {"node": "unary", "op": "mask_sign_pos", "child": {"node": "field", "field": "state"}},
        }
        node = node_from_dict(raw)
        program = Program(node)
        self.assertTrue(is_valid_program(program, available_fields={"signal", "state"}))


if __name__ == "__main__":
    unittest.main()
