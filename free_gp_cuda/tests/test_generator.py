from __future__ import annotations

import random
import unittest

from alpha_gen.free_gp_cuda.generator import (
    ProgramGeneratorConfig,
    crossover_programs,
    get_subtree,
    iter_paths,
    mutate_program,
    random_program,
    replace_subtree,
)
from alpha_gen.free_gp_cuda.program import FieldNode, Program, is_valid_program, node_depth, node_size, program_key


class GeneratorTests(unittest.TestCase):
    def test_random_program_is_valid_and_reproducible(self) -> None:
        fields = ("a", "b", "c")
        config = ProgramGeneratorConfig(max_depth=4, max_size=32)
        first = random_program(fields, config=config, random_state=7)
        second = random_program(fields, config=config, random_state=7)

        self.assertEqual(program_key(first), program_key(second))
        self.assertTrue(is_valid_program(first, available_fields=fields, max_depth=4, max_size=32))
        self.assertLessEqual(node_depth(first.root), 4)
        self.assertLessEqual(node_size(first.root), 32)

    def test_mutation_and_crossover_keep_program_valid(self) -> None:
        fields = ("a", "b", "c", "d")
        config = ProgramGeneratorConfig(max_depth=4, max_size=40)
        rng = random.Random(3)
        left = random_program(fields, config=config, random_state=rng)
        right = random_program(fields, config=config, random_state=rng)

        mutated = mutate_program(left, fields, config=config, random_state=rng)
        crossed = crossover_programs(left, right, fields, config=config, random_state=rng)

        self.assertTrue(is_valid_program(mutated, available_fields=fields, max_depth=4, max_size=40))
        self.assertTrue(is_valid_program(crossed, available_fields=fields, max_depth=4, max_size=40))

    def test_path_get_and_replace(self) -> None:
        program = Program(FieldNode("a"))
        paths = tuple(iter_paths(program.root))
        self.assertEqual(paths, (((), program.root),))
        self.assertEqual(get_subtree(program.root, ()), FieldNode("a"))
        replaced = replace_subtree(program.root, (), FieldNode("b"))
        self.assertEqual(replaced, FieldNode("b"))


if __name__ == "__main__":
    unittest.main()
