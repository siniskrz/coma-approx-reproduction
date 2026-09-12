import unittest

try:
    import torch
    from comattack.attacks.extractive_suffix import AttackforLLMLingua2
except ImportError:
    torch = None
    AttackforLLMLingua2 = None


@unittest.skipUnless(torch is not None, "requires optional SPC torch dependency")
class LLMLingua2CoordinateSearchTest(unittest.TestCase):
    def test_widens_half_the_pool_without_growing_it(self):
        incumbent = torch.tensor([1, 1, 1])
        candidates = torch.tensor([[2, 1, 1], [1, 3, 1],
                                   [1, 1, 4], [5, 1, 1]])

        widened = AttackforLLMLingua2._add_second_coordinate(candidates, incumbent)

        self.assertEqual(widened.shape, candidates.shape)
        self.assertEqual(widened.ne(incumbent).sum(dim=1).tolist(), [2, 1, 2, 1])
        self.assertTrue(torch.equal(candidates, torch.tensor(
            [[2, 1, 1], [1, 3, 1], [1, 1, 4], [5, 1, 1]])))


if __name__ == "__main__":
    unittest.main()
