import unittest
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple


class SearchResult(NamedTuple):
    distance: int
    value: Any


@dataclass
class TrieNode:
    key: str
    values: list[Any]
    children: dict[str, "TrieNode"]


class Trie:
    def __init__(self):
        self.root = TrieNode(key="", children={}, values=[])

    def insert(self, key: Iterable[str], value: Any):
        """Inserts a key-value pair into the trie.

        The key may be any iterable of strings. This could be a single str object, or a list of
        strings."""
        node = self.root
        for part in key:
            if part not in node.children:
                node.children[part] = TrieNode(key=part, children={}, values=[])

            node = node.children[part]

        node.values.append(value)

    def exact_search(self, key: Iterable[str]) -> list[Any]:
        """Searches for a key in the trie.

        The key may be any iterable of strings. This could be a single str object, or a list of
        strings.

        If the key is found, the value associated with the key is returned. If the key is not found,
        None is returned.
        """
        node = self.root
        for part in key:
            if part not in node.children:
                return []
            node = node.children[part]
        return node.values

    def distance_search(
        self,
        key: Sequence[str],
        distance_func: Callable[[str, str], int],
        node_max_distance: int = 3,
        max_distance: int = 3,
    ) -> list[SearchResult]:
        """Searches for a key in the trie and returns a list of the closest matches.

        The key may be any iterable of strings. This could be a single str object, or a list of
        strings.

        If a distance function is provided, the search will return the value of the closest match
        within the given distance. If no such match is found, None is returned.
        """
        matches: list[SearchResult] = []

        # Perform a depth-first search to find the closest matches.
        # As we traverse the trie, we not only check the distance of the given key to the children ensure
        # the single node distance is less than the max_distance, but we also check to see that the cumulative
        # distance of the path to the node is less than the max_distance.
        def dfs(node: TrieNode, search_key: Sequence[str], path_distance: int):
            # print(search_key, path_distance)
            if path_distance > max_distance:
                return
            if len(search_key) == 0:
                for value in node.values:
                    matches.append(SearchResult(distance=path_distance, value=value))
                return
            for child_key, child in node.children.items():
                if search_key[0] == child_key:
                    dfs(child, search_key[1:], path_distance)
                    continue
                if search_key[0] == None or child_key == None:
                    continue
                node_distance = distance_func(search_key[0], child_key)
                # print(f"Checking {search_key[0]}~={child_key} with distance {node_distance}")
                if node_distance <= node_max_distance:
                    dfs(child, search_key[1:], path_distance + node_distance)
                if node_distance > 0:
                    # Search considering this is a deletion
                    delete_distance = distance_func("", child_key)
                    # print("Start deletion search with cost", delete_distance)
                    dfs(child, search_key, path_distance + delete_distance)

                    # Search considering this is an insertion
                    insert_distance = distance_func(search_key[0], "")
                    # print("Start insertion search with cost", insert_distance)
                    dfs(node, search_key[1:], path_distance + insert_distance)

        dfs(self.root, list(key), 0)

        # Remove duplicate values and keep only the one with the smallest distance.
        values: dict[Any, SearchResult] = {}
        for match in matches:
            if (
                match.value not in values
                or match.distance < values[match.value].distance
            ):
                values[match.value] = match

        unique_matches = list(sorted(values.values(), key=lambda x: x.distance))

        return unique_matches


# class TestLevenshtein(unittest.TestCase):
#     def test_levenshtein_by_char(self):
#         from Levenshtein import distance
#
#         trie = Trie()
#         trie.insert("hello world", "hello world")
#
#         # assert trie.exact_search("hello world") == "hello world"
#         # assert trie.exact_search("hello") is None
#
#         # assert not trie.distance_search("h", distance), trie.distance_search("h", distance)
#
#         def _subtest(search_key, expected):
#             with self.subTest(search_key=search_key):
#                 results = trie.distance_search(search_key, distance, max_distance=5)
#                 self.assertEqual(results, expected)
#
#         _subtest(
#             "hello world",
#             [SearchResult(0, "hello world")],
#         )
#         _subtest(
#             "hello forld",
#             [SearchResult(1, "hello world")],
#         )
#         _subtest(
#             "hello orld",
#             [SearchResult(1, "hello world")],
#         )
#         _subtest(
#             "hello wworld",
#             [SearchResult(1, "hello world")],
#         )
#         _subtest(
#             "goodbye world",
#             [],
#         )
#
#     def test_levenshtein_by_word(self):
#         from Levenshtein import distance
#
#         trie = Trie()
#         trie.insert(
#             "Turn on the living room light".split(), "turn on living room light"
#         )
#         trie.insert(
#             "Turn on the living room lights".split(), "turn on living room light"
#         )
#         trie.insert("Turn on the bedroom light".split(), "turn on bedroom light")
#         trie.insert("Turn on the bedroom lights".split(), "turn on bedroom light")
#         trie.insert("Turn on the kitchen light".split(), "turn on kitchen light")
#         trie.insert("Turn on the kitchen lights".split(), "turn on kitchen light")
#
#         def _subtest(search_key, expected):
#             with self.subTest(search_key=search_key):
#                 results = trie.distance_search(
#                     search_key.split(), distance, max_distance=7
#                 )
#                 self.assertEqual(results, expected)
#
#         # Test exact matches
#         _subtest(
#             "Turn on the living room light",
#             [SearchResult(0, "turn on living room light")],
#         )
#         _subtest(
#             "Turn on the living room lights",
#             [SearchResult(0, "turn on living room light")],
#         )
#         _subtest(
#             "Turn on the bedroom light",
#             [SearchResult(0, "turn on bedroom light")],
#         )
#         _subtest(
#             "Turn on the bedroom lights",
#             [SearchResult(0, "turn on bedroom light")],
#         )
#         _subtest(
#             "Turn on the kitchen light",
#             [SearchResult(0, "turn on kitchen light")],
#         )
#         _subtest(
#             "Turn on the kitchen lights",
#             [SearchResult(0, "turn on kitchen light")],
#         )
#
#         # Test some close matches
#         _subtest(
#             "Turn on the living room lite",
#             [SearchResult(3, "turn on living room light")],
#         )
#         # TODO: This test fails. Using word matching this distance is too high because "room"
#         # becomes a 4 character insertion adn then when checking the next word, it becomes
#         # a 4 character deletion. Maybe combine word based and character based matching?
#         _subtest(
#             "Turn on the livingroom lite",
#             [SearchResult(4, "turn on living room light")],
#         )
#
#         # Test some misses
#         _subtest(
#             "Turn the livingroom lite on",
#             [],
#         )  # Distance too high because words are in the wrong order
#         # TODO: This fails because "off" is too close to "on" and falls in our per node distance
#         # threshold. We could avoid this by decreasing that threshold, but then it would prevent
#         # us from finding correct matches for `light` and `lite`. Maybe the lower thershold is
#         # fine for Levenshtein becuase it's intended for text based matching. I think it's ok if
#         # someone types "lite" and we reject it because it's too far from "light", but I don't
#         # think it's ok to match "on" if someone writes "off". Matching "lite" with "light" is
#         # more important for phonetic search, so I'll try to match there.
#         _subtest(
#             "Turn off the living room light",
#             [],
#         )  # Distance too high because of the word "off"
#         _subtest(
#             "Turn on the den light on",
#             [],
#         )


class TestSearches(unittest.TestCase):
    def setUp(self) -> None:
        self.trie: Trie | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.sentences = {
            "Turn on the living room light": "turn on living room light",
            "Turn on the living room lights": "turn on living room light",
            "Turn on the bedroom light": "turn on bedroom light",
            "Turn on the bedroom lights": "turn on bedroom light",
            "Turn on the kitchen light": "turn on kitchen light",
            "Turn on the kitchen lights": "turn on kitchen light",
        }

        cls.test_cases: list[tuple[str, str|None]] = [
            ("Turn on the living room light", "turn on living room light"),
            ("Turn on the living room lights", "turn on living room light"),
            ("Turn on the bedroom light", "turn on bedroom light"),
            ("Turn on the bedroom lights", "turn on bedroom light"),
            ("Turn on the kitchen light", "turn on kitchen light"),
            ("Turn on the kitchen lights", "turn on kitchen light"),
            ("Turn on the living room lite", "turn on living room light"),
            ("Turn on the livingroom lite", "turn on living room light"),
            ("Turn the livingroom lite on", None),
            ("Turn off the living room light", None),
            ("Turn on the den light on", None),
        ]
        cls.benchmark_summaries: list[tuple[str, int, int, int]] = []

    @classmethod
    def tearDownClass(cls) -> None:
        assert cls.benchmark_summaries
        for summary in cls.benchmark_summaries:
            print(
                "\n",
                f"{summary[0]}: {len(cls.test_cases)} tests. "
                f"{summary[1]} mismatch, {summary[2]} false positives, {summary[3]} false negatives.",
                "\n",
            )

    def build_trie(self, pre_proccess_func: Callable[[str], Any] = lambda x: x) -> Trie:
        trie = Trie()
        for sentence, value in self.sentences.items():
            trie.insert(pre_proccess_func(sentence), value)
        return trie

    def _test_results(
        self,
        test_name: str,
        distance_func: Callable[[str, str], int],
        node_max_distance: int = 3,
        max_distance: int = 3,
        pre_proccess_func: Callable[[str], Any] = lambda x: x,
    ):
        trie = self.trie or self.build_trie(pre_proccess_func)
        mismatch, false_positives, false_negatives = 0, 0, 0
        for search_key, expected in self.test_cases:
            with self.subTest(search_key=search_key):
                results = trie.distance_search(
                    pre_proccess_func(search_key),
                    distance_func,
                    node_max_distance,
                    max_distance,
                )

                # Build benchmark summary
                if expected is None and results:
                    false_positives += 1
                elif not results and expected is not None:
                    false_negatives += 1
                elif expected is None and not results:
                    pass
                elif results[0].value != expected:
                    mismatch += 1

                if expected is None:
                    self.assertEqual(results, [])
                    continue
                else:
                    self.assertNotEqual(results, [])

                self.assertEqual(results[0].value, expected)

        self.benchmark_summaries.append((test_name, mismatch, false_positives, false_negatives))

    # def test_levenshtein_char(self):
    #     from Levenshtein import distance
    #
    #     self._test_results(
    #         distance,
    #     )
    #
    # def test_levenshtein_word(self):
    #     from Levenshtein import distance
    #
    #     self._test_results(
    #         distance,
    #         pre_proccess_func=lambda x: x.split(),
    #     )
    #
    # def test_soundex_levenshtein(self):
    #     import fuzzy
    #     from Levenshtein import distance
    #
    #     soundex = fuzzy.Soundex(4)
    #     self._test_results(
    #         distance,
    #         pre_proccess_func=lambda x: soundex(x),
    #     )
    #
    # def test_fuzzy_dmeta_levenshtein(self):
    #     import fuzzy
    #     from Levenshtein import distance
    #
    #     dmeta = fuzzy.DMetaphone(10)
    #     self._test_results(
    #         distance,
    #         pre_proccess_func=lambda x: dmeta(x),
    #     )
    

    def test_phonetics_metaphone_levenshtein(self):
        import phonetics
        from Levenshtein import distance

        self._test_results(
            "ponetics_metaphone_levenshtein",
            distance,
            pre_proccess_func=phonetics.metaphone,
        )

    def test_phonetics_dmetaphone_levenshtein(self):
        import phonetics
        from Levenshtein import distance

        self._test_results(
            "ponetics_dmetaphone_levenshtein",
            distance,
            pre_proccess_func=phonetics.dmetaphone,
        )

    def test_phonetics_soundex_levenshtein(self):
        import phonetics
        from Levenshtein import distance

        self._test_results(
            "ponetics_soundex_levenshtein",
            distance,
            pre_proccess_func=phonetics.soundex
        )

    def test_phonetics_nysiis_levenshtein(self):
        import phonetics
        from Levenshtein import distance

        self._test_results(
            "ponetics_nysiis_levenshtein",
            distance,
            pre_proccess_func=phonetics.nysiis,
        )


if __name__ == "__main__":
    unittest.main()
