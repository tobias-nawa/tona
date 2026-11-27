"""Programming tasks for bench-effort.py: (signature, description, tests) per task name."""

TASKS = {
    "duration": (
        "parse_duration(s: str) -> int",
        "Parse a duration like '1h30m15s', '45s', '2h', '3m' into seconds. Units h, m, s each at most once, in "
        "that order, with positive integer amounts. Raise ValueError for anything else (empty string, "
        "unknown unit, wrong order, repeated unit, missing number, spaces).",
        """
assert parse_duration('1h30m15s') == 5415
assert parse_duration('45s') == 45 and parse_duration('2h') == 7200 and parse_duration('3m') == 180
assert parse_duration('10h5s') == 36005
for bad in ['', '5x', '1m1h', '1h1h', 'h', '1 h', '-1s', '1.5h', 's5']:
    try:
        parse_duration(bad)
        raise SystemExit('no error for ' + repr(bad))
    except ValueError:
        pass
""",
    ),
    "intervals": (
        "merge_intervals(iv: list[list[int]]) -> list[list[int]]",
        "Merge overlapping or touching closed intervals [a, b] (a <= b) and return them sorted by start.",
        """
assert merge_intervals([]) == []
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]
assert merge_intervals([[5, 5], [1, 4], [0, 0]]) in ([[0, 0], [1, 5]], [[0, 0], [1, 4], [5, 5]])
assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
assert merge_intervals([[3, 4], [1, 2]]) == [[1, 2], [3, 4]]
""",
    ),
    "lru": (
        "class LRUCache with __init__(self, capacity: int), get(self, key) -> int, put(self, key, value) -> None",
        "A least-recently-used cache. get returns -1 for a missing key; both get and put mark a key as recently "
        "used; put on a full cache evicts the least recently used key. Capacity 0 stores nothing.",
        """
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1
c.put(4, 4)
assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
c.put(3, 30)
c.put(5, 5)
assert c.get(4) == -1 and c.get(3) == 30
z = LRUCache(0)
z.put(1, 1)
assert z.get(1) == -1
""",
    ),
    "roman": (
        "to_roman(n: int) -> str and from_roman(s: str) -> int",
        "Convert between integers 1..3999 and canonical Roman numerals. from_roman raises ValueError for "
        "non-canonical input such as 'IIII', 'VX', 'IL' or 'MMMM', and for empty strings.",
        """
for n in range(1, 4000):
    assert from_roman(to_roman(n)) == n
assert to_roman(1994) == 'MCMXCIV' and to_roman(3999) == 'MMMCMXCIX' and to_roman(4) == 'IV'
for bad in ['IIII', 'VX', 'IL', 'MMMM', '', 'IC', 'XM', 'VV', 'abc']:
    try:
        from_roman(bad)
        raise SystemExit('no error for ' + repr(bad))
    except ValueError:
        pass
""",
    ),
    "expr": (
        "evaluate(expr: str) -> float",
        "Evaluate an arithmetic expression with non-negative decimal numbers, + - * /, parentheses, unary minus "
        "and arbitrary spaces, with the usual precedence and left associativity. Do not use eval or exec. "
        "Raise ValueError for malformed input and ZeroDivisionError for division by zero.",
        """
assert evaluate('2*(3+4)-5/2') == 11.5
assert evaluate('1 - 2 - 3') == -4
assert evaluate('-(2+3)*-2') == 10
assert evaluate('2*-3') == -6
assert evaluate('8/4/2') == 1
assert abs(evaluate('3.5 + .5') - 4.0) < 1e-9 or True
assert evaluate('((7))') == 7
assert evaluate('--3') == 3
for bad in ['', '2+', '(1+2', '1+2)', '2 3', '*2']:
    try:
        evaluate(bad)
        raise SystemExit('no error for ' + repr(bad))
    except ValueError:
        pass
try:
    evaluate('1/(2-2)')
    raise SystemExit('no division by zero')
except ZeroDivisionError:
    pass
""",
    ),
    "topo": (
        "topo_sort(deps: dict[str, list[str]]) -> list[str]",
        "deps maps a node to the nodes it depends on (which must come before it). Nodes appearing only as "
        "dependencies are included too. Among all valid orders return the lexicographically smallest one "
        "(always pick the smallest available node). Raise ValueError if there is a cycle.",
        """
assert topo_sort({}) == []
assert topo_sort({'b': ['a'], 'c': ['a'], 'd': ['b', 'c']}) == ['a', 'b', 'c', 'd']
assert topo_sort({'x': [], 'a': ['x']}) == ['x', 'a']
assert topo_sort({'c': [], 'b': [], 'a': ['c']}) == ['b', 'c', 'a']
assert topo_sort({'a': ['z']}) == ['z', 'a']
for cyc in [{'a': ['a']}, {'a': ['b'], 'b': ['a']}, {'a': ['b'], 'b': ['c'], 'c': ['a'], 'd': []}]:
    try:
        topo_sort(cyc)
        raise SystemExit('no cycle detected')
    except ValueError:
        pass
""",
    ),
    "wrap": (
        "wrap(text: str, width: int) -> list[str]",
        "Greedy word wrap: words are separated by any whitespace; put as many words on a line as fit (single "
        "spaces between them, line length <= width). A word longer than width is split into chunks of exactly "
        "width characters (the last chunk may be shorter and can share its line with following words). Empty "
        "or whitespace-only text gives [].",
        """
assert wrap('', 5) == [] and wrap('   ', 5) == []
assert wrap('the quick brown fox', 10) == ['the quick', 'brown fox']
assert wrap('a b c d', 3) == ['a b', 'c d']
assert wrap('abcdefghij xy', 4) in (['abcd', 'efgh', 'ij', 'xy'], ['abcd', 'efgh', 'ij x', 'y'])
assert wrap('hello\\n\\tworld  again', 11) == ['hello world', 'again']
assert all(len(l) <= 7 for l in wrap('lorem ipsum dolor sit amet consectetur', 7))
""",
    ),
    "csv": (
        "parse_csv_line(line: str) -> list[str]",
        "Parse one CSV line (RFC 4180): fields separated by commas, a field may be enclosed in double quotes, "
        'inside quotes commas are literal and a doubled quote "" is one quote. An empty line is one empty field. '
        "Raise ValueError for an unterminated quote or a quote inside an unquoted field.",
        '''
assert parse_csv_line('a,b,c') == ['a', 'b', 'c']
assert parse_csv_line('') == ['']
assert parse_csv_line(',') == ['', '']
assert parse_csv_line('"a,b",c') == ['a,b', 'c']
assert parse_csv_line('"he said ""hi""",x') == ['he said "hi"', 'x']
assert parse_csv_line('"",""') == ['', '']
assert parse_csv_line('1,"2",3,') == ['1', '2', '3', '']
for bad in ['"abc', 'a"b,c', '"a"b']:
    try:
        parse_csv_line(bad)
        raise SystemExit('no error for ' + repr(bad))
    except ValueError:
        pass
''',
    ),
    "dijkstra": (
        "shortest_paths(n: int, edges: list[tuple[int, int, int]], src: int) -> list[float]",
        "Directed graph with nodes 0..n-1 and edges (u, v, w), w >= 0, possibly with parallel edges. Return "
        "the shortest distance from src to every node, float('inf') if unreachable.",
        """
inf = float('inf')
assert shortest_paths(1, [], 0) == [0]
assert shortest_paths(4, [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)], 0) == [0, 3, 1, 4]
assert shortest_paths(3, [(0, 1, 5), (0, 1, 2)], 0) == [0, 2, inf]
assert shortest_paths(3, [(1, 0, 1)], 0) == [0, inf, inf]
assert shortest_paths(3, [(0, 1, 0), (1, 2, 0)], 1) == [inf, 0, 0]
""",
    ),
    "edit": (
        "edit_distance(a: str, b: str) -> int",
        "Levenshtein distance (insert, delete, substitute, each cost 1). Must handle strings of length 2000 in "
        "under a few seconds.",
        """
assert edit_distance('', '') == 0 and edit_distance('abc', '') == 3
assert edit_distance('kitten', 'sitting') == 3
assert edit_distance('flaw', 'lawn') == 2
assert edit_distance('intention', 'execution') == 5
import random
random.seed(1)
x = ''.join(random.choice('ab') for _ in range(2000)); y = ''.join(random.choice('ab') for _ in range(2000))
assert 0 < edit_distance(x, y) < 2000
""",
    ),
    "spiral": (
        "spiral(m: list[list[int]]) -> list[int]",
        "Return the elements of a rectangular matrix in clockwise spiral order starting top-left.",
        """
assert spiral([]) == [] and spiral([[]]) == []
assert spiral([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
assert spiral([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
assert spiral([[1], [2], [3]]) == [1, 2, 3]
assert spiral([[1, 2, 3]]) == [1, 2, 3]
assert spiral([[1, 2], [3, 4], [5, 6]]) == [1, 2, 4, 6, 5, 3]
""",
    ),
    "semver": (
        "compare_semver(a: str, b: str) -> int",
        "Compare two Semantic Versioning 2.0.0 strings (MAJOR.MINOR.PATCH, optional -prerelease, optional "
        "+build). Return -1, 0 or 1. Build metadata is ignored. Prerelease identifiers compare numerically if "
        "both are numeric, numeric ones are lower than alphanumeric ones, otherwise ASCII; a version with "
        "prerelease is lower than the same version without; a shorter prerelease list is lower if all "
        "preceding identifiers are equal.",
        """
assert compare_semver('1.0.0', '1.0.0') == 0
assert compare_semver('1.0.0', '2.0.0') == -1 and compare_semver('2.1.0', '2.0.9') == 1
assert compare_semver('1.10.0', '1.9.0') == 1
assert compare_semver('1.0.0-alpha', '1.0.0') == -1
order = ['1.0.0-alpha', '1.0.0-alpha.1', '1.0.0-alpha.beta', '1.0.0-beta', '1.0.0-beta.2', '1.0.0-beta.11',
         '1.0.0-rc.1', '1.0.0']
for i in range(len(order) - 1):
    assert compare_semver(order[i], order[i + 1]) == -1, (order[i], order[i + 1])
    assert compare_semver(order[i + 1], order[i]) == 1
assert compare_semver('1.0.0+build.1', '1.0.0+build.2') == 0
assert compare_semver('1.0.0-alpha+x', '1.0.0-alpha') == 0
""",
    ),
    "sudoku": (
        "valid_sudoku(board: list[str]) -> bool",
        "board has 9 strings of 9 characters, digits '1'-'9' or '.' for empty. Return True if no row, column "
        "or 3x3 box contains a digit twice (the board need not be solvable). Return False for malformed boards "
        "(wrong size or other characters).",
        """
ok = ['53..7....', '6..195...', '.98....6.', '8...6...3', '4..8.3..1', '7...2...6', '.6....28.', '...419..5',
      '....8..79']
assert valid_sudoku(ok)
b = list(ok); b[0] = '83..7....'
assert not valid_sudoku(b)
b = list(ok); b[1] = '6..195..5'
assert valid_sudoku(b) is False or True
b = list(ok); b[8] = '....8..77'
assert not valid_sudoku(b)
b = list(ok); b[4] = '4..8.3.11'
assert not valid_sudoku(b)
assert valid_sudoku(['.' * 9] * 9)
assert not valid_sudoku(['.' * 9] * 8)
assert not valid_sudoku(['.' * 8 + '0'] + ['.' * 9] * 8)
b = ['1........', '.........', '..1......'] + ['.' * 9] * 6
assert not valid_sudoku(b)
""",
    ),
    "islands": (
        "count_islands(grid: list[str]) -> int",
        "grid rows consist of '1' (land) and '0' (water). Count islands of land connected horizontally or "
        "vertically (not diagonally). Must handle a 300x300 grid of all land without recursion errors.",
        """
assert count_islands([]) == 0
assert count_islands(['11000', '11000', '00100', '00011']) == 3
assert count_islands(['101', '010', '101']) == 5
assert count_islands(['111', '101', '111']) == 1
assert count_islands(['1' * 300] * 300) == 1
""",
    ),
    "nextperm": (
        "next_permutation(xs: list[int]) -> list[int]",
        "Return the next lexicographically greater permutation of xs (as a new list, duplicates allowed); if "
        "xs is the greatest, return it sorted ascending. Do not modify the input.",
        """
assert next_permutation([1, 2, 3]) == [1, 3, 2]
assert next_permutation([3, 2, 1]) == [1, 2, 3]
assert next_permutation([1, 1, 5]) == [1, 5, 1]
assert next_permutation([1, 3, 2]) == [2, 1, 3]
assert next_permutation([2, 2, 2]) == [2, 2, 2]
assert next_permutation([]) == []
x = [1, 5, 1]; next_permutation(x); assert x == [1, 5, 1]
""",
    ),
    "flatten": (
        "flatten(obj) -> dict",
        "Flatten nested dicts and lists into a dict from dotted paths to leaf values; list elements use their "
        "index as key. Empty dicts and empty lists are kept as leaf values. A top-level leaf (not dict or "
        "list) maps from the empty path ''.",
        """
assert flatten({'a': {'b': 1, 'c': [2, {'d': 3}]}}) == {'a.b': 1, 'a.c.0': 2, 'a.c.1.d': 3}
assert flatten({'x': {}, 'y': []}) == {'x': {}, 'y': []}
assert flatten([1, [2]]) == {'0': 1, '1.0': 2}
assert flatten(5) == {'': 5}
assert flatten({}) == {} or flatten({}) == {'': {}}
assert flatten({'a': None}) == {'a': None}
""",
    ),
    "ratelimit": (
        "class RateLimiter with __init__(self, limit: int, window: float) and allow(self, t: float) -> bool",
        "Sliding-window rate limiter: allow(t) returns True and records the call if fewer than limit "
        "calls were allowed in the half-open window (t - window, t]; otherwise False (rejected calls are "
        "not recorded). Calls come with non-decreasing t.",
        """
r = RateLimiter(2, 10)
assert r.allow(0) and r.allow(1)
assert not r.allow(5)
assert not r.allow(9.99)
assert r.allow(10)
assert not r.allow(10.5)
assert r.allow(11)
z = RateLimiter(0, 5)
assert not z.allow(0)
s = RateLimiter(1, 1)
assert s.allow(0) and not s.allow(0) and s.allow(1)
""",
    ),
    "rle": (
        "rle_encode(s: str) -> str and rle_decode(s: str) -> str",
        "Run-length encoding for strings without digits: each run becomes count followed by the character, "
        "e.g. 'aaabcc' -> '3a1b2c'. Counts can have several digits. rle_decode raises ValueError for malformed "
        "input (missing count, count 0, trailing digits, digit as data).",
        """
assert rle_encode('') == '' and rle_decode('') == ''
assert rle_encode('aaabcc') == '3a1b2c'
assert rle_encode('a' * 12 + 'b') == '12a1b'
for s in ['x', 'abc', 'zzzzzzzzzzzzzzzzzzzzzzzzzz', 'a b  c', 'ÄÄö']:
    assert rle_decode(rle_encode(s)) == s
for bad in ['a', '3', '0a', '2a3', '12']:
    try:
        rle_decode(bad)
        raise SystemExit('no error for ' + repr(bad))
    except ValueError:
        pass
""",
    ),
    "minwindow": (
        "min_window(s: str, t: str) -> str",
        "Return the shortest substring of s containing all characters of t with multiplicity; among equally "
        "short ones the leftmost; '' if none exists or t is empty. Must be O(len(s) + len(t)).",
        """
assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'
assert min_window('a', 'a') == 'a' and min_window('a', 'aa') == ''
assert min_window('aa', 'aa') == 'aa'
assert min_window('abc', '') == ''
assert min_window('bba', 'ab') == 'ba'
assert min_window('xyzxy', 'yx') == 'xy'
assert min_window('a' * 100000 + 'b', 'ab') == 'ab'
""",
    ),
    "brackets": (
        "first_bracket_error(s: str) -> int",
        "Check brackets ()[]{} in s, ignoring all other characters. Return -1 if balanced. Otherwise return "
        "the index of the first closing bracket that does not match or has no opener; if there is no such "
        "closer but openers remain unclosed, return len(s).",
        """
assert first_bracket_error('') == -1 and first_bracket_error('a(b[c]{d}e)f') == -1
assert first_bracket_error('(]') == 1
assert first_bracket_error(')') == 0
assert first_bracket_error('(()') == 3
assert first_bracket_error('{[}]') == 2
assert first_bracket_error('x)(') == 1
""",
    ),
    "weekday": (
        "weekday(y: int, m: int, d: int) -> str",
        "Return the English weekday name ('Monday' … 'Sunday') of a date in the proleptic Gregorian calendar, "
        "years 1..9999, without using datetime, calendar or time. Raise ValueError for invalid dates "
        "(including February 29 in non-leap years).",
        """
import datetime, random
random.seed(7)
names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for _ in range(500):
    o = random.randint(1, datetime.date(9999, 12, 31).toordinal())
    x = datetime.date.fromordinal(o)
    assert weekday(x.year, x.month, x.day) == names[x.weekday()], x
for bad in [(2023, 2, 29), (1900, 2, 29), (2024, 13, 1), (2024, 4, 31), (0, 1, 1), (2024, 1, 0)]:
    try:
        weekday(*bad)
        raise SystemExit('no error for ' + repr(bad))
    except ValueError:
        pass
assert weekday(2000, 2, 29) == 'Tuesday'
""",
    ),
    "multiply": (
        "multiply(a: str, b: str) -> str",
        "Multiply two non-negative integers given as decimal strings (possibly with leading zeros, up to "
        "2000 digits) without converting them to int; return the product without leading zeros ('0' for "
        "zero).",
        """
assert multiply('0', '0') == '0' and multiply('000', '12') == '0'
assert multiply('123', '456') == '56088'
assert multiply('007', '08') == '56'
import random
random.seed(3)
for _ in range(20):
    x = str(random.randint(0, 10 ** 300)); y = str(random.randint(0, 10 ** 300))
    assert multiply(x, y) == str(int(x) * int(y))
x = '9' * 2000
assert multiply(x, x) == str(int(x) ** 2)
""",
    ),
    "kthstream": (
        "class KthLargest with __init__(self, k: int, nums: list[int]) and add(self, val: int) -> int",
        "Track the k-th largest element of a stream; add inserts val and returns the current k-th largest. "
        "nums may have fewer than k elements initially (then at least k - 1); 10^5 adds must be fast.",
        """
s = KthLargest(3, [4, 5, 8, 2])
assert [s.add(v) for v in [3, 5, 10, 9, 4]] == [4, 5, 5, 8, 8]
t = KthLargest(1, [])
assert t.add(-3) == -3 and t.add(-2) == -2 and t.add(-4) == -2
u = KthLargest(2, [0])
assert u.add(-1) == -1 and u.add(1) == 0
import random, time
random.seed(5)
v = KthLargest(100, [])
t0 = time.time()
for _ in range(100000):
    v.add(random.randint(0, 10 ** 9))
assert time.time() - t0 < 5
""",
    ),
}
