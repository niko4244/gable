"""Deterministic AST bug injection. Site indices are stable across
multiple in-place mutations because operators never change tree shape."""
import ast
import copy

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add}


def _ops_for(node):
    ops = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1 \
            and type(node.ops[0]) in _CMP:
        ops.append("cmp")
    if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _ARITH:
        ops.append("arith")
    if isinstance(node, ast.BoolOp):
        ops.append("boolop")
    if isinstance(node, ast.Constant) and type(node.value) is int:
        ops += ["const+1", "const-1"]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        ops.append("drop_neg")
    return ops


def enumerate_sites(source):
    tree = ast.parse(source)
    sites = []
    for idx, node in enumerate(ast.walk(tree)):
        for op in _ops_for(node):
            sites.append({"idx": idx, "op": op,
                          "line": getattr(node, "lineno", 0)})
    return sites


def _apply(node, op):
    if op == "cmp":
        node.ops[0] = _CMP[type(node.ops[0])]()
    elif op == "arith":
        node.op = _ARITH[type(node.op)]()
    elif op == "boolop":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif op == "const+1":
        node.value += 1
    elif op == "const-1":
        node.value -= 1
    elif op == "drop_neg":
        node.op = ast.UAdd()
    else:
        raise ValueError(op)


def apply_sites(source, sites):
    """Return source with all given sites applied (sites from enumerate_sites)."""
    tree = copy.deepcopy(ast.parse(source))
    nodes = list(ast.walk(tree))
    for s in sites:
        _apply(nodes[s["idx"]], s["op"])
    return ast.unparse(tree) + "\n"
