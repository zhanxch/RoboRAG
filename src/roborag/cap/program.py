"""Bounded interpreter for the explicitly documented Python CaP subset.

No exec/eval or access to Python objects outside JSON values and device methods.
This is a program interpreter, not a sandbox for arbitrary Python packages.
"""
import ast
import json
import operator


class ProgramError(ValueError):
    pass


METHODS = {"call_tool", "read_resource"}
ALLOWED = (ast.Module, ast.Assign, ast.Expr, ast.If, ast.For, ast.Break, ast.Continue,
           ast.Pass, ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
           ast.Dict, ast.Subscript, ast.Call, ast.Attribute, ast.keyword, ast.BinOp,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UnaryOp, ast.USub, ast.UAdd,
           ast.Not, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
           ast.In, ast.NotIn, ast.BoolOp, ast.And, ast.Or)
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos,
       ast.Not: operator.not_, ast.Eq: operator.eq, ast.NotEq: operator.ne,
       ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
       ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
       ast.NotIn: lambda a, b: a not in b}


def parse(source):
    if not isinstance(source, str) or not source.strip() or len(source) > 20000:
        raise ProgramError("program must contain 1..20000 characters")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ProgramError(f"unsupported Python construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ProgramError("private names are unavailable")
        if isinstance(node, ast.Constant) and type(node.value) not in (str, int, float, bool, type(None)):
            raise ProgramError("only JSON constants are supported")
        if isinstance(node, (ast.Assign, ast.For)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(not isinstance(t, ast.Name) or t.id in ("mcp", "range") for t in targets):
                raise ProgramError("assign only to local variable names")
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "mcp" or node.attr not in METHODS:
                raise ProgramError("only documented MCP methods are available")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute) and not (
                    isinstance(node.func, ast.Name) and node.func.id == "range"):
                raise ProgramError("calls must target MCP methods or range")
            if any(k.arg is None for k in node.keywords):
                raise ProgramError("expanded keyword arguments are unsupported")
    return tree


class _Break(Exception): pass
class _Continue(Exception): pass


class Program:
    def __init__(self, mcp, emit, *, max_operations=5000, max_calls=64):
        self.mcp, self.emit = mcp, emit
        self.remaining, self.calls = max_operations, max_calls
        self.variables = {}

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ProgramError("program operation budget exhausted")

    def execute(self, source):
        tree = parse(source)  # Reject unsupported syntax before any device call.
        self.block(tree.body)

    def block(self, nodes):
        for node in nodes:
            self.tick()
            if isinstance(node, ast.Assign):
                value = self.expr(node.value)
                for target in node.targets:
                    self.variables[target.id] = value
            elif isinstance(node, ast.Expr):
                self.expr(node.value)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test) else node.orelse)
            elif isinstance(node, ast.For):
                values = self.expr(node.iter)
                if not isinstance(values, (list, tuple, range)) or len(values) > 100:
                    raise ProgramError("for requires at most 100 list/range items")
                for value in values:
                    self.tick()
                    self.variables[node.target.id] = value
                    try:
                        self.block(node.body)
                    except _Continue:
                        continue
                    except _Break:
                        break
                else:
                    self.block(node.orelse)
            elif isinstance(node, ast.Break):
                raise _Break()
            elif isinstance(node, ast.Continue):
                raise _Continue()

    def expr(self, node):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.variables[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expr(x) for x in node.elts]
        if isinstance(node, ast.Dict):
            return {self.expr(k): self.expr(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            return self.expr(node.value)[self.expr(node.slice)]
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand)
            if not isinstance(node.op, ast.Not) and type(value) not in (int, float):
                raise ProgramError("arithmetic requires numbers")
            return OPS[type(node.op)](value)
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left), self.expr(node.right)
            if type(a) not in (int, float) or type(b) not in (int, float):
                raise ProgramError("arithmetic requires numbers")
            value = OPS[type(node.op)](a, b)
            if abs(value) > 1e12:
                raise ProgramError("arithmetic result out of bounds")
            return value
        if isinstance(node, ast.BoolOp):
            for child in node.values:
                result = self.expr(child)
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                    return result
            return result
        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            for op, child in zip(node.ops, node.comparators):
                right = self.expr(child)
                if not OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            args = [self.expr(x) for x in node.args]
            kwargs = {k.arg: self.expr(k.value) for k in node.keywords}
            if isinstance(node.func, ast.Name):
                if kwargs or not 1 <= len(args) <= 3 or any(type(x) is not int for x in args):
                    raise ProgramError("range requires 1..3 integers")
                result = range(*args)
                if len(result) > 100:
                    raise ProgramError("range exceeds 100 iterations")
                return result
            self.calls -= 1
            if self.calls < 0:
                raise ProgramError("MCP call budget exhausted")
            name = node.func.attr
            event = {"method": name, "args": args, "kwargs": kwargs}
            try:
                result = getattr(self.mcp, name)(*args, **kwargs)
                # Deep copy JSON values; programs never receive a driver object.
                result = json.loads(json.dumps(result, allow_nan=False))
            except Exception as exc:
                self.emit(dict(event, status="error", error=f"{type(exc).__name__}: {exc}",
                               result=getattr(exc, "result", None)))
                raise
            self.emit(dict(event, status="returned", result=result))
            return result
        raise ProgramError(f"unsupported expression: {type(node).__name__}")
