"""Tests for the TYPE_EXPR annotation format and backtick type expressions."""

import ast
import textwrap
import unittest

import annotationlib
from annotationlib import Format, TypeExpr, get_annotations
from typing import AssignableTo, ConditionalType, Unpack, Union


class MyClass:
    pass


class TestSourceFormat(unittest.TestCase):
    """The SOURCE format: exact annotation source stored by the compiler."""

    def test_function(self):
        def f(a: int, b: list[int] | None = None, *args: int, **kw: str) -> None: ...

        self.assertEqual(
            get_annotations(f, format=Format.SOURCE),
            {
                "a": "int",
                "b": "list[int] | None",
                "args": "int",
                "kw": "str",
                "return": "None",
            },
        )

    def test_class(self):
        class C:
            x: int
            y: "MyClass"
            z: int if AssignableTo[float, int] else str

        self.assertEqual(
            get_annotations(C, format=Format.SOURCE),
            {
                "x": "int",
                "y": "'MyClass'",
                "z": "int if AssignableTo[float, int] else str",
            },
        )

    def test_not_optimized(self):
        # The stored source must not be affected by constant folding.
        def f(n: 1 | 2 | 3): ...

        self.assertEqual(
            get_annotations(f, format=Format.SOURCE), {"n": "1 | 2 | 3"}
        )

    def test_conditional_annotations(self):
        class C:
            if True:
                x: int
            else:
                y: str

        self.assertEqual(
            get_annotations(C, format=Format.SOURCE), {"x": "int"}
        )

    def test_star_unpack(self):
        def f(*args: *tuple[int, str]): ...

        self.assertEqual(
            get_annotations(f, format=Format.SOURCE),
            {"args": "*tuple[int, str]"},
        )

    def test_evaluate_functions(self):
        type Alias[T: (int, str)] = list[T] if COND else T

        self.assertEqual(
            annotationlib.call_evaluate_function(
                Alias.evaluate_value, Format.SOURCE
            ),
            "list[T] if COND else T",
        )
        tp = Alias.__type_params__[0]
        self.assertEqual(
            annotationlib.call_evaluate_function(
                tp.evaluate_constraints, Format.SOURCE
            ),
            "(int, str)",
        )

    def test_fallback_for_manual_annotate(self):
        # Annotate functions that only support VALUE fall back to the
        # STRING reconstruction.
        # Written defensively (int literal, default arg) because the STRING
        # fallback runs this function with fake globals.
        def annotate(format, /, __NotImplementedError=NotImplementedError):
            if format > 2:
                raise __NotImplementedError(format)
            return {"x": list[int]}

        self.assertEqual(
            annotationlib.call_annotate_function(annotate, Format.SOURCE),
            {"x": "list[int]"},
        )


class TestASTFormat(unittest.TestCase):
    def test_simple(self):
        def f(a: list[int] | None): ...

        anns = get_annotations(f, format=Format.AST)
        self.assertIsInstance(anns["a"], ast.BinOp)
        self.assertEqual(ast.unparse(anns["a"]), "list[int] | None")

    def test_conditional(self):
        def f() -> int if AssignableTo[float, int] else str: ...

        anns = get_annotations(f, format=Format.AST)
        self.assertIsInstance(anns["return"], ast.IfExp)

    def test_unoptimized(self):
        def f(n: 1 | 2 | 3): ...

        anns = get_annotations(f, format=Format.AST)
        self.assertIsInstance(anns["n"], ast.BinOp)
        self.assertEqual(ast.unparse(anns["n"]), "1 | 2 | 3")


class TestTypeExprFormat(unittest.TestCase):
    def test_regular_annotations_unchanged(self):
        def f(a: int, b: list[MyClass] | None, c: tuple[int, str]): ...

        self.assertEqual(
            get_annotations(f, format=Format.TYPE_EXPR),
            {
                "a": int,
                "b": Union[list[MyClass], None],
                "c": tuple[int, str],
            },
        )

    def test_conditional(self):
        def f(x: float) -> int if AssignableTo[float, int] else str: ...

        anns = get_annotations(f, format=Format.TYPE_EXPR)
        self.assertEqual(
            anns["return"],
            ConditionalType(int, str, AssignableTo[float, int]),
        )
        # ... while the VALUE format collapses the conditional at runtime.
        self.assertEqual(get_annotations(f)["return"], int)

    def test_name_resolution_closure(self):
        def outer():
            class Local:
                pass

            def g(x: Local): ...

            return g, Local

        g, Local = outer()
        self.assertEqual(
            get_annotations(g, format=Format.TYPE_EXPR), {"x": Local}
        )

    def test_name_resolution_class_namespace(self):
        class C:
            Inner = MyClass
            x: Inner

        self.assertEqual(
            get_annotations(C, format=Format.TYPE_EXPR), {"x": MyClass}
        )

    def test_name_resolution_type_params(self):
        def g[T](x: T if AssignableTo[T, int] else list[T]): ...

        (T,) = g.__type_params__
        anns = get_annotations(g, format=Format.TYPE_EXPR)
        self.assertEqual(
            anns["x"], ConditionalType(T, list[T], AssignableTo[T, int])
        )

    def test_unresolved_name(self):
        def f(x: Undefined): ...

        with self.assertRaises(NameError):
            get_annotations(f, format=Format.TYPE_EXPR)

    def test_stringized_annotation(self):
        def f(x: "MyClass"): ...

        self.assertEqual(
            get_annotations(f, format=Format.TYPE_EXPR), {"x": MyClass}
        )

    def test_string_inside_subscript_preserved(self):
        # Nested strings (e.g. Literal values) are not treated as forward
        # references.
        from typing import Literal

        def f(x: Literal["a", "b"]): ...

        self.assertEqual(
            get_annotations(f, format=Format.TYPE_EXPR),
            {"x": Literal["a", "b"]},
        )

    def test_star_unpack(self):
        def f(*args: *tuple[int, str]): ...

        self.assertEqual(
            get_annotations(f, format=Format.TYPE_EXPR),
            {"args": Unpack[tuple[int, str]]},
        )

    def test_unsupported_syntax(self):
        def f(x: lambda: int): ...

        with self.assertRaises(SyntaxError):
            get_annotations(f, format=Format.TYPE_EXPR)

    def test_eval_type_expr(self):
        self.assertIs(annotationlib.eval_type_expr("int"), int)
        self.assertEqual(
            annotationlib.eval_type_expr(
                "int if AssignableTo[float, int] else str",
                globals={"AssignableTo": AssignableTo},
            ),
            ConditionalType(int, str, AssignableTo[float, int]),
        )
        node = ast.parse("list[int]", mode="eval").body
        self.assertEqual(annotationlib.eval_type_expr(node), list[int])


class TestBacktickSyntax(unittest.TestCase):
    def test_basic(self):
        te = `int`
        self.assertIsInstance(te, TypeExpr)
        self.assertEqual(te.source, "int")
        self.assertIs(te.evaluate(), int)
        self.assertIs(te.evaluate(format=Format.VALUE), int)
        self.assertIsInstance(te.ast, ast.Name)
        self.assertEqual(repr(te), "<TypeExpr `int`>")

    def test_conditional(self):
        te = `int if AssignableTo[float, int] else str`
        self.assertEqual(
            te.evaluate(),
            ConditionalType(int, str, AssignableTo[float, int]),
        )
        # VALUE evaluation collapses the conditional.
        self.assertIs(te.evaluate(format=Format.VALUE), int)
        self.assertEqual(
            te.evaluate(format=Format.SOURCE),
            "int if AssignableTo[float, int] else str",
        )

    def test_lazy(self):
        # The enclosed expression is not evaluated when the display is.
        te = `Undefined1 if Undefined2 else Undefined3`
        self.assertEqual(
            te.source, "Undefined1 if Undefined2 else Undefined3"
        )
        with self.assertRaises(NameError):
            te.evaluate()

    def test_closure(self):
        def make(x):
            return `list[x]`

        self.assertEqual(make(bytes).evaluate(), list[bytes])

    def test_class_namespace(self):
        class C:
            Inner = MyClass
            te = `Inner`

        self.assertIs(C.te.evaluate(), MyClass)

    def test_parameterizing_generics(self):
        class Gen[T]:
            pass

        te = `int if AssignableTo[float, int] else str`
        alias = Gen[te]
        self.assertEqual(alias.__args__, (te,))
        self.assertEqual(
            alias.__args__[0].evaluate(),
            ConditionalType(int, str, AssignableTo[float, int]),
        )

    def test_equality(self):
        self.assertEqual(`int`, `int`)
        self.assertNotEqual(`int`, `str`)
        self.assertEqual(hash(`int`), hash(`int`))
        self.assertNotEqual(`int`, "int")

    def test_in_annotation(self):
        class Gen[T]:
            pass

        def f(x: Gen[`int if AssignableTo[float, int] else str`]): ...

        anns = get_annotations(f, format=Format.TYPE_EXPR)
        self.assertEqual(
            anns["x"],
            Gen[ConditionalType(int, str, AssignableTo[float, int])],
        )
        source = get_annotations(f, format=Format.SOURCE)
        self.assertEqual(
            source["x"], "Gen[`int if AssignableTo[float, int] else str`]"
        )
        string = get_annotations(f, format=Format.STRING)
        self.assertEqual(string, source)

    def test_ast_roundtrip(self):
        tree = ast.parse("x = `int if C else str`")
        node = tree.body[0].value
        self.assertIsInstance(node, ast.TypeExpr)
        self.assertIsInstance(node.body, ast.IfExp)
        self.assertEqual(ast.unparse(tree), "x = `int if C else str`")
        compile(tree, "<test>", "exec")

    def test_nested_transparent(self):
        te = ``int``
        self.assertIsInstance(te, TypeExpr)
        # The inner display evaluates to a TypeExpr as well...
        inner = te.evaluate(format=Format.VALUE)
        self.assertIsInstance(inner, TypeExpr)
        # ...but with TYPE_EXPR semantics backticks are transparent.
        self.assertIs(te.evaluate(), int)

    def test_syntax_errors(self):
        cases = [
            "`int` = 1",
            "`int",
            "`",
            "``",
            "`int)`",
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(SyntaxError):
                    compile(case, "<test>", "exec")

    def test_no_yield_or_walrus(self):
        cases = [
            "def f():\n    yield `(yield)`",
            "`(x := 1)`",
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(SyntaxError):
                    compile(case, "<test>", "exec")


class TestDemoConstructs(unittest.TestCase):
    def test_conditional_type(self):
        ct = ConditionalType(int, str, AssignableTo[float, int])
        self.assertIs(ct.if_true, int)
        self.assertIs(ct.if_false, str)
        self.assertEqual(ct.condition, AssignableTo[float, int])
        self.assertEqual(
            repr(ct),
            "ConditionalType(if_true=int, if_false=str, "
            "condition=AssignableTo[float, int])",
        )

    def test_assignable_to(self):
        cond = AssignableTo[float, int]
        self.assertIs(cond.source, float)
        self.assertIs(cond.target, int)
        self.assertTrue(cond)
        with self.assertRaises(TypeError):
            AssignableTo[int]


if __name__ == "__main__":
    unittest.main()
