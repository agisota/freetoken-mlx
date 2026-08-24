from __future__ import annotations

import argparse
import hashlib
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import freetoken  # noqa: E402
from freetoken.mlx_generate import (  # noqa: E402
    GenerationOptions,
    _make_generate,
    _validate_options,
    build_parser,
    main,
)


class FakeMX:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class FakeTokenizer:
    chat_template = None

    def __init__(self, templated: str | None = None):
        self.templated = templated
        if templated is not None:
            self.chat_template = "template"
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.templated


def response(
    text: str,
    tokens: int,
    *,
    from_draft: bool = False,
    prompt_tokens: int = 10,
    prompt_tps: float = 5.0,
    generation_tps: float = 2.0,
    finish_reason: str | None = None,
):
    return SimpleNamespace(
        text=text,
        generation_tokens=tokens,
        from_draft=from_draft,
        prompt_tokens=prompt_tokens,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        finish_reason=finish_reason,
    )


@contextmanager
def fake_backend(*, result=0, raises: BaseException | None = None):
    module_name = "freetoken.mlx_backend"
    old_module = sys.modules.get(module_name)
    had_attr = hasattr(freetoken, "mlx_backend")
    old_attr = getattr(freetoken, "mlx_backend", None)
    module = ModuleType(module_name)
    original_generate = object()
    module._generate = original_generate
    seen = {}

    def backend_parser():
        parser = argparse.ArgumentParser(prog="ft generate")
        parser.add_argument("--backend", choices=("mlx",), required=True)
        parser.add_argument("--model", required=True)
        parser.add_argument("--prompt", default="Hello")
        parser.add_argument("--raw-prompt", action="store_true")
        parser.add_argument("--max-tokens", type=int, default=1)
        parser.add_argument("--draft-model")
        parser.add_argument("--num-draft-tokens", type=int, default=2)
        return parser

    def backend_main(argv):
        seen["argv"] = list(argv)
        seen["patched"] = module._generate is not original_generate
        if raises is not None:
            raise raises
        return result

    module.build_parser = backend_parser
    module.main = backend_main
    sys.modules[module_name] = module
    setattr(freetoken, "mlx_backend", module)
    try:
        yield module, original_generate, seen
    finally:
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module
        if had_attr:
            setattr(freetoken, "mlx_backend", old_attr)
        else:
            delattr(freetoken, "mlx_backend")


class MLXGenerateTest(unittest.TestCase):
    def test_default_wrapper_preserves_stream_arguments_and_adds_metrics(self):
        mx = FakeMX()
        tokenizer = FakeTokenizer()
        calls = []

        def stream_generate(model, actual_tokenizer, prompt, **kwargs):
            calls.append((model, actual_tokenizer, prompt, kwargs))
            yield response("A", 1, from_draft=True)
            yield response("B", 2, from_draft=False)
            # mlx-lm emits a final response with the same token count. It must
            # not double-count speculative acceptance.
            yield response("", 2, from_draft=False, finish_reason="length")

        args = SimpleNamespace(
            prompt="Hello",
            raw_prompt=True,
            max_tokens=2,
            num_draft_tokens=2,
            draft_model=None,
        )
        stdout = StringIO()
        with redirect_stdout(stdout):
            report = _make_generate(GenerationOptions())(
                mx, stream_generate, "model", tokenizer, None, args
            )

        self.assertEqual(stdout.getvalue(), "AB\n")
        self.assertEqual(mx.synchronize_calls, 1)
        self.assertEqual(calls[0][2], "Hello")
        self.assertEqual(calls[0][3], {"max_tokens": 2})
        self.assertEqual(report["runtime_report_schema_version"], 1)
        self.assertEqual(report["generated_tokens"], 2)
        self.assertEqual(report["prompt"]["tokens"], 10)
        self.assertEqual(report["prompt"]["seconds"], 2.0)
        self.assertEqual(report["decode"]["seconds"], 1.0)
        self.assertEqual(report["decode"]["finish_reason"], "length")
        self.assertEqual(
            report["prompt"]["sha256"], hashlib.sha256(b"Hello").hexdigest()
        )
        self.assertEqual(report["speculative"]["accepted_tokens"], 1)
        self.assertEqual(report["speculative"]["draft_output_fraction"], 0.5)
        self.assertIsNone(report["kv_cache"]["max_size"])
        self.assertGreaterEqual(report["time_to_first_output_seconds"], 0.0)
        self.assertGreaterEqual(report["elapsed_seconds"], 0.0)

    def test_optional_kv_and_prefill_controls_are_forwarded(self):
        mx = FakeMX()
        captured = {}

        def stream_generate(*args, **kwargs):
            captured.update(kwargs)
            yield response("x", 1, finish_reason="length")

        options = GenerationOptions(
            max_kv_size=None,
            kv_bits=4,
            kv_group_size=32,
            quantized_kv_start=256,
            prefill_step_size=512,
        )
        args = SimpleNamespace(
            prompt="p",
            raw_prompt=True,
            max_tokens=1,
            num_draft_tokens=2,
            draft_model=None,
        )
        with redirect_stdout(StringIO()):
            report = _make_generate(options)(
                mx, stream_generate, object(), FakeTokenizer(), None, args
            )

        self.assertEqual(captured, {
            "max_tokens": 1,
            "prefill_step_size": 512,
            "kv_bits": 4,
            "kv_group_size": 32,
            "quantized_kv_start": 256,
        })
        self.assertEqual(report["kv_cache"], {
            "max_size": None,
            "bits": 4,
            "group_size": 32,
            "quantized_start": 256,
            "prefill_step_size": 512,
        })

    def test_draft_kwargs_are_forwarded_and_final_response_is_not_double_counted(self):
        mx = FakeMX()
        draft = object()
        captured = {}

        def stream_generate(*args, **kwargs):
            captured.update(kwargs)
            yield response("a", 1, from_draft=True)
            yield response("b", 2, from_draft=True)
            yield response("", 2, from_draft=True, finish_reason="length")

        args = SimpleNamespace(
            prompt="p",
            raw_prompt=True,
            max_tokens=2,
            num_draft_tokens=3,
            draft_model="draft-id",
        )
        with redirect_stdout(StringIO()):
            report = _make_generate(GenerationOptions(kv_bits=8))(
                mx, stream_generate, object(), FakeTokenizer(), draft, args
            )

        self.assertIs(captured["draft_model"], draft)
        self.assertEqual(captured["num_draft_tokens"], 3)
        self.assertEqual(captured["kv_bits"], 8)
        self.assertEqual(report["speculative"]["accepted_tokens"], 2)
        self.assertEqual(report["speculative"]["draft_output_fraction"], 1.0)

    def test_chat_template_is_hashed_and_passed_to_mlx_lm(self):
        tokenizer = FakeTokenizer("templated prompt")
        captured = {}

        def stream_generate(model, actual_tokenizer, prompt, **kwargs):
            captured["prompt"] = prompt
            yield response("x", 1)

        args = SimpleNamespace(
            prompt="raw",
            raw_prompt=False,
            max_tokens=1,
            num_draft_tokens=2,
            draft_model=None,
        )
        with redirect_stdout(StringIO()):
            report = _make_generate(GenerationOptions())(
                FakeMX(), stream_generate, object(), tokenizer, None, args
            )

        self.assertEqual(captured["prompt"], "templated prompt")
        self.assertEqual(
            report["prompt"]["sha256"],
            hashlib.sha256(b"templated prompt").hexdigest(),
        )
        self.assertEqual(tokenizer.calls[0][1]["add_generation_prompt"], True)
        self.assertEqual(tokenizer.calls[0][1]["tokenize"], False)

    def test_nonfinite_upstream_rates_are_reported_as_null(self):
        def stream_generate(*args, **kwargs):
            yield response(
                "x", 1, prompt_tps=float("nan"), generation_tps=float("inf")
            )

        args = SimpleNamespace(
            prompt="p", raw_prompt=True, max_tokens=1,
            num_draft_tokens=2, draft_model=None,
        )
        with redirect_stdout(StringIO()):
            report = _make_generate(GenerationOptions())(
                FakeMX(), stream_generate, object(), FakeTokenizer(), None, args
            )
        self.assertIsNone(report["prompt"]["tokens_per_second"])
        self.assertIsNone(report["prompt"]["seconds"])
        self.assertIsNone(report["decode"]["tokens_per_second"])
        self.assertIsNone(report["decode"]["seconds"])

    def test_empty_stream_is_a_controlled_runtime_error(self):
        args = SimpleNamespace(
            prompt="p", raw_prompt=True, max_tokens=1,
            num_draft_tokens=2, draft_model=None,
        )
        with redirect_stdout(StringIO()), self.assertRaisesRegex(
            RuntimeError, "no generation response"
        ):
            _make_generate(GenerationOptions())(
                FakeMX(), lambda *args, **kwargs: iter(()),
                object(), FakeTokenizer(), None, args,
            )

    def test_option_validation_rejects_unsupported_combinations(self):
        base = dict(
            max_kv_size=None,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=5000,
            prefill_step_size=None,
            draft_model=None,
        )
        invalid = (
            ({**base, "max_kv_size": 4}, "four retained prefix"),
            ({**base, "quantized_kv_start": -1}, "cannot be negative"),
            ({**base, "prefill_step_size": 0}, "at least 1"),
            ({**base, "max_kv_size": 5, "draft_model": "draft"}, "not supported"),
            ({**base, "max_kv_size": 5, "kv_bits": 4}, "RotatingKVCache"),
        )
        for values, message in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(
                ValueError, message
            ):
                _validate_options(SimpleNamespace(**values))

    def test_main_strips_wrapper_flags_and_restores_backend_hook(self):
        with fake_backend(result=7) as (backend, original, seen):
            result = main([
                "--backend", "mlx",
                "--model", "target",
                "--max-kv-size", "2048",
                "--prefill-step-size", "512",
            ])
            self.assertEqual(result, 7)
            self.assertTrue(seen["patched"])
            self.assertEqual(seen["argv"], [
                "--backend", "mlx", "--model", "target"
            ])
            self.assertIs(backend._generate, original)

    def test_backend_hook_is_restored_after_exception(self):
        error = RuntimeError("backend failure")
        with fake_backend(raises=error) as (backend, original, seen):
            with self.assertRaisesRegex(RuntimeError, "backend failure"):
                main(["--backend", "mlx", "--model", "target"])
            self.assertTrue(seen["patched"])
            self.assertIs(backend._generate, original)

    def test_none_argv_uses_process_arguments(self):
        with fake_backend(result=3) as (backend, original, seen), patch.object(
            sys,
            "argv",
            ["ft", "--backend", "mlx", "--model", "target", "--kv-bits", "8"],
        ):
            self.assertEqual(main(None), 3)
            self.assertEqual(
                seen["argv"], ["--backend", "mlx", "--model", "target"]
            )
            self.assertIs(backend._generate, original)

    def test_combined_help_exposes_kv_controls(self):
        with fake_backend():
            help_text = build_parser().format_help()
        self.assertIn("--max-kv-size", help_text)
        self.assertIn("--kv-bits {4,8}", help_text)
        self.assertIn("--prefill-step-size", help_text)

    def test_invalid_cli_option_exits_with_argparse_code_two(self):
        with fake_backend(), self.assertRaises(SystemExit) as raised:
            main([
                "--backend", "mlx", "--model", "target",
                "--max-kv-size", "4",
            ])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
