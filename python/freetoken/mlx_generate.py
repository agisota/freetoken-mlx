"""Extended MLX generation controls and metrics layered over the core backend."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import dataclass
from typing import Sequence

RUNTIME_REPORT_SCHEMA_VERSION = 1
DEFAULT_QUANTIZED_KV_START = 5000


@dataclass(frozen=True)
class GenerationOptions:
    max_kv_size: int | None = None
    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START
    prefill_step_size: int | None = None


def _add_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help=(
            "maximum rotating KV-cache length for non-speculative generation; "
            "keeps the first four tokens"
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=(4, 8),
        help="quantize KV cache to 4 or 8 bits after --quantized-kv-start",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        choices=(32, 64, 128),
        default=64,
        help="KV quantization group size (default: 64)",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="token offset at which KV quantization begins (default: 5000)",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        help="prompt tokens processed per prefill chunk",
    )


def _extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_options(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    from freetoken.mlx_backend import build_parser as build_backend_parser

    parser = build_backend_parser()
    _add_options(parser)
    return parser


def _validate_options(args: argparse.Namespace) -> GenerationOptions:
    if args.max_kv_size is not None and args.max_kv_size <= 4:
        raise ValueError(
            "--max-kv-size must exceed the four retained prefix tokens"
        )
    if args.quantized_kv_start < 0:
        raise ValueError("--quantized-kv-start cannot be negative")
    if args.prefill_step_size is not None and args.prefill_step_size < 1:
        raise ValueError("--prefill-step-size must be at least 1")
    if args.draft_model is not None and args.max_kv_size is not None:
        raise ValueError(
            "--max-kv-size is not supported with --draft-model by mlx-lm 0.31"
        )
    if args.max_kv_size is not None and args.kv_bits is not None:
        raise ValueError(
            "--max-kv-size cannot be combined with --kv-bits because "
            "mlx-lm 0.31 cannot quantize RotatingKVCache"
        )
    return GenerationOptions(
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        prefill_step_size=args.prefill_step_size,
    )


def _finite_float(value, *, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _make_generate(options: GenerationOptions):
    def generate(mx, stream_generate, model, tokenizer, draft_model, args):
        prompt = args.prompt
        if not args.raw_prompt and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        prompt_bytes = prompt.encode("utf-8")
        generation_options = {}
        if draft_model is not None:
            generation_options.update(
                draft_model=draft_model,
                num_draft_tokens=args.num_draft_tokens,
            )
        if options.max_kv_size is not None:
            generation_options["max_kv_size"] = options.max_kv_size
        if options.prefill_step_size is not None:
            generation_options["prefill_step_size"] = options.prefill_step_size
        if options.kv_bits is not None:
            generation_options.update(
                kv_bits=options.kv_bits,
                kv_group_size=options.kv_group_size,
                quantized_kv_start=options.quantized_kv_start,
            )

        started = time.perf_counter()
        first_output_seconds = None
        generated = 0
        accepted_draft_tokens = 0
        last_response = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            **generation_options,
        ):
            if first_output_seconds is None:
                first_output_seconds = time.perf_counter() - started
            print(response.text, end="", flush=True)
            response_tokens = int(response.generation_tokens)
            if response_tokens > generated and response.from_draft:
                accepted_draft_tokens += 1
            generated = response_tokens
            last_response = response
        mx.synchronize()
        elapsed_seconds = time.perf_counter() - started
        print()
        if last_response is None:
            raise RuntimeError("mlx-lm returned no generation response")

        prompt_tokens = int(last_response.prompt_tokens)
        prompt_tps = _finite_float(last_response.prompt_tps)
        generation_tps = _finite_float(last_response.generation_tps)
        prompt_seconds = (
            prompt_tokens / prompt_tps
            if prompt_tps is not None and prompt_tps > 0
            else None
        )
        decode_seconds = (
            generated / generation_tps
            if generation_tps is not None and generation_tps > 0
            else None
        )
        return {
            "runtime_report_schema_version": RUNTIME_REPORT_SCHEMA_VERSION,
            "generated_tokens": generated,
            "elapsed_seconds": elapsed_seconds,
            "time_to_first_output_seconds": first_output_seconds,
            "prompt": {
                "utf8_bytes": len(prompt_bytes),
                "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "tokens": prompt_tokens,
                "tokens_per_second": prompt_tps,
                "seconds": prompt_seconds,
            },
            "decode": {
                "tokens": generated,
                "tokens_per_second": generation_tps,
                "seconds": decode_seconds,
                "finish_reason": getattr(last_response, "finish_reason", None),
            },
            "kv_cache": {
                "max_size": options.max_kv_size,
                "bits": options.kv_bits,
                "group_size": (
                    options.kv_group_size if options.kv_bits is not None else None
                ),
                "quantized_start": (
                    options.quantized_kv_start
                    if options.kv_bits is not None
                    else None
                ),
                "prefill_step_size": options.prefill_step_size,
            },
            "speculative": {
                "enabled": draft_model is not None,
                "draft_model": args.draft_model,
                "num_draft_tokens": (
                    args.num_draft_tokens if draft_model is not None else 0
                ),
                "accepted_tokens": accepted_draft_tokens,
                "draft_output_fraction": (
                    accepted_draft_tokens / generated if generated else 0.0
                ),
            },
        }

    return generate


def main(argv: Sequence[str] | None = None) -> int:
    from freetoken import mlx_backend

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    try:
        options = _validate_options(parsed)
    except ValueError as exc:
        parser.error(str(exc))

    _, backend_arguments = _extra_parser().parse_known_args(arguments)
    original_generate = mlx_backend._generate
    mlx_backend._generate = _make_generate(options)
    try:
        return mlx_backend.main(backend_arguments)
    finally:
        mlx_backend._generate = original_generate


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GenerationOptions",
    "build_parser",
    "main",
]
