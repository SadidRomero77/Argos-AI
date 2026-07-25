"""Ejecuta el banco de tool-calling contra uno o varios modelos locales.

    uv run python -m argos.evals qwen3:4b qwen3:8b

Es la evaluación que decide si un modelo sirve como cerebro. Mide tasa de acierto
en elegir la skill y en pasarle parámetros válidos — no perplexity.
"""

from __future__ import annotations

import argparse
import sys

from argos.config import Settings
from argos.evals.toolcalling import evaluar, imprimir_comparativa
from argos.models.local import LocalProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="argos.evals", description="Banco de tool-calling")
    parser.add_argument("modelos", nargs="+", help="modelos a evaluar, ej. qwen3:4b qwen3:8b")
    parser.add_argument(
        "--url", default="http://localhost:11434/v1", help="endpoint OpenAI-compatible"
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=16384,
        help=(
            "ventana de contexto. El default de Ollama (4096) es insuficiente: "
            "el prompt de sistema más las definiciones de skills ya lo consumen"
        ),
    )
    args = parser.parse_args(argv)

    settings = Settings()
    informes = []

    for modelo in args.modelos:
        provider = LocalProvider(model=modelo, base_url=args.url, num_ctx=args.num_ctx)
        ok, mensaje = provider.health()
        if not ok:
            print(f"error: {mensaje}", file=sys.stderr)
            return 2

        print(f"\n╭─ {modelo}  (num_ctx={args.num_ctx})")
        informes.append(evaluar(provider, settings=settings))

    imprimir_comparativa(informes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
