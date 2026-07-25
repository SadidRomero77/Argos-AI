"""CLI de ARGOS. Un turno del agente desde la terminal.

argos "¿cuántos tests tiene el proyecto?"
argos --approve ask "borra las notas viejas"   # aprobar interactivamente
argos --local "hola"                           # sin salir a ninguna API
"""

from __future__ import annotations

import argparse
import sys

from argos.config import get_settings
from argos.harness.loop import AgentLoop, StopReason
from argos.harness.permissions import (
    AlwaysAllowApprover,
    ConsoleApprover,
    DenyAllApprover,
    PermissionGate,
)
from argos.models.base import TaskKind
from argos.models.router import ModelRouter
from argos.tools.registry import default_registry
from argos.trace import Tracer

_APROBADORES = {
    "deny": DenyAllApprover,
    "ask": ConsoleApprover,
    "yes": AlwaysAllowApprover,
}

_MOTIVOS = {
    StopReason.MAX_ITERATIONS: "se agotaron las iteraciones",
    StopReason.TOKEN_BUDGET: "se agotó el presupuesto de tokens",
    StopReason.USD_BUDGET: "se agotó el presupuesto en dólares",
    StopReason.REFUSED: "el modelo declinó la petición",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argos", description="Agente ARGOS")
    parser.add_argument("mensaje", help="qué quieres que haga el agente")
    parser.add_argument(
        "--kind",
        choices=[str(k) for k in TaskKind],
        default=str(TaskKind.ROUTINE),
        help="tipo de tarea; determina el modelo (por defecto: routine)",
    )
    parser.add_argument(
        "--approve",
        choices=list(_APROBADORES),
        default="deny",
        help=(
            "cómo resolver los permisos 'ask': deny (por defecto, falla cerrado), "
            "ask (pregunta por consola), yes (aprueba todo — sólo desarrollo)"
        ),
    )
    parser.add_argument(
        "--local", action="store_true", help="fuerza modo local; no llama a ninguna API externa"
    )
    parser.add_argument("--quiet", action="store_true", help="sólo la respuesta, sin resumen")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = get_settings()
    if args.local:
        settings.models.local_only = True

    tracer = Tracer(settings=settings)
    gate = PermissionGate(approver=_APROBADORES[args.approve](), tracer=tracer, settings=settings)

    try:
        router = ModelRouter.from_settings(settings=settings, tracer=tracer)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    registry = default_registry(gate=gate, tracer=tracer)
    loop = AgentLoop(router=router, registry=registry, tracer=tracer, settings=settings)

    result = loop.run(args.mensaje, kind=TaskKind(args.kind))

    print(result.text or "(sin respuesta)")

    if not args.quiet:
        print(
            f"\n— {result.iterations} iteración(es) · "
            f"{result.usage.total_tokens} tokens · ${result.usd:.4f}",
            file=sys.stderr,
        )
        if result.skill_calls:
            print(f"— skills: {', '.join(result.skill_calls)}", file=sys.stderr)
        if not result.ok:
            print(f"— interrumpido: {_MOTIVOS[result.stopped_because]}", file=sys.stderr)
        print(f"— traza: {tracer.path}", file=sys.stderr)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
