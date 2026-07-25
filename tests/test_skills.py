"""Las skills se prueban SIN modelo. Si alguna necesitara un LLM, estaría mal diseñada.

Estas pruebas cubren el confinamiento de rutas y la allowlist de ejecutables, que
son los dos controles de seguridad que impiden que una skill se convierta en un
shell arbitrario.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from argos.skills.base import ConfinementError, Skill, SkillResult, resolve_confined
from argos.skills.fs import Glob, ReadFile, WriteNote
from argos.skills.shell import RunCommand

# ─────────────────────────── confinamiento de rutas ───────────────────────────


def test_resolve_confined_permite_rutas_dentro(tmp_path):
    (tmp_path / "sub").mkdir()
    assert resolve_confined("sub/x.txt", tmp_path) == (tmp_path / "sub" / "x.txt").resolve()


def test_resolve_confined_rechaza_travesia_con_puntos(tmp_path):
    with pytest.raises(ConfinementError):
        resolve_confined("../../etc/passwd", tmp_path)


def test_resolve_confined_rechaza_ruta_absoluta_externa(tmp_path):
    with pytest.raises(ConfinementError):
        resolve_confined("/etc/passwd", tmp_path)


def test_resolve_confined_rechaza_symlink_que_escapa(tmp_path):
    """Caso que una comprobación puramente sintáctica dejaría pasar."""
    fuera = tmp_path.parent / "fuera_secreto.txt"
    fuera.write_text("secreto")
    raiz = tmp_path / "raiz"
    raiz.mkdir()
    (raiz / "atajo").symlink_to(fuera)

    with pytest.raises(ConfinementError):
        resolve_confined("atajo", raiz)


# ──────────────────────────────── read_file ───────────────────────────────────


def test_read_file_lee(tmp_path):
    (tmp_path / "hola.txt").write_text("contenido", encoding="utf-8")
    result = ReadFile(root=tmp_path).execute({"path": "hola.txt"})
    assert result.ok
    assert result.output == "contenido"


def test_read_file_bloquea_travesia(tmp_path):
    result = ReadFile(root=tmp_path).execute({"path": "../../../etc/passwd"})
    assert not result.ok
    assert "fuera de la raíz" in result.error


def test_read_file_informa_si_no_existe(tmp_path):
    result = ReadFile(root=tmp_path).execute({"path": "fantasma.txt"})
    assert not result.ok
    assert "no existe" in result.error


def test_read_file_rechaza_parametros_invalidos(tmp_path):
    result = ReadFile(root=tmp_path).execute({})
    assert not result.ok
    assert "parámetros inválidos" in result.error


def test_read_file_trunca_archivos_enormes(tmp_path):
    (tmp_path / "grande.txt").write_text("x" * 300_000, encoding="utf-8")
    result = ReadFile(root=tmp_path).execute({"path": "grande.txt"})
    assert result.ok
    assert result.data["truncated"] is True
    assert len(result.output) < 250_000


# ──────────────────────────────── write_note ──────────────────────────────────


def test_write_note_guarda(tmp_path):
    result = WriteNote(notes_dir=tmp_path).execute({"title": "Mi Nota", "content": "hola"})
    assert result.ok
    assert (tmp_path / "mi-nota.md").read_text(encoding="utf-8") == "hola\n"


def test_write_note_neutraliza_un_titulo_con_travesia(tmp_path):
    """El título es texto del modelo: no puede controlar la ruta de escritura."""
    result = WriteNote(notes_dir=tmp_path).execute(
        {"title": "../../fuera", "content": "intento de fuga"}
    )
    assert result.ok
    assert not (tmp_path.parent / "fuera.md").exists()
    assert list(tmp_path.glob("*.md"))  # se escribió, pero dentro


def test_write_note_append(tmp_path):
    skill = WriteNote(notes_dir=tmp_path)
    skill.execute({"title": "diario", "content": "linea 1"})
    skill.execute({"title": "diario", "content": "linea 2", "append": True})
    texto = (tmp_path / "diario.md").read_text(encoding="utf-8")
    assert "linea 1" in texto and "linea 2" in texto


def test_write_note_rechaza_contenido_vacio(tmp_path):
    result = WriteNote(notes_dir=tmp_path).execute({"title": "vacia", "content": "   "})
    assert not result.ok
    assert "vacía" in result.error


# ──────────────────────────────── run_command ─────────────────────────────────


def test_run_command_ejecuta_lo_permitido():
    result = RunCommand().execute({"argv": ["echo_no_existe_pero_ls_si"]})
    assert not result.ok  # no está en la allowlist

    result = RunCommand().execute({"argv": ["ls", "-d", "."]})
    assert result.ok
    assert result.data["returncode"] == 0


def test_run_command_bloquea_ejecutable_fuera_de_la_allowlist():
    result = RunCommand().execute({"argv": ["rm", "-rf", "/"]})
    assert not result.ok
    assert "lista blanca" in result.error


def test_run_command_bloquea_ruta_absoluta_al_ejecutable():
    """/bin/rm esquivaría una allowlist que sólo mire el nombre."""
    result = RunCommand().execute({"argv": ["/bin/rm", "-rf", "/"]})
    assert not result.ok
    assert "ruta" in result.error


def test_run_command_corrige_al_modelo_si_manda_una_linea_de_shell():
    """El fallo real a detectar: toda la línea como UN string en vez de una lista."""
    result = RunCommand().execute({"argv": ["ls -la | grep foo"]})
    assert not result.ok
    assert "no usa shell" in result.error
    assert "['git', 'status']" in result.error  # la corrección es accionable


def test_run_command_permite_metacaracteres_en_argumentos_legitimos(tmp_path):
    """Con shell=False son literales inofensivos. Bloquearlos daría falsos positivos.

    `python3 -c 'x; y'`, `grep 'a|b'` y `find -name '*.py'` son todos usos válidos.
    """
    result = RunCommand(allowlist=frozenset({"python3"})).execute(
        {"argv": ["python3", "-c", "a = 1; print(a)"]}
    )
    assert result.ok
    assert result.output == "1"


def test_run_command_sin_shell_no_expande_globs(tmp_path):
    """La garantía estructural: '*' llega como literal, no lo expande nadie."""
    (tmp_path / "a.txt").touch()
    result = RunCommand(cwd=tmp_path).execute({"argv": ["ls", "*.txt"]})
    assert not result.ok  # ls busca un archivo llamado literalmente "*.txt"
    assert result.data["returncode"] != 0


def test_run_command_respeta_el_timeout():
    result = RunCommand(allowlist=frozenset({"python3"})).execute(
        {"argv": ["python3", "-c", "import time; time.sleep(5)"], "timeout_s": 0.3}
    )
    assert not result.ok
    assert "timeout" in result.error


def test_run_command_devuelve_exit_code_distinto_de_cero_como_dato():
    """Un comando que falla es información para el agente, no una excepción."""
    result = RunCommand().execute({"argv": ["ls", "/ruta/que/no/existe/jamas"]})
    assert not result.ok
    assert result.data["returncode"] != 0
    assert result.output  # el stderr llega al agente para que pueda reaccionar


def test_run_command_exige_argv_no_vacio():
    assert not RunCommand().execute({"argv": []}).ok


# ──────────────────────────── contrato de la base ─────────────────────────────


def test_una_skill_que_revienta_no_tumba_el_agente():
    class Explosiva(Skill):
        name = "explosiva"
        description = "revienta a propósito"

        class Params(BaseModel):
            pass

        def run(self, params):
            raise RuntimeError("boom")

    result = Explosiva().execute({})
    assert not result.ok
    assert "boom" in result.error


def test_tool_definition_tiene_la_forma_de_la_api():
    definicion = ReadFile.tool_definition()
    assert definicion["name"] == "read_file"
    assert definicion["input_schema"]["type"] == "object"
    assert "path" in definicion["input_schema"]["properties"]
    # La descripción debe ser prescriptiva: dice cuándo usarla.
    assert "Úsala" in definicion["description"]
    # El nombre literal se ancla en el texto: evita que el modelo lo traduzca.
    assert definicion["description"].startswith("[read_file]")


def test_skill_result_es_evaluable_como_booleano():
    assert bool(SkillResult.success("ok"))
    assert not bool(SkillResult.fail("mal"))


# ────────────────────────────────── glob ──────────────────────────────────────


def test_glob_cuenta_y_lista(tmp_path):
    (tmp_path / "src").mkdir()
    for n in ("a.py", "b.py"):
        (tmp_path / "src" / n).touch()
    (tmp_path / "src" / "c.txt").touch()

    result = Glob(root=tmp_path).execute({"pattern": "src/*.py"})

    assert result.ok
    assert result.data["total"] == 2
    assert "2 archivo(s)" in result.output
    assert sorted(result.data["paths"]) == ["src/a.py", "src/b.py"]


def test_glob_recursivo(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "x.py").touch()
    (tmp_path / "a" / "b" / "y.py").touch()

    assert Glob(root=tmp_path).execute({"pattern": "**/*.py"}).data["total"] == 2


def test_glob_sin_coincidencias_no_es_un_error(tmp_path):
    """Cero resultados es información válida, no un fallo."""
    result = Glob(root=tmp_path).execute({"pattern": "**/*.rs"})
    assert result.ok
    assert result.data["total"] == 0


def test_glob_no_escapa_de_la_raiz(tmp_path):
    """Un patrón con '..' no puede sacar la búsqueda del proyecto."""
    (tmp_path.parent / "secreto_fuera.py").touch()
    raiz = tmp_path / "proyecto"
    raiz.mkdir()
    (raiz / "dentro.py").touch()

    result = Glob(root=raiz).execute({"pattern": "../*.py"})

    assert result.data["total"] == 0
    assert "secreto_fuera" not in result.output


def test_glob_ignora_directorios(tmp_path):
    (tmp_path / "paquete.py").mkdir()  # un directorio que acaba en .py
    (tmp_path / "real.py").touch()

    assert Glob(root=tmp_path).execute({"pattern": "*.py"}).data["total"] == 1


def test_glob_trunca_la_lista_pero_no_el_conteo(tmp_path):
    """El conteo debe ser exacto aunque no se listen todas las rutas."""
    for i in range(30):
        (tmp_path / f"f{i:02}.py").touch()

    result = Glob(root=tmp_path).execute({"pattern": "*.py", "limit": 5})

    assert result.data["total"] == 30
    assert len(result.data["files"]) == 5
    assert "30 archivo(s)" in result.output
    assert "se listan 5" in result.output


def test_glob_devuelve_tamanos_ya_calculados(tmp_path):
    """El modelo no debe parsear tamaños de la salida cruda de `find` o `ls -l`.

    Se observó a qwen3:4b leer un número de inodo de `find -ls` como si fuera el
    tamaño (621708 bytes para un archivo de 1704) y descartar con eso el conteo
    correcto que glob le había dado.
    """
    (tmp_path / "chico.py").write_bytes(b"x" * 10)
    (tmp_path / "grande.py").write_bytes(b"x" * 5000)

    result = Glob(root=tmp_path).execute({"pattern": "*.py"})

    assert result.data["largest"] == {"path": "grande.py", "bytes": 5000}
    assert {f["path"]: f["bytes"] for f in result.data["files"]} == {
        "chico.py": 10,
        "grande.py": 5000,
    }
    assert "5000 bytes" in result.output


def test_glob_ordena_por_tamano_si_se_pide(tmp_path):
    for nombre, tam in (("a.py", 10), ("b.py", 900), ("c.py", 50)):
        (tmp_path / nombre).write_bytes(b"x" * tam)

    result = Glob(root=tmp_path).execute({"pattern": "*.py", "by_size": True})

    assert [f["path"] for f in result.data["files"]] == ["b.py", "c.py", "a.py"]


def test_glob_el_mayor_se_calcula_sobre_TODOS_no_sobre_los_listados(tmp_path):
    """Truncar la lista no debe falsear cuál es el mayor."""
    (tmp_path / "aaa_pequeno.py").write_bytes(b"x" * 10)
    (tmp_path / "zzz_enorme.py").write_bytes(b"x" * 9999)

    result = Glob(root=tmp_path).execute({"pattern": "*.py", "limit": 1})

    assert len(result.data["files"]) == 1
    assert result.data["files"][0]["path"] == "aaa_pequeno.py"  # orden alfabético
    assert result.data["largest"]["path"] == "zzz_enorme.py"  # pero el mayor es correcto
