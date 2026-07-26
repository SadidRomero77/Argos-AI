"""Clima vía Open-Meteo.

Se eligió Open-Meteo sobre OpenWeatherMap y compañía por una razón práctica:
**no necesita clave ni registro**. Una skill que exige darse de alta en un
servicio es una skill que se queda sin usar.

Devuelve el dato **ya interpretado**, no el JSON crudo. Es el principio 80/20:
convertir el código WMO 61 en "lluvia ligera" es una tabla, no una tarea para el
modelo — y se observó a qwen3:4b leer un número de inodo como si fuera un tamaño
en la salida de `find -ls`. Lo que puede calcular el código, lo calcula el código.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from argos.skills.base import Skill, SkillResult

GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"

# Códigos WMO. La traducción es determinista: no tiene sentido gastar tokens del
# modelo en adivinar qué significa un 61.
WMO = {
    0: "despejado",
    1: "mayormente despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "niebla",
    48: "niebla con escarcha",
    51: "llovizna ligera",
    53: "llovizna moderada",
    55: "llovizna intensa",
    56: "llovizna helada ligera",
    57: "llovizna helada intensa",
    61: "lluvia ligera",
    63: "lluvia moderada",
    65: "lluvia fuerte",
    66: "lluvia helada ligera",
    67: "lluvia helada fuerte",
    71: "nieve ligera",
    73: "nieve moderada",
    75: "nieve intensa",
    77: "granizo fino",
    80: "chubascos ligeros",
    81: "chubascos moderados",
    82: "chubascos violentos",
    85: "chubascos de nieve ligeros",
    86: "chubascos de nieve fuertes",
    95: "tormenta",
    96: "tormenta con granizo",
    99: "tormenta con granizo fuerte",
}


class WeatherParams(BaseModel):
    place: str = Field(description="Ciudad o lugar, ej. 'Bogotá' o 'Madrid, España'")
    days: int = Field(default=0, ge=0, le=6, description="Días de pronóstico además de hoy")


class Weather(Skill):
    name = "weather"
    description = (
        "[weather] Consulta el clima actual y el pronóstico de un lugar. Úsala "
        "siempre que pregunten por el tiempo, la temperatura o si va a llover. "
        "Devuelve datos reales y actuales — no los inventes ni digas que no puedes "
        "consultarlos."
    )
    Params = WeatherParams

    def __init__(self, client: httpx.Client | None = None) -> None:
        # Timeout corto: si el servicio no responde en unos segundos, es mejor
        # decirlo que dejar al usuario esperando.
        self._client = client or httpx.Client(timeout=httpx.Timeout(12.0, connect=4.0))

    def run(self, params: WeatherParams) -> SkillResult:
        geo = self._client.get(
            GEOCODING, params={"name": params.place, "count": 1, "language": "es"}
        )
        geo.raise_for_status()
        resultados = geo.json().get("results") or []
        if not resultados:
            return SkillResult.fail(
                f"no encuentro ningún lugar llamado '{params.place}'. Prueba con 'ciudad, país'."
            )

        sitio = resultados[0]
        consulta = {
            "latitude": sitio["latitude"],
            "longitude": sitio["longitude"],
            "current": ",".join(
                (
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "weather_code",
                    "wind_speed_10m",
                )
            ),
            "timezone": "auto",
        }
        if params.days:
            consulta["daily"] = "weather_code,temperature_2m_max,temperature_2m_min"
            consulta["forecast_days"] = params.days + 1

        respuesta = self._client.get(FORECAST, params=consulta)
        respuesta.raise_for_status()
        datos = respuesta.json()
        actual = datos.get("current") or {}

        nombre = ", ".join(
            filter(None, [sitio.get("name"), sitio.get("admin1"), sitio.get("country")])
        )
        cielo = WMO.get(actual.get("weather_code"), "condiciones desconocidas")
        temp = actual.get("temperature_2m")
        sensacion = actual.get("apparent_temperature")

        lineas = [
            f"{nombre}: {temp} °C ({cielo}).",
            f"Sensación térmica {sensacion} °C, humedad {actual.get('relative_humidity_2m')}%, "
            f"viento {actual.get('wind_speed_10m')} km/h.",
        ]

        diario = datos.get("daily") or {}
        if params.days and diario.get("time"):
            lineas.append("")
            # Se salta el índice 0: es hoy, y ya está descrito arriba.
            for i in range(1, min(params.days + 1, len(diario["time"]))):
                lineas.append(
                    f"{diario['time'][i]}: {WMO.get(diario['weather_code'][i], '—')}, "
                    f"mín {diario['temperature_2m_min'][i]} °C / "
                    f"máx {diario['temperature_2m_max'][i]} °C"
                )

        return SkillResult.success(
            "\n".join(lineas),
            place=nombre,
            temperature_c=temp,
            condition=cielo,
        )
