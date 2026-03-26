# hex_ai_player

Jugador de Hex para proyecto académico de Inteligencia Artificial.

## Descripción

Este repositorio contiene un agente para Hex implementado en Python, diseñado para operar bajo límite estricto de tiempo por jugada y compatible con tamaño de tablero variable.

El enfoque principal combina:

- Monte Carlo Tree Search (MCTS) como motor de exploración global.
- Capa táctica determinista para decisiones críticas de corto horizonte.
- Heurísticas específicas de Hex (conectividad, patrones tácticos y two-distance).

## Estructura del repositorio

- `Ronald_Cabrera_Martínez/solution.py`: implementación del jugador.
- `Ronald_Cabrera_Martínez/estrategia.pdf`: documento técnico de estrategia.


## Características técnicas del agente

- Búsqueda MCTS con UCT.
- Control de tiempo por jugada con margen de seguridad.
- Reutilización de información en estados repetidos (transposiciones).
- Política de simulación mixta para balancear calidad y throughput.
- Evaluación estructural con heurística two-distance.
- Detección de patrones tácticos locales en Hex.

## Compatibilidad

El jugador es agnóstico al tamaño de tablero y está pensado para funcionar con cualquier `N > 2`.


## Uso esperado

Este repositorio se integra en el entorno de evaluación del curso/proyecto. El punto de entrada del agente está en `solution.py` y expone la clase de jugador requerida por el framework de Hex.

## Notas

- No se requiere conectividad de red para ejecutar el agente.
- No se realiza lectura/escritura de archivos durante la jugada.
- La documentación técnica completa está en `estrategia.pdf`.