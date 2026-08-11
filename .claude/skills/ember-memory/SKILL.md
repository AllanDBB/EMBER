---
name: ember-memory
description: Use when adding, modifying, or debugging a memory architecture in EMBER — SDM, ENN, spiking, or a new substrate — or when touching ember.core policies, TraceStore, or the Memory protocol. Covers the contract, the mandatory tests, and the four failure modes that already broke published results once.
---

# Agregar o tocar una arquitectura de memoria

## El contrato

Toda memoria satisface `ember.core.protocol.Memory`:

```python
mem = Arquitectura(dim=32, capacity=20, seed=0)
mem.write(key, value, pred_error=0.5)  # key se normaliza acá adentro
resultado = mem.read(query)  # ReadResult(value, similarity, index)
len(mem)  # trazas guardadas ahora
mem.capacity  # cuántas pueden coexistir
```

Sobre memoria vacía, `read` devuelve `EMPTY_READ`, nunca una excepción.

## Cómo agregarla

1. **Empezá desde `TraceStore`**, no desde arreglos propios. Es donde vive toda
   la contabilidad de trazas, y por eso los bugs de contabilidad no se pueden
   repetir en dos lugares.

2. **Reutilizá las políticas de `ember.core.policies`** para el ciclo de vida de
   traza: fuerza inicial, decaimiento, desalojo. Si tu sustrato necesita algo
   distinto para leer o escribir, eso sí es propio — pero el ciclo de vida no.

3. **Registrala en `ARCHITECTURES`** (`src/ember/memories/__init__.py`). Con eso
   `tests/memories/test_contrato.py` la somete automáticamente a siete
   invariantes, y entra al benchmark de dos fases.

4. **Si es lenta**, agregala a `LENTAS` en `exp04_arch_benchmark.py` y marcá sus
   pruebas con `@pytest.mark.slow`.

## Los cuatro modos de falla que ya rompieron resultados

No son hipotéticos. Los cuatro estaban en el código piloto y cada uno invalidó
números que llegaron a un borrador.

### 1. Reimplementar un mecanismo que ya existe

El piloto tenía dos motores: uno para el NAS y otro para el benchmark. El
desalojo FIFO era `argmax(age)` en uno y `pop(0)` en el otro; la fuerza inicial
incluía novedad en uno y no en el otro; el umbral de acierto era 0.90 y 0.75.
Las tablas resultantes no medían lo mismo aunque el paper las presentara como
comparables.

**Si estás escribiendo una segunda versión de un desalojo o de una compuerta,
parás.** Usá la que existe o cambiá la que existe.

### 2. No revertir lo que escribiste en un sustrato distribuido

```python
# al escribir
self.V[activas] += strength * k
# al desalojar  ← lo que hacía el piloto
self.V[activas] -= k  # MAL: resta 1×, sumó strength×
```

Cada desalojo dejaba un residuo proporcional a `strength - 1`. Sobre cientos de
escrituras en capacidad 20, los contadores terminaban dominados por basura de
trazas ya borradas, y todas las lecturas se hacían contra eso.

**Guardá el escalar exacto en `store.contribution` y restalo.** Si tu conjunto
activo es estocástico (Spiking-SDM), guardalo también por traza: no se puede
recalcular idéntico.

Test obligatorio:

```python
def test_los_contadores_vuelven_a_cero_al_desalojar_todo():
    mem = TuMemoria(dim=32, capacity=3, seed=0)
    for i, k in enumerate(_claves(40)):
        mem.write(k, i, pred_error=0.9 if i % 2 else 0.1)
    while len(mem):
        mem._desalojar(0)
    assert float(np.abs(mem.V).max()) == pytest.approx(0.0, abs=1e-3)
```

### 3. Sembrar con algo que no es una semilla

```python
rng = np.random.default_rng(id(neuron_ids) % 2**32)  # la dirección de memoria
```

Los resultados de esa arquitectura no eran replicables entre corridas ni entre
plataformas. **Toda aleatoriedad sale de `self.rng`**, derivado del `seed` que
pasa el experimento.

Test obligatorio: dos instancias con la misma semilla dan resultados idénticos
bit a bit, y con semillas distintas dan distintos.

### 4. Un parámetro cuyo valor lo vuelve inoperante en tu dominio

Sutil y el más difícil de ver. `Radius` tenía umbral absoluto 0.70; en R³² dos
vectores gaussianos tienen coseno ~0.18, así que el círculo de activación
contenía siempre exactamente una traza y el modo degeneraba en vecino más
cercano. El mecanismo estaba implementado y era correcto — el umbral lo mataba.

**Todo umbral de similitud debería ser relativo a la escala del dominio**, no
absoluto. Si ponés uno absoluto, verificá con `exp03_axis_liveness` que el eje
sigue vivo.

## Antes de dar por terminado

```bash
uv run pytest tests/memories/ -q
uv run python -m experiments.exp03_axis_liveness   # ningún eje puede dar 0.000
uv run python -m experiments.exp04_arch_benchmark  # entra al benchmark
```

Si `exp04` cambia números que ya están en `paper/`, anotá qué se movió y por qué
en `docs/notas/`, como se hizo con `2026-08-11-numeros-movidos.md`. Un número
del paper que cambia sin registro es peor que un número equivocado.

## Contexto del dominio

Para qué son SDM, engrama, STDP, la ley del umbral: skill `ember-domain`.
