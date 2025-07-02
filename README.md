# Generación y Evaluación Automática de Conversaciones Sintéticas

Este repositorio contiene el sistema completo desarrollado para:

- Generar conversaciones sintéticas a partir de especificaciones estructuradas basadas en diálogos reales. Mediante los prompts se generan las especificaciones, los diálogos sintéticos y las evaluaciones.
- Evaluar de manera automática la calidad de los diálogos generados mediante métricas cuantitativas y clasificadores.

El sistema está dividido en dos fases:

**Fase 1: Generación de Diálogos**

- `dialog_generator.py`: Clase principal `DialogGenerator` que genera especificaciones y diálogos.
- `dialog_evaluator.py`: Clase `DialogEvaluator` que evalúa cualitativamente los diálogos.
- `main.py`: Script orquestador que lanza el proceso completo de generación y evaluación.

**Fase 2: Evaluación Cuantitativa**

- `evaluation_pipeline.py`: Script principal que ejecuta todas las métricas y exporta resultados.
- `evaluation_classes.py`: Módulo con clases de métricas y visualización.
- `__init__.py`: Archivo de inicialización del paquete.

**Prompts utilizados**

- `prompt_refined_generate_specification.txt`
- `prompt_refined_generate_dialog.txt`
- `prompt_refined_evaluate_dialog.txt`

---

## Instalación de dependencias

Primero, instala los requisitos con:

```bash
pip install -r requirements.txt
```

Asegúrate de tener Python >=3.8.

---

## Estructura de carpetas esperadas

El sistema espera las siguientes carpetas dentro de tu directorio de trabajo:

```
./real_dialogs/
./specifications/
./generated_dialogs/
./evaluated_dialogs/
./prompts/
./data/
```

Importante: crea estas carpetas antes de la ejecución y ajusta las rutas en los ficheros.

---

## Ejecución Fase 1 (Generación)

Edita `main.py` para configurar:

- Backend de inferencia (`Groq`, `local`, etc.)
- Datos reales
- Modelo deseado
- Rutas de entrada y salida

Lanza la generación:

```bash
python main.py
```

El sistema generará:

- Especificaciones en `/specifications`
- Diálogos sintéticos en `/generated_dialogs`
- Evaluaciones automáticas en `/evaluated_dialogs`
- Template Excel consolidado con todo

Espera que tú rellenes la parte (human) de la template para pasar a la siguiente fase.

---

## Ejecución Fase 2 (Evaluación cuantitativa)

Edita `evaluation_pipeline.py` si deseas cambiar rutas o archivos de entrada.

Para lanzar la evaluación completa:

```bash
python evaluation_pipeline.py
```

El sistema creará automáticamente:

- Carpeta `/metrics` con todos los cálculos en Excel.
- Carpeta `/figures` con visualizaciones (PNG, PDF, HTML).

---

## Configuración de claves API

Si usas Groq, crea un archivo `.env` o `keys.env` con:

```
GROQ_API_KEY=tu_clave_api
```

---

## Recomendaciones

- Para reproducir todo el flujo (generación + evaluación), usa Google Colab con montaje de Google Drive.
- Si empleas otros backends (OpenAI, Hugging Face), adapta los métodos `call_model` en las clases.

---

## Autor

Este sistema fue desarrollado en el contexto de un Trabajo Fin de Grado en Ingeniería y Sistemas de Datos por Pilar Urrutia Melgarejo.
2/07/2025

---

## Licencia

Uso académico y personal. Para otros usos, consultar previamente.

