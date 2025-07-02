import sys
sys.path.append("/content/drive/MyDrive/TFG_LLM/EVALUATION") 
from pathlib import Path
import pandas as pd
import plotly.express as px

from evaluation_classes import (
    DialogLexicalEvaluator,
    DialogSemanticEvaluator,
    DialogAutoMetrics,
    DialogAutoHumanEvaluator,
    DialogQualityClassifier,
    DialogVisualizationToolkit
)

# === Config ===
BASE_DIR = Path("/content/drive/MyDrive/TFG_LLM/DATA_GEMMA2")
FILE = BASE_DIR / "evaluation_human_template_gemma2.xlsx"
METRICS_DIR = BASE_DIR / "metrics"
FIGURES_DIR = BASE_DIR / "figures"

# Crear carpetas organizadas
for sub in ["auto", "lexical", "semantic", "human_auto", "classifier"]:
    (METRICS_DIR / sub).mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# === Load ===
df = pd.read_excel(FILE)
print(f"✅ Cargado {len(df)} diálogos")

# === Métricas automáticas BLEU, ROUGE, BERTScore ===
print("\n Métricas automáticas")
auto = DialogAutoMetrics(df)
auto.compute_all()
auto.save_results(METRICS_DIR / "auto" / "metrics_auto_gemma2.xlsx")
print("✅ Métricas automáticas guardadas.")

# === Métricas léxicas ===
print("\n Métricas léxicas")
lex = DialogLexicalEvaluator(df)
lex.compute_all()
lex.save_results(METRICS_DIR / "lexical" / "metrics_lexical_gemma2.xlsx")
print(f"✅ Guardadas métricas léxicas")

# === Métricas semánticas ===
sem = DialogSemanticEvaluator(
    df,
    output_metrics=METRICS_DIR / "semantic",
    output_figures=FIGURES_DIR / "semantic"
)
sem.compute_all()
sem.save_results(METRICS_DIR / "semantic" / "metrics_semantic_gemma2.xlsx")
print("✅ Métricas semánticas guardadas.")

# === Evaluaciones humanas vs automáticas ===
print("🔍 Comparación con evaluaciones humanas...")
hum = DialogAutoHumanEvaluator(
    df,
    output_metrics=METRICS_DIR / "human_auto",
    output_figures=FIGURES_DIR / "human_auto"
)
hum.compute_all()
print("✅ Comparación humano-auto exportada correctamente.")


# === Clasificador de aceptabilidad ===
print("🔍 Clasificador de aceptabilidad...")
clf = DialogQualityClassifier(
    output_metrics=METRICS_DIR / "classifier",
    output_figures=FIGURES_DIR / "classifier"
)
clf.load_data(df)
clf.compute_features()
clf.train()
clf.save_results()
df_clf = clf.df  # Para seguir con visualizaciones
print("✅ Clasificador entrenado y resultados exportados.")


# === Visualización final ===
print("\n📊 Visualización completa")
viz = DialogVisualizationToolkit(
    df_clf,
    output_dir=FIGURES_DIR / "visualizations",
    export_pdf=True,
    export_html=True
)
viz.plot_interactive_histograms()
viz.plot_wordclouds()
viz.plot_boxplots_by_prediction()
viz.plot_human_score_distributions()
print(f"✅ Figuras guardadas en: {FIGURES_DIR / 'visualizations'}")

