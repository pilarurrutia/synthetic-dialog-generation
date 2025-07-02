# generation_gemma7b.py
import time
import json
import re
from dialog_generator import DialogGenerator
from dialog_evaluator import DialogEvaluator
from pathlib import Path
import pandas as pd

# === CONFIGURACIÓN ===
BASE_DIR = Path("/content/drive/MyDrive/TFG_LLM/DATA_GEMMA2")
REAL_DIR = Path("/content/drive/MyDrive/TFG_LLM/DATA_GEMMA2/real_dialogs")
SPEC_DIR = BASE_DIR / "specifications"
GEN_DIR = BASE_DIR / "generated_dialogs"
EVAL_DIR = BASE_DIR / "evaluated_dialogs"
PROMPTS_DIR = Path("/content/drive/MyDrive/TFG_LLM/PROMPTS")

BACKEND = "groq"
MODEL_NAME = "gemma2-9b-it"

# Crear carpetas si no existen
for folder in [SPEC_DIR, GEN_DIR, EVAL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Inicializar generador y evaluador
generator = DialogGenerator(model_name=MODEL_NAME, backend=BACKEND)
evaluator = DialogEvaluator(model_name=MODEL_NAME, backend=BACKEND)

# Cargar los real dialogs
real_dialogs = sorted(REAL_DIR.glob("dialog_*.txt"))[:15]

# === Generar Especificaciones y Diálogos ===
for dialog_path in real_dialogs:
    dialog_id = dialog_path.stem
    print(f"\n🔁 Processing: {dialog_id}")

    try:
        real_dialog = dialog_path.read_text()

        # 1. Generar especificación
        spec = generator.generate_specification(
            real_dialog,
            prompt_path=str(PROMPTS_DIR / "prompt_refined_generate_specification.txt")
        )
        (SPEC_DIR / f"{dialog_id}_spec.json").write_text(json.dumps(spec, indent=2))

        # 2. Generar diálogo sintético
        generated = generator.generate_dialog(
            spec,
            prompt_path=str(PROMPTS_DIR / "prompt_refined_generate_dialog.txt")
        )
        (GEN_DIR / f"{dialog_id}_gen.txt").write_text(generated)

        # 3. Evaluar automáticamente
        evaluation = evaluator.evaluate(
            generated_dialog=generated,
            reference_dialog=real_dialog,
            specification=spec,
            prompt_path=str(PROMPTS_DIR / "prompt_refined_evaluate_dialog.txt")
        )
        (EVAL_DIR / f"{dialog_id}_eval.json").write_text(json.dumps(evaluation, indent=2))

        # Espera para evitar rate limit
        time.sleep(4)

    except Exception as e:
        print(f"❌ Error in {dialog_id}: {e}")

# === Crear plantilla de evaluación humana ===
rows = []
for gen_file in sorted(GEN_DIR.glob("dialog_*_gen.txt")):
    dialog_id = gen_file.stem.replace("_gen", "")
    try:
        generated_dialog = gen_file.read_text().strip()
        real_path = REAL_DIR / f"{dialog_id}.txt"
        spec_path = SPEC_DIR / f"{dialog_id}_spec.json"
        eval_path = EVAL_DIR / f"{dialog_id}_eval.json"

        real_dialog = real_path.read_text().strip() if real_path.exists() else ""
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        auto_eval = json.loads(eval_path.read_text()) if eval_path.exists() else {}

        row = {
            "ID": dialog_id,
            "Generated Dialog": generated_dialog,
            "Real Dialog": real_dialog,
            "Specification": json.dumps(spec, indent=2),
            "Fluency (auto)": auto_eval.get("fluency", None),
            "Coherence (auto)": auto_eval.get("coherence", None),
            "Realism (auto)": auto_eval.get("realism", None),
            "Fidelity (auto)": auto_eval.get("fidelity_to_specification", None),
            "Engagement (auto)": auto_eval.get("engagement", None),
            "Originality (auto)": auto_eval.get("originality", None),
            "Comments (auto)": auto_eval.get("comments", "No comment provided."),
            "Fluency (human)": "",
            "Coherence (human)": "",
            "Realism (human)": "",
            "Fidelity (human)": "",
            "Engagement (human)": "",
            "Originality (human)": "",
            "Comments (human)": ""
        }
        rows.append(row)

    except Exception as e:
        print(f"❌ Error with dialog {dialog_id}: {e}")

# Guardar plantilla
df = pd.DataFrame(rows)
output_path = BASE_DIR / "evaluation_human_template_gemma2.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Template refined creada con {len(df)} diálogos: {output_path}")
