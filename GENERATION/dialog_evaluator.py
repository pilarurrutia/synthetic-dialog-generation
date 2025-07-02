import os
import re
import ast
import json
from pathlib import Path

class DialogEvaluator:
    def __init__(self, model_name="llama3-8b-8192", backend="groq", env_path=None):
        self.backend = backend
        self.model_name = model_name

        if self.backend == "groq":
            from openai import OpenAI
            key = os.getenv("GROQ_API_KEY")
            if env_path:
                with open(env_path) as f:
                    for line in f:
                        if "GROQ_API_KEY" in line:
                            key = line.strip().split("=")[1]
                            break
            if not key:
                raise ValueError("GROQ_API_KEY not found. Provide via .env or env_path.")
            self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

        elif self.backend == "local":
            from llama_cpp import Llama
            self.pipeline = Llama(
                model_path="/content/llama-2-7b-chat.Q4_K_M.gguf",
                n_ctx=2048,
                n_threads=4,
                chat_format="llama-2",
                n_gpu_layers=20
            )

    def _load_prompt(self, path):
        with open(path, "r") as f:
            return f.read()

    def extract_json(self, text):
        # Intentar encontrar el primer JSON válido
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            # Limpiar dobles llaves si aparecen por error del modelo
            json_str = json_str.replace("{{", "{").replace("}}", "}")
            try:
                return json.loads(json_str)
            except Exception as e:
                print("❌ Error parsing JSON:\n", json_str)
                raise e
        raise ValueError("No valid JSON object found in response.")

    def evaluate(self, generated_dialog, reference_dialog=None, specification=None, prompt_path="prompts/prompt_evaluate_dialog.txt"):
        prompt = self._load_prompt(prompt_path)
        prompt = prompt.replace("{GENERATED_DIALOG_HERE}", generated_dialog)
        prompt = prompt.replace("{REFERENCE_DIALOG_HERE}", reference_dialog if reference_dialog else "")
        prompt = prompt.replace("{SPECIFICATION_HERE}", str(specification) if specification else "")

        if self.backend == "groq":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            output = response.choices[0].message.content
        else:
            response = self.pipeline.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            output = response["choices"][0]["message"]["content"]

        print("🟡 Model output for evaluation:\n", output)
        parsed = self.extract_json(output)

        # Validar tipo de salida
        if not isinstance(parsed, dict):
            raise ValueError(f"❌ Output is not a dictionary: {parsed}")

        return parsed