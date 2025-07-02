import os
import re
import ast
import json
import subprocess

class DialogGenerator:
    def __init__(self, backend="groq", model_name="llama3-8b-8192", env_path=None, model_path=None, auto_model=False):
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
            if auto_model:
                self.model_path = self._download_model_if_needed()
            else:
                if not model_path:
                    raise ValueError("No model_path provided and auto_model is False.")
                self.model_path = model_path
            self._install_llama_cpp()
            from llama_cpp import Llama
            self.pipeline = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                chat_format="llama-2",
                n_gpu_layers=20
            )

    def _install_llama_cpp(self):
        try:
            import llama_cpp
        except ImportError:
            print("🔧 Installing llama-cpp-python...")
            subprocess.run(["pip", "install", "llama-cpp-python"], check=True)

    def _download_model_if_needed(self):
        model_filename = "llama-2-7b-chat.Q4_K_M.gguf"
        model_path = f"/content/{model_filename}"
        if not os.path.exists(model_path):
            print("⬇️ Downloading LLaMA 2 7B Chat model...")
            subprocess.run([
                "wget", "-O", model_path,
                "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
            ], check=True)
        else:
            print("✅ Model already exists.")
        return model_path

    def _load_prompt(self, path):
        with open(path, "r") as f:
            return f.read()

    def extract_json(self, text):

        # Step 1: Find the full JSON block from the first { to the last }
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or start > end:
            raise ValueError("No valid JSON block found in the text.")
    
        json_text = text[start:end+1]

        # Step 2: Replace curly/smart quotes with straight ones
        json_text = json_text.replace('“', '"').replace('”', '"').replace("’", "'")

        # Step 3: Fix malformed keys like "P2: "value" → "P2": "value"
        json_text = re.sub(r'("P\d)(:\s*")', r'\1"\2', json_text)

        # Step 4: Balance unmatched braces if any
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)
        
        # Step 5: Fix common closing bracket typos (e.g., double ]), fix misplaced closing braces and key-value issues
        json_text = re.sub(r'\]\s*\]', ']', json_text)
        json_text = re.sub(r'(\d)\},', r'\1,', json_text)
        # ✅ Only fix "P2's X" when it's used as a dictionary key
        json_text = re.sub(r'"(P\d)\'s ([^"]+)"(?=\s*:)', lambda m: f'"{m.group(1)}_{m.group(2)}"', json_text)
        
        # Step 6: Fix unterminated strings at end of file (e.g., in "comments")
        if json_text.count('"') % 2 != 0:
            json_text += '"'

        # Step 7: Try parsing
        try:
            return json.loads(json_text)
        except Exception as e:
            print("❌ Parsing failed after cleanup:\n", json_text)
            raise e

            
    def generate_specification(self, real_dialog, prompt_path="prompts/prompt_generate_specification.txt"):
        prompt = self._load_prompt(prompt_path)
        prompt_filled = prompt.replace("{REAL_DIALOG_HERE}", real_dialog)

        if self.backend == "groq":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_filled}],
                temperature=0.7
            )
            output = response.choices[0].message.content
        else:
            response = self.pipeline.create_chat_completion(
                messages=[{"role": "user", "content": prompt_filled}]
            )
            output = response["choices"][0]["message"]["content"]

        print("🟡 Model output for specification:\n", output)
        return self.extract_json(output)

    def generate_dialog(self, specification, prompt_path="prompts/prompt_generate_dialog.txt"):
        prompt = self._load_prompt(prompt_path)
        prompt_filled = prompt.replace("{SPECIFICATION_HERE}", str(specification))

        if self.backend == "groq":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_filled}],
                temperature=0.8
            )
            return response.choices[0].message.content
        else:
            response = self.pipeline.create_chat_completion(
                messages=[{"role": "user", "content": prompt_filled}]
            )
            return response["choices"][0]["message"]["content"]