from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from collections import Counter
import mauve
import pandas as pd
import numpy as np
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

#nltk.download('punkt')

# Incluye aquí todas las clases: 
# - DialogAutoMetrics
import evaluate

import evaluate

class DialogAutoMetrics:
    def __init__(self, df, dialog_column="Generated Dialog", reference_column="Real Dialog"):
        self.df = df.copy()
        self.dialog_column = dialog_column
        self.reference_column = reference_column

    def compute_bleu(self):
        bleu_metric = evaluate.load("bleu")
        scores = []
        for pred, ref in zip(self.df[self.dialog_column], self.df[self.reference_column]):
            result = bleu_metric.compute(predictions=[str(pred).strip()], references=[[str(ref).strip()]])
            scores.append(result["bleu"])
        self.df["BLEU"] = scores

    def compute_rouge(self):
        rouge_metric = evaluate.load("rouge")
        r1, r2, rl = [], [], []
        for pred, ref in zip(self.df[self.dialog_column], self.df[self.reference_column]):
            result = rouge_metric.compute(predictions=[pred], references=[ref])
            r1.append(result["rouge1"])
            r2.append(result["rouge2"])
            rl.append(result["rougeL"])
        self.df["ROUGE-1"] = r1
        self.df["ROUGE-2"] = r2
        self.df["ROUGE-L"] = rl

    def compute_bertscore(self):
        from bert_score import score
        P, R, F1 = score(
            self.df[self.dialog_column].astype(str).tolist(),
            self.df[self.reference_column].astype(str).tolist(),
            lang="en", verbose=False
        )
        self.df["BERTScore_P"] = P.numpy()
        self.df["BERTScore_R"] = R.numpy()
        self.df["BERTScore_F1"] = F1.numpy()

    def compute_all(self):
        self.compute_bleu()
        self.compute_rouge()
        self.compute_bertscore()

    def save_results(self, path):
        cols_to_keep = [
            self.dialog_column, self.reference_column,
            "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L",
            "BERTScore_P", "BERTScore_R", "BERTScore_F1"
        ]
        self.df[cols_to_keep].to_excel(path, index=False)


class DialogLexicalEvaluator:
    def __init__(self, df, text_col="Generated Dialog"):
        self.df = df.copy()
        self.text_col = text_col
        self.results = {}

    def _tokenize(self, text):
        return text.lower().split()

    def _distinct(self, tokens, n=1):
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    def _repetitiveness(self, tokens):
        counter = Counter(tokens)
        total = sum(counter.values())
        most_common = counter.most_common(1)[0][1] if counter else 0
        return most_common / total if total > 0 else 0.0

    def compute_all(self):
        d1_list, d2_list, rep_list = [], [], []
        for text in self.df[self.text_col].dropna():
            tokens = self._tokenize(text)
            d1 = self._distinct(tokens, n=1)
            d2 = self._distinct(tokens, n=2)
            rep = self._repetitiveness(tokens)

            d1_list.append(d1)
            d2_list.append(d2)
            rep_list.append(rep)

        self.df["Distinct-1"] = d1_list
        self.df["Distinct-2"] = d2_list
        self.df["Repetitiveness"] = rep_list

        self.results = {
            "Distinct-1": round(sum(d1_list) / len(d1_list), 4),
            "Distinct-2": round(sum(d2_list) / len(d2_list), 4),
            "Repetitiveness": round(sum(rep_list) / len(rep_list), 4),
        }

    def summary(self):
        return pd.Series(self.results)

    def plot_distributions(self):
        plt.figure(figsize=(15, 4))
        for i, col in enumerate(["Distinct-1", "Distinct-2", "Repetitiveness"]):
            plt.subplot(1, 3, i + 1)
            sns.histplot(self.df[col], kde=True, bins=20, color="steelblue")
            plt.title(col)
            plt.xlabel("Score")
            plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def save_results(self, path):
        self.df[["Distinct-1", "Distinct-2", "Repetitiveness"]].to_excel(path, index=False)

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import mauve

class DialogSemanticEvaluator:
    def __init__(self, df, generated_col="Generated Dialog", reference_col="Real Dialog", output_metrics=None, output_figures=None):
        self.df = df.copy()
        self.generated_col = generated_col
        self.reference_col = reference_col
        self.output_metrics = Path(output_metrics) if output_metrics else None
        self.output_figures = Path(output_figures) if output_figures else None
        if self.output_metrics:
            self.output_metrics.mkdir(parents=True, exist_ok=True)
        if self.output_figures:
            self.output_figures.mkdir(parents=True, exist_ok=True)

    def compute_meteor(self):
        scores = []
        for pred, ref in zip(self.df[self.generated_col], self.df[self.reference_col]):
            try:
                score_val = meteor_score([word_tokenize(str(ref))], word_tokenize(str(pred)))
            except:
                score_val = 0.0
            scores.append(score_val)
        self.df["METEOR"] = scores

    def compute_mauve(self):
        p_text = self.df[self.generated_col].astype(str).tolist()
        q_text = self.df[self.reference_col].astype(str).tolist()
        mauve_result = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, verbose=False)
        self.mauve_score = mauve_result.mauve  

        if self.output_metrics:
            pd.DataFrame({"MAUVE Score": [mauve_result.mauve]}).to_excel(self.output_metrics / "mauve_score.xlsx", index=False)

    def plot_length_distributions(self):
        self.df["Generated Length"] = self.df[self.generated_col].apply(lambda x: len(str(x).split()))
        self.df["Real Length"] = self.df[self.reference_col].apply(lambda x: len(str(x).split()))

        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["Generated Length"], color="blue", label="Generated", kde=True)
        sns.histplot(self.df["Real Length"], color="green", label="Real", kde=True)
        plt.title("Distribución de longitud (nº de palabras)")
        plt.xlabel("Número de palabras")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        if self.output_figures:
            plt.savefig(self.output_figures / "length_distribution.png", bbox_inches='tight')
            plt.savefig(self.output_figures / "length_distribution.pdf", bbox_inches='tight')
        plt.close()

    def compute_all(self):
        self.compute_meteor()
        self.compute_mauve()
        self.plot_length_distributions()

    def save_results(self, path):
        self.df[["METEOR"]].to_excel(path, index=False)
        if self.output_metrics:
            pd.DataFrame({"MAUVE Score": [self.mauve_score]}).to_excel(
                self.output_metrics / "mauve_score.xlsx", index=False
        )


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
#HUMAN CORR

class DialogAutoHumanEvaluator:
    def __init__(self, df, verbose=False, output_metrics=None, output_figures=None):
        self.df = df.copy()
        self.verbose = verbose
        self.output_metrics = Path(output_metrics) if output_metrics else None
        self.output_figures = Path(output_figures) if output_figures else None
        if self.output_metrics:
            self.output_metrics.mkdir(parents=True, exist_ok=True)
        if self.output_figures:
            self.output_figures.mkdir(parents=True, exist_ok=True)

        self.criteria = ["Fluency", "Coherence", "Realism", "Fidelity", "Engagement", "Originality"]
        self.auto_cols = [f"{c} (auto)" for c in self.criteria]
        self.human_cols = [f"{c} (human)" for c in self.criteria]
        for col in self.auto_cols + self.human_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df_clean = self.df.dropna(subset=self.human_cols)

    def compute_all(self):
        self._compute_mae_and_bias()
        self._compute_correlations()
        self._compute_mae_by_bin()
        self.save_results()

    def _compute_mae_and_bias(self):
        self.mae_dict, self.bias_dict = {}, {}
        for c in self.criteria:
            a, h = f"{c} (auto)", f"{c} (human)"
            mae = mean_absolute_error(self.df_clean[h], self.df_clean[a])
            bias = self.df_clean[a].mean() - self.df_clean[h].mean()
            self.mae_dict[c] = mae
            self.bias_dict[c] = bias

        df_stats = pd.DataFrame({
            "MAE": self.mae_dict,
            "Bias (Auto - Human)": self.bias_dict
        }).T
        if self.output_metrics:
            df_stats.to_excel(self.output_metrics / "mae_bias_summary.xlsx")
        self.df_stats = df_stats

    def _compute_correlations(self):
        pearson_vals, spearman_vals = {}, {}
        corr_matrix = pd.DataFrame(index=self.criteria, columns=["Pearson", "Spearman"])

        for c in self.criteria:
            a, h = f"{c} (auto)", f"{c} (human)"
            if self.df_clean[a].std() > 0 and self.df_clean[h].std() > 0:
                p_corr, _ = pearsonr(self.df_clean[a], self.df_clean[h])
                s_corr, _ = spearmanr(self.df_clean[a], self.df_clean[h])
                pearson_vals[c] = p_corr
                spearman_vals[c] = s_corr
                corr_matrix.loc[c] = [p_corr, s_corr]
            else:
                corr_matrix.loc[c] = [None, None]

        if self.output_metrics:
            corr_matrix.to_excel(self.output_metrics / "correlations_summary.xlsx")

        if self.output_figures:
            plt.figure(figsize=(6, 4))
            sns.heatmap(corr_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlaciones Auto vs Humano")
            plt.tight_layout()
            plt.savefig(self.output_figures / "correlations_heatmap.png", bbox_inches='tight')
            plt.savefig(self.output_figures / "correlations_heatmap.pdf", bbox_inches='tight')
            plt.close()

    def _compute_mae_by_bin(self):
        bin_labels = ['Low (1–2)', 'Mid (3)', 'High (4–5)']
        bin_ranges = [0, 2.5, 3.5, 5.1]
        mae_bin_summary = {}

        for c in self.criteria:
            a, h = f"{c} (auto)", f"{c} (human)"
            df_binned = self.df_clean[[a, h]].copy()
            df_binned["bin"] = pd.cut(df_binned[h], bins=bin_ranges, labels=bin_labels)
            mae_by_bin = {}
            for label in bin_labels:
                bin_group = df_binned[df_binned["bin"] == label]
                if not bin_group.empty:
                    mae_by_bin[label] = mean_absolute_error(bin_group[h], bin_group[a])
                else:
                    mae_by_bin[label] = None
            mae_bin_summary[c] = mae_by_bin

        df_bins = pd.DataFrame(mae_bin_summary).T[bin_labels]
        if self.output_metrics:
            df_bins.to_excel(self.output_metrics / "mae_by_bins.xlsx")

        if self.output_figures:
            plt.figure(figsize=(8, 4))
            sns.heatmap(df_bins.astype(float), annot=True, cmap="Blues", fmt=".2f")
            plt.title("MAE por tramos de puntuación humana")
            plt.tight_layout()
            plt.savefig(self.output_figures / "mae_by_bins_heatmap.png", bbox_inches='tight')
            plt.savefig(self.output_figures / "mae_by_bins_heatmap.pdf", bbox_inches='tight')
            plt.close()

    def save_results(self):
        if self.output_metrics:
            self.df.to_excel(self.output_metrics / "metrics_human_auto_full.xlsx", index=False)

# - DialogQualityClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

class DialogQualityClassifier:
    def __init__(self, max_depth=3, output_metrics=None, output_figures=None):
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.human_cols = ['Fluency (human)', 'Coherence (human)', 'Realism (human)',
                           'Fidelity (human)', 'Engagement (human)', 'Originality (human)']
        self.output_metrics = Path(output_metrics) if output_metrics else None
        self.output_figures = Path(output_figures) if output_figures else None
        if self.output_metrics:
            self.output_metrics.mkdir(parents=True, exist_ok=True)
        if self.output_figures:
            self.output_figures.mkdir(parents=True, exist_ok=True)

    def load_data(self, df):
        self.df = df.copy()
        self.df = self.df.dropna(subset=["Generated Dialog"] + self.human_cols)

    def compute_features(self):
        def extract_features(dialog):
            lines = str(dialog).split("\n")
            words = " ".join(lines).split()
            return pd.Series([
                len(lines),
                len(words),
                len(set(words)) / (len(words) + 1e-6)
            ])
        self.df[["n_turns", "n_words", "repetition_ratio"]] = self.df["Generated Dialog"].apply(extract_features)
        self.X = self.df[["n_turns", "n_words", "repetition_ratio"]]
        self.df["acceptable"] = self.df[self.human_cols].mean(axis=1) >= 4
        self.y = self.df["acceptable"]

    def train(self):
        self.model.fit(self.X, self.y)
        self.df["prediction"] = self.model.predict(self.X)

    def save_results(self):
        # Export full DataFrame with predictions
        if self.output_metrics:
            self.df.to_excel(self.output_metrics / "metrics_classifier_full.xlsx", index=False)

        # Export decision tree
        if self.output_figures:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(self.model, feature_names=self.X.columns.tolist(), class_names=["No", "Yes"],
                      filled=True, rounded=True, fontsize=9, ax=ax)
            plt.tight_layout()
            plt.savefig(self.output_figures / "decision_tree.png", bbox_inches='tight')
            plt.savefig(self.output_figures / "decision_tree.pdf", bbox_inches='tight')
            plt.close()

            # Export text version of tree
            tree_text = export_text(self.model, feature_names=list(self.X.columns))
            with open(self.output_figures / "decision_tree.txt", "w") as f:
                f.write(tree_text)

            # Export feature importance
            importance = pd.Series(self.model.feature_importances_, index=self.X.columns)
            plt.figure(figsize=(6, 4))
            sns.barplot(x=importance.values, y=importance.index, palette="viridis")
            plt.title("Importancia de variables")
            plt.tight_layout()
            plt.savefig(self.output_figures / "feature_importance.png", bbox_inches='tight')
            plt.savefig(self.output_figures / "feature_importance.pdf", bbox_inches='tight')
            plt.close()

import plotly.express as px
#VISUALIZATION
class DialogVisualizationToolkit:
    def __init__(self, df, output_dir=None, export_pdf=True, export_html=False):
        self.df = df
        self.output_dir = Path(output_dir) if output_dir else None
        self.export_pdf = export_pdf
        self.export_html = export_html
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, fig, filename_base):
        if self.output_dir:
            fig_path_png = self.output_dir / f"{filename_base}.png"
            fig.write_image(fig_path_png)

            if self.export_html:
                fig_path_html = self.output_dir / f"{filename_base}.html"
                fig.write_html(fig_path_html)

            if self.export_pdf:
                fig_path_pdf = self.output_dir / f"{filename_base}.pdf"
                fig.write_image(fig_path_pdf, format="pdf")

    def plot_interactive_histograms(self):
        cols = [c for c in self.df.columns if any(c.startswith(prefix) for prefix in ["BLEU", "ROUGE", "BERT", "Distinct", "Repetitiveness"])]
        for col in cols:
            fig = px.histogram(self.df, x=col, nbins=30, title=f"Distribución de {col}", marginal="box")
            self._save_fig(fig, f"histogram_{col}")

    def plot_boxplots_by_prediction(self, prediction_col="prediction", human_cols=None):
        if prediction_col not in self.df.columns:
            print(f"⚠️ Columna '{prediction_col}' no encontrada. No se puede generar boxplots.")
            return

        if human_cols is None:
            human_cols = [c for c in self.df.columns if "(human)" in c]

        for col in human_cols:
            fig = px.box(self.df, x=prediction_col, y=col, points="all", title=f"{col} según predicción del clasificador")
            self._save_fig(fig, f"boxplot_{col.replace(' ', '_')}")

    def plot_human_score_distributions(self):
        human_cols = [c for c in self.df.columns if c.endswith("(human)")]
        for col in human_cols:
            fig = px.histogram(self.df, x=col, nbins=20, title=f"Distribución: {col}", marginal="rug")
            self._save_fig(fig, f"distribution_{col.replace(' ', '_')}")

    def plot_wordclouds(self, text_col='Generated Dialog', label_col='acceptable'):
        if label_col not in self.df.columns:
            print(f"Columna '{label_col}' no encontrada")
            return

        def clean_text(text):
            tokens = text.split()
            return " ".join([t for t in tokens if not t.lower().startswith("p1:") and not t.lower().startswith("p2:")])

        text_true = " ".join(self.df[self.df[label_col] == True][text_col].dropna().apply(clean_text))
        text_false = " ".join(self.df[self.df[label_col] == False][text_col].dropna().apply(clean_text))

        wc_true = WordCloud(width=600, height=400, background_color='white').generate(text_true)
        wc_false = WordCloud(width=600, height=400, background_color='white').generate(text_false)

        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        axs[0].imshow(wc_true, interpolation='bilinear')
        axs[0].axis("off")
        axs[0].set_title("Wordcloud - Aceptables")

        axs[1].imshow(wc_false, interpolation='bilinear')
        axs[1].axis("off")
        axs[1].set_title("Wordcloud - No aceptables")

        plt.tight_layout()

        if self.output_dir:
            path_base = self.output_dir / "wordclouds"
            plt.savefig(f"{path_base}.png", bbox_inches='tight')
            if self.export_pdf:
                plt.savefig(f"{path_base}.pdf", bbox_inches='tight')

        plt.show()