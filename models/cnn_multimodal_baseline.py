import numpy as np
import time
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from llm.groq_client import GroqAnalyst

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class CNNMultimodalTrainer:
    def __init__(self, epochs=20, lr=0.001, debug=False):
        self.llm = GroqAnalyst()
        self.debug = debug

        # Control de tasa
        self.base_delay = 0.6   # segundos entre requests
        self.max_retries = 5

    # ==========================================
    # CONVERSIÓN
    # ==========================================

    def _tensor_to_points(self, img_tensor):
        points = []
        h, w = img_tensor.shape[-2:]

        for i in range(h):
            for j in range(w):
                val = float(np.mean(img_tensor[:, i, j]))
                if val > 0:
                    points.append({"lat": float(i), "lon": float(j)})

        return points[:20]

    def _tensor_to_gbif(self, tab_tensor):
        gbif = []
        for i, val in enumerate(tab_tensor):
            if val > 0:
                gbif.append({
                    "species": f"feature_{i}",
                    "count": int(val * 10)
                })

        return gbif[:10]

    def _build_prompt(self, question, points, gbif):

        map_str = "\n".join([f"- cell ({p['lat']}, {p['lon']})" for p in points])
        gbif_str = "\n".join([f"- {g['species']}: {g['count']}" for g in gbif])

        return f"""
You are an ecological niche model.

Environmental conditions:
{gbif_str}

Spatial distribution:
{map_str}

Question:
{question}

Rules:
- Answer ONLY "YES" or "NO"

Answer:
"""

    # =============
    # LLM CALL 
    # =============

    def _generate_with_retry(self, prompt):
        delay = self.base_delay

        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate_simple(prompt)

                # pequeño delay 
                time.sleep(self.base_delay)

                return response

            except Exception as e:
                msg = str(e).lower()

                # detectar rate limit
                if "429" in msg or "rate" in msg:
                    if self.debug:
                        print(f"[RATE LIMIT] retry {attempt}")

                    time.sleep(delay)
                    delay *= 2  # backoff exponencial
                else:
                    if self.debug:
                        print(f"[ERROR] {e}")
                    return "INVALID"

        return "INVALID"

    def _normalize(self, text):
        text = text.lower()
        if "yes" in text:
            return 1
        elif "no" in text:
            return 0
        return -1

    # ==========================================
    # API
    # ==========================================

    def train(self, X_img, X_tab, y):
        print("[LLM BASELINE] No training required.")

    def evaluate(self, X_img, X_tab, y):

        print("[INFO] Ejecutando LLM baseline...")

        predictions = []

        #  Reducir carga
        max_samples = 30  

        for i in range(min(len(X_img), max_samples)):

            points = self._tensor_to_points(X_img[i])
            gbif = self._tensor_to_gbif(X_tab[i])

            prompt = self._build_prompt(
                "Is this environment suitable for the species?",
                points,
                gbif
            )

            response = self._generate_with_retry(prompt)
            pred = self._normalize(response)

            predictions.append(pred)

        # ==========================================
        # MÉTRICAS
        # ==========================================

        valid = [(p, label) for p, label in zip(predictions, y) if p != -1]

        if not valid:
            print("[ERROR] No valid predictions.")
            return {}

        preds_clean, labels_clean = zip(*valid)

        acc = accuracy_score(labels_clean, preds_clean)
        f1 = f1_score(labels_clean, preds_clean, zero_division=0)
        precision = precision_score(labels_clean, preds_clean, zero_division=0)
        recall = recall_score(labels_clean, preds_clean, zero_division=0)

        unique, counts = np.unique(preds_clean, return_counts=True)
        dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))

        print("\n========================================")
        print("[LLM BASELINE RESULTADOS]")
        print(f"Samples evaluados: {len(preds_clean)}")
        print(f"Distribución preds: {dist}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print("========================================")

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "distribution": dist
        }