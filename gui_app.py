import os
import sys
import webbrowser
import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox


class IrisMLOpsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Classification - MLOps System")
        self.root.geometry("640x720")
        self.root.minsize(580, 680)
        self.root.configure(bg="#f4f6f9")

        # Load model
        self.model_path = os.path.join(os.path.dirname(__file__), "models", "model.joblib")
        self.model_artifact = self.load_model()

        self.setup_ui()

    def load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Model Not Found",
                f"Trained model not found at:\n{self.model_path}\nPlease run DVC pipeline first.",
            )
            return None
        return joblib.load(self.model_path)

    def setup_ui(self):
        # Header banner
        header_frame = tk.Frame(self.root, bg="#1e3a8a", padx=20, pady=16)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="🌸 Iris Flower Classification",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg="#1e3a8a",
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_frame,
            text="MLOps Production Inference System | Random Forest Classifier",
            font=("Segoe UI", 10),
            fg="#bfdbfe",
            bg="#1e3a8a",
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Main content card
        main_frame = tk.Frame(self.root, bg="white", padx=24, pady=18, bd=1, relief=tk.SOLID)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Preset test buttons
        presets_frame = tk.LabelFrame(
            main_frame,
            text="Quick Load Sample Measurements",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            padx=12,
            pady=8,
        )
        presets_frame.pack(fill=tk.X, pady=(0, 15))

        btn_setosa = tk.Button(
            presets_frame,
            text="🌸 Setosa",
            bg="#ecfdf5",
            fg="#065f46",
            font=("Segoe UI", 9, "bold"),
            relief=tk.GROOVE,
            command=lambda: self.set_inputs(5.1, 3.5, 1.4, 0.2),
        )
        btn_setosa.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        btn_versicolor = tk.Button(
            presets_frame,
            text="🌿 Versicolor",
            bg="#eff6ff",
            fg="#1e40af",
            font=("Segoe UI", 9, "bold"),
            relief=tk.GROOVE,
            command=lambda: self.set_inputs(6.0, 2.9, 4.5, 1.5),
        )
        btn_versicolor.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        btn_virginica = tk.Button(
            presets_frame,
            text="🌺 Virginica",
            bg="#faf5ff",
            fg="#6b21a8",
            font=("Segoe UI", 9, "bold"),
            relief=tk.GROOVE,
            command=lambda: self.set_inputs(6.9, 3.1, 5.4, 2.1),
        )
        btn_virginica.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        # Sliders & entry inputs
        inputs_frame = tk.LabelFrame(
            main_frame,
            text="Flower Measurements (centimeters)",
            font=("Segoe UI", 10, "bold"),
            bg="white",
            padx=14,
            pady=10,
        )
        inputs_frame.pack(fill=tk.X, pady=(0, 15))

        self.val_sl = tk.DoubleVar(value=5.1)
        self.val_sw = tk.DoubleVar(value=3.5)
        self.val_pl = tk.DoubleVar(value=1.4)
        self.val_pw = tk.DoubleVar(value=0.2)

        self.create_slider(inputs_frame, "Sepal Length (cm):", self.val_sl, 4.0, 8.0, 0)
        self.create_slider(inputs_frame, "Sepal Width (cm):", self.val_sw, 2.0, 4.5, 1)
        self.create_slider(inputs_frame, "Petal Length (cm):", self.val_pl, 1.0, 7.0, 2)
        self.create_slider(inputs_frame, "Petal Width (cm):", self.val_pw, 0.1, 2.5, 3)

        # Predict Button
        btn_predict = tk.Button(
            main_frame,
            text="🔍 Classify Iris Species",
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            pady=8,
            cursor="hand2",
            relief=tk.FLAT,
            command=self.predict,
        )
        btn_predict.pack(fill=tk.X, pady=(0, 15))

        # Result display card
        self.result_frame = tk.Frame(main_frame, bg="#f8fafc", bd=1, relief=tk.RIDGE, padx=16, pady=12)
        self.result_frame.pack(fill=tk.X)

        self.lbl_result_title = tk.Label(
            self.result_frame,
            text="Predicted Species:",
            font=("Segoe UI", 11),
            fg="#64748b",
            bg="#f8fafc",
        )
        self.lbl_result_title.pack(anchor="w")

        self.lbl_result_species = tk.Label(
            self.result_frame,
            text="-- Click Classify --",
            font=("Segoe UI", 20, "bold"),
            fg="#0f172a",
            bg="#f8fafc",
        )
        self.lbl_result_species.pack(anchor="w", pady=(2, 6))

        self.lbl_confidence = tk.Label(
            self.result_frame,
            text="Confidence: --",
            font=("Segoe UI", 10),
            fg="#475569",
            bg="#f8fafc",
        )
        self.lbl_confidence.pack(anchor="w")

        self.lbl_probs = tk.Label(
            self.result_frame,
            text="",
            font=("Consolas", 9),
            fg="#334155",
            bg="#f8fafc",
        )
        self.lbl_probs.pack(anchor="w", pady=(4, 0))

        # Bottom external tools links
        bottom_frame = tk.Frame(self.root, bg="#f4f6f9", padx=20, pady=10)
        bottom_frame.pack(fill=tk.X)

        btn_api_docs = tk.Button(
            bottom_frame,
            text="🌐 Open FastAPI Docs (Swagger)",
            bg="#ffffff",
            fg="#2563eb",
            font=("Segoe UI", 9, "underline"),
            bd=0,
            cursor="hand2",
            command=lambda: webbrowser.open("http://127.0.0.1:8000/docs"),
        )
        btn_api_docs.pack(side=tk.LEFT, padx=(0, 15))

        btn_mlflow_docs = tk.Button(
            bottom_frame,
            text="📊 Open MLflow Dashboard",
            bg="#ffffff",
            fg="#059669",
            font=("Segoe UI", 9, "underline"),
            bd=0,
            cursor="hand2",
            command=lambda: webbrowser.open("http://127.0.0.1:5000"),
        )
        btn_mlflow_docs.pack(side=tk.LEFT)

        # Run an initial prediction
        self.predict()

    def create_slider(self, parent, label_text, var, from_val, to_val, row):
        f = tk.Frame(parent, bg="white")
        f.pack(fill=tk.X, pady=4)

        lbl = tk.Label(f, text=label_text, width=18, anchor="w", font=("Segoe UI", 9), bg="white")
        lbl.pack(side=tk.LEFT)

        slider = ttk.Scale(
            f,
            from_=from_val,
            to=to_val,
            variable=var,
            orient=tk.HORIZONTAL,
            command=lambda val: self.update_entry(var, val_lbl),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        val_lbl = tk.Label(f, text=f"{var.get():.1f}", width=5, font=("Consolas", 10, "bold"), bg="#f1f5f9")
        val_lbl.pack(side=tk.RIGHT)
        setattr(self, f"val_lbl_{row}", val_lbl)

    def update_entry(self, var, lbl):
        lbl.config(text=f"{var.get():.1f}")

    def set_inputs(self, sl, sw, pl, pw):
        self.val_sl.set(sl)
        self.val_sw.set(sw)
        self.val_pl.set(pl)
        self.val_pw.set(pw)

        self.val_lbl_0.config(text=f"{sl:.1f}")
        self.val_lbl_1.config(text=f"{sw:.1f}")
        self.val_lbl_2.config(text=f"{pl:.1f}")
        self.val_lbl_3.config(text=f"{pw:.1f}")

        self.predict()

    def predict(self):
        if not self.model_artifact:
            return

        clf = self.model_artifact["model"]
        target_names = self.model_artifact["target_names"]

        data = pd.DataFrame([{
            "sepal_length": round(self.val_sl.get(), 2),
            "sepal_width": round(self.val_sw.get(), 2),
            "petal_length": round(self.val_pl.get(), 2),
            "petal_width": round(self.val_pw.get(), 2),
        }])

        pred_idx = int(clf.predict(data)[0])
        species = target_names[pred_idx]
        probs = clf.predict_proba(data)[0]
        confidence = float(max(probs)) * 100

        # Colors by species
        colors = {
            "Setosa": "#059669",      # Green
            "Versicolor": "#2563eb",  # Blue
            "Virginica": "#7c3aed",   # Purple
        }
        color = colors.get(species, "#0f172a")

        self.lbl_result_species.config(text=f"{species}", fg=color)
        self.lbl_confidence.config(
            text=f"Confidence: {confidence:.1f}%  |  Model: Random Forest",
            fg="#334155",
        )

        probs_text = " | ".join([f"{target_names[i]}: {probs[i]*100:.1f}%" for i in range(len(target_names))])
        self.lbl_probs.config(text=f"Probabilities:  {probs_text}")


if __name__ == "__main__":
    root = tk.Tk()
    app = IrisMLOpsApp(root)
    root.mainloop()
