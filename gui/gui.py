# gui.py
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QTextEdit
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model import DiabetesModel
from xai_utils import PERM_AVAILABLE, explain_with_permutation

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class DiabetesPredictorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Diagnosis Predictor - XAI")
        self.setGeometry(100, 100, 1500, 760)

        # Model
        self.model = DiabetesModel()
        self.feature_names = self.model.feature_names
        self.metadata = ['PatientID', 'Name', 'Gender']
        self.all_columns = self.metadata + self.feature_names + ['Prediction', 'Probability']

        # Data
        self.all_records_df = pd.DataFrame(columns=self.all_columns)
        self.uploaded_data = None
        self.uploaded_meta = None
        self.current_patient_idx = 0
        self.num_patients = 0

        # Figures
        self.fig_lime = Figure()
        self.fig_perm = Figure()
        self.fig_prob = Figure()
        self.canvas_lime = FigureCanvas(self.fig_lime)
        self.canvas_perm = FigureCanvas(self.fig_perm)
        self.canvas_prob = FigureCanvas(self.fig_prob)

        # UI setup
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # ----- Left panel: Inputs -----
        left_panel.addWidget(QLabel("Patient Input"))
        self.upload_btn = QPushButton("Upload CSV")
        self.upload_btn.clicked.connect(self.load_file)
        left_panel.addWidget(self.upload_btn)

        self.manual_btn = QPushButton("Enter Manually")
        self.manual_btn.clicked.connect(self.show_manual_inputs)
        left_panel.addWidget(self.manual_btn)

        self.save_btn = QPushButton("Save Records")
        self.save_btn.clicked.connect(self.save_records)
        left_panel.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Records")
        self.load_btn.clicked.connect(self.load_records)
        left_panel.addWidget(self.load_btn)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Patient")
        self.prev_btn.clicked.connect(self.show_prev_patient)
        self.next_btn = QPushButton("Next Patient")
        self.next_btn.clicked.connect(self.show_next_patient)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        left_panel.addLayout(nav_layout)

        # Manual input fields
        self.manual_layout = QVBoxLayout()
        self.manual_inputs = []
        for col in self.metadata:
            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            entry = QLineEdit()
            row.addWidget(entry)
            self.manual_layout.addLayout(row)
            self.manual_inputs.append((col, entry))
        for col in self.feature_names:
            row = QHBoxLayout()
            row.addWidget(QLabel(col))
            entry = QLineEdit()
            if col in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','Age']:
                entry.setValidator(QIntValidator(0, 200))
            else:
                entry.setValidator(QDoubleValidator(0.0, 100.0, 2))
            row.addWidget(entry)
            self.manual_layout.addLayout(row)
            self.manual_inputs.append((col, entry))
        self.manual_widget = QWidget()
        self.manual_widget.setLayout(self.manual_layout)
        self.manual_widget.hide()
        left_panel.addWidget(self.manual_widget)

        self.predict_btn = QPushButton("Predict Current Patient")
        self.predict_btn.clicked.connect(self.predict_manual)
        left_panel.addWidget(self.predict_btn)

        main_layout.addLayout(left_panel, 2)

        # ----- Right panel: Outputs -----
        top_labels = QHBoxLayout()
        self.pred_label = QLabel("Prediction: ---")
        self.prob_label = QLabel("Probability: ---")
        top_labels.addWidget(self.pred_label)
        top_labels.addWidget(self.prob_label)
        right_panel.addLayout(top_labels)

        self.tabs = QTabWidget()
        # LIME tab
        if LIME_AVAILABLE:
            self.tabs.addTab(self.canvas_lime, "Local Explanation (LIME)")
        else:
            self.lime_msg = QTextEdit("LIME not installed. Local explanations disabled.")
            self.lime_msg.setReadOnly(True)
            self.tabs.addTab(self.lime_msg, "Local Explanation (LIME)")
        # Permutation importance tab
        self.tabs.addTab(self.canvas_perm, "Global Importance (Permutation)")
        # Probability tab
        self.tabs.addTab(self.canvas_prob, "Probability Distribution")
        # Records tab
        self.records_table = QTableWidget()
        self.records_table.setColumnCount(len(self.all_columns))
        self.records_table.setHorizontalHeaderLabels(self.all_columns)
        self.tabs.addTab(self.records_table, "Patient Records")
        right_panel.addWidget(self.tabs)

        main_layout.addLayout(right_panel, 5)
        self.setLayout(main_layout)

    # ------------------ Manual input ------------------
    def show_manual_inputs(self):
        self.manual_widget.show()

    def predict_manual(self):
        values, meta_vals = [], []
        for label, entry in self.manual_inputs:
            text = entry.text()
            if not text:
                QMessageBox.warning(self, "Missing Data", f"Please enter {label}")
                return
            if label in self.metadata:
                meta_vals.append(text)
            else:
                values.append(float(text))
        self.uploaded_data = np.array([values])
        self.uploaded_meta = np.array([meta_vals])
        self.num_patients = 1
        self.current_patient_idx = 0
        self.display_current_patient()

    # ------------------ CSV Upload ------------------
    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not path: return
        df = pd.read_csv(path)
        missing_cols = [c for c in self.metadata + self.feature_names if c not in df.columns]
        if missing_cols:
            QMessageBox.critical(self, "Missing Columns", f"Missing columns: {', '.join(missing_cols)}")
            return
        self.uploaded_meta = df[self.metadata].values
        self.uploaded_data = df[self.feature_names].values.astype(float)
        self.num_patients = len(df)
        self.current_patient_idx = 0
        self.display_current_patient()
        QMessageBox.information(self, "Loaded", f"Loaded {self.num_patients} patient(s)")

    # ------------------ Display patient ------------------
    def display_current_patient(self):
        if self.uploaded_data is None: return
        idx = self.current_patient_idx
        patient_numeric = self.uploaded_data[idx].reshape(1, -1)
        patient_meta = self.uploaded_meta[idx]

        pred_class, pred_prob = self.model.predict(patient_numeric)
        self.pred_label.setText(f"Prediction: {'Diabetes Likely' if pred_class[0]==1 else 'Not Likely'}")
        self.prob_label.setText(f"Probability: {pred_prob[0]*100:.1f}%")

        # Update LIME
        if LIME_AVAILABLE:
            try:
                explainer = LimeTabularExplainer(
                    self.uploaded_data, feature_names=self.feature_names,
                    class_names=['No Diabetes', 'Diabetes'], discretize_continuous=True
                )
                lime_exp = explainer.explain_instance(patient_numeric.flatten(), self.model.model.predict_proba, num_features=8)
                vals = lime_exp.as_list()
                self._draw_lime(vals)
            except Exception as e:
                self.lime_msg.setPlainText(f"LIME error: {e}")

        # Update permutation importance
        self._draw_permutation_importance()

        # Probability chart
        self._draw_probability(pred_prob[0])

        # Update records table
        patient_id = str(patient_meta[0])
        if patient_id not in self.all_records_df['PatientID'].astype(str).values:
            rec = pd.DataFrame([list(patient_meta) + list(patient_numeric.flatten())],
                               columns=self.metadata + self.feature_names)
            rec['Prediction'] = ['Diabetes Likely' if pred_class[0]==1 else 'Not Likely']
            rec['Probability'] = pred_prob
            self.all_records_df = pd.concat([self.all_records_df, rec], ignore_index=True)
            self.update_records_table()

    # ------------------ Drawing functions ------------------
    def _draw_lime(self, lime_list):
        feature_labels, weights = zip(*lime_list)
        fig = self.fig_lime
        fig.clear()
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(feature_labels))
        ax.barh(y_pos, weights, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel("Weight")
        ax.set_title("Local Explanation (LIME)")
        fig.tight_layout()
        self.canvas_lime.draw()

    def _draw_permutation_importance(self):
        if PERM_AVAILABLE:
            try:
                df = explain_with_permutation(self.model)
                fig = self.fig_perm
                fig.clear()
                ax = fig.add_subplot(111)
                ax.barh(df['feature'], df['importance_mean'], xerr=df['importance_std'], color='skyblue')
                ax.set_xlabel("Permutation Importance (Mean ± STD)")
                ax.set_title("Global Feature Importance")
                fig.tight_layout()
                self.canvas_perm.draw()
            except Exception as e:
                fig = self.fig_perm
                fig.clear()
                ax = fig.add_subplot(111)
                ax.text(0.05, 0.5, f"Error: {e}", fontsize=10, wrap=True)
                ax.axis('off')
                fig.tight_layout()
                self.canvas_perm.draw()

    def _draw_probability(self, prob):
        fig = self.fig_prob
        fig.clear()
        ax = fig.add_subplot(111)
        ax.pie([1-prob, prob], labels=['No Diabetes','Diabetes'], autopct="%.1f%%")
        ax.set_title("Probability Distribution")
        fig.tight_layout()
        self.canvas_prob.draw()

    # ------------------ Navigation ------------------
    def show_next_patient(self):
        if self.uploaded_data is None: return
        self.current_patient_idx = (self.current_patient_idx + 1) % self.num_patients
        self.display_current_patient()

    def show_prev_patient(self):
        if self.uploaded_data is None: return
        self.current_patient_idx = (self.current_patient_idx - 1) % self.num_patients
        self.display_current_patient()

    # ------------------ Records Table ------------------
    def update_records_table(self):
        df = self.all_records_df
        self.records_table.setRowCount(len(df))
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                val = row[col]
                display = f"{val:.3f}" if isinstance(val, float) else str(val)
                self.records_table.setItem(i, j, QTableWidgetItem(display))

    def save_records(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Records", "", "CSV Files (*.csv)")
        if path:
            self.all_records_df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Records saved to {path}")

    def load_records(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Records", "", "CSV Files (*.csv)")
        if path:
            self.all_records_df = pd.read_csv(path)
            self.update_records_table()
            QMessageBox.information(self, "Loaded", f"Loaded {len(self.all_records_df)} records")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiabetesPredictorGUI()
    window.show()
    sys.exit(app.exec_())
