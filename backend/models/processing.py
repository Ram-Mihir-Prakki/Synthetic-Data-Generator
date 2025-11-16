import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("processing")

DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_INPUT = DATA_DIR / "loan.csv"
PROCESSED_OUTPUT = DATA_DIR / "loan_processed.csv"

class Preprocessor:
    def __init__(self):
        self.default_num_cols = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
        ]
        self.default_cat_cols = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.num_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.encoder: Optional[OrdinalEncoder] = None
        self.encoder_categories_: Optional[List[np.ndarray]] = None

    def _normalize_dependents(self, s: pd.Series) -> pd.Series:
        s = s.fillna("0").astype(str).str.strip()
        s = s.replace({"3+": "3"})
        return s

    def fit_from_dataframe(self, df: pd.DataFrame):
        df = df.copy()
        available_num = [c for c in self.default_num_cols if c in df.columns]
        available_cat = [c for c in self.default_cat_cols if c in df.columns]
        self.num_cols = available_num
        self.cat_cols = available_cat
        if self.num_cols:
            num_df = df.loc[:, self.num_cols].astype(float)
            self.num_imputer = SimpleImputer(strategy="median")
            num_imputed = self.num_imputer.fit_transform(num_df)
            self.scaler = StandardScaler()
            self.scaler.fit(num_imputed)
        else:
            self.num_imputer = None
            self.scaler = None
        if self.cat_cols:
            cat_df = df.loc[:, self.cat_cols].astype(object).copy()
            if "Dependents" in cat_df.columns:
                cat_df["Dependents"] = self._normalize_dependents(cat_df["Dependents"])
            self.cat_imputer = SimpleImputer(strategy="constant", fill_value="UNK")
            cat_imputed = self.cat_imputer.fit_transform(cat_df)
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.encoder.fit(cat_imputed)
            self.encoder_categories_ = [np.asarray(cats) for cats in self.encoder.categories_]
        else:
            self.cat_imputer = None
            self.encoder = None
            self.encoder_categories_ = None
        logger.info("Fitted preprocessor. num_cols=%s cat_cols=%s", self.num_cols, self.cat_cols)

    def fit_from_csv(self, csv_path: Optional[str] = None):
        path = Path(csv_path) if csv_path else DEFAULT_INPUT
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path)
        self.fit_from_dataframe(df)

    def transform_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_imputer is None and self.encoder is None:
            raise RuntimeError("Preprocessor not fitted")
        df = df.copy()
        if self.num_cols:
            num_df = df.loc[:, self.num_cols].astype(float)
            num_imputed = self.num_imputer.transform(num_df)
            num_scaled = self.scaler.transform(num_imputed)
        else:
            num_scaled = np.zeros((len(df), 0))
        if self.cat_cols:
            cat_df = df.loc[:, self.cat_cols].astype(object).copy()
            if "Dependents" in cat_df.columns:
                cat_df["Dependents"] = self._normalize_dependents(cat_df["Dependents"])
            cat_imputed = self.cat_imputer.transform(cat_df)
            cat_encoded = self.encoder.transform(cat_imputed)
        else:
            cat_encoded = np.zeros((len(df), 0))
        return num_scaled, cat_encoded

    def inverse_transform_rows(self, numeric_array: np.ndarray, cat_array: np.ndarray) -> List[List]:
        nums_inv = self.scaler.inverse_transform(numeric_array) if self.scaler is not None else np.zeros((numeric_array.shape[0], 0))
        cat_idx = np.rint(cat_array).astype(int) if cat_array is not None and cat_array.size else np.zeros((numeric_array.shape[0], 0), dtype=int)
        rows_out = []
        for i in range(nums_inv.shape[0]):
            row = []
            for v in nums_inv[i]:
                if np.isnan(v):
                    row.append(None)
                else:
                    if float(v).is_integer():
                        row.append(int(round(v)))
                    else:
                        row.append(float(v))
            for j, col in enumerate(self.cat_cols):
                idx = int(cat_idx[i, j]) if j < cat_idx.shape[1] else -1
                cats = self.encoder_categories_[j] if self.encoder_categories_ is not None else []
                if idx < 0 or idx >= len(cats):
                    row.append("UNK")
                else:
                    row.append(str(cats[idx]))
            rows_out.append(row)
        return rows_out

    def columns(self) -> List[str]:
        return self.num_cols + self.cat_cols

    def save_processed_csv(self, input_csv: Optional[str] = None, out_path: Optional[str] = None):
        in_path = Path(input_csv) if input_csv else DEFAULT_INPUT
        out_path = Path(out_path) if out_path else PROCESSED_OUTPUT
        if not in_path.exists():
            raise FileNotFoundError(f"{in_path} not found")
        df = pd.read_csv(in_path)
        if "Dependents" in df.columns:
            df["Dependents"] = df["Dependents"].astype(str).str.replace("3+", "3", regex=False).fillna("0")
        df = df.fillna({"Gender": "UNK", "Married": "UNK", "Self_Employed": "UNK"})
        if "LoanAmount" in df.columns:
            df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce")
        if self.scaler is None or self.encoder is None:
            self.fit_from_dataframe(df)
        df.to_csv(out_path, index=False)
        logger.info("Saved processed CSV to %s", out_path)
        return out_path

def prepare_from_kaggle(csv_path: Optional[str] = None):
    p = Path(csv_path) if csv_path else DEFAULT_INPUT
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Download it into data/ and name loan.csv or pass path.")
    pre = Preprocessor()
    pre.fit_from_csv(str(p))
    pre.save_processed_csv(str(p), str(PROCESSED_OUTPUT))
    return pre
