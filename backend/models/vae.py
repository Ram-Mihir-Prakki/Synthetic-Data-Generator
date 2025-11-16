import numpy as np

class VAEModel:
    def __init__(self, preprocessor):
        self.pre = preprocessor
        self.rng = np.random.RandomState(1234)
        self.min_income = 500
        self.max_income = 250000
        self.min_loan = 20
        self.max_loan = 700
        self.valid_terms = [60, 120, 180, 240, 300, 360]

    def _postprocess(self, nums, cats):
        n = nums.shape[0]
        rows = []
        nums_inv = self.pre.scaler.inverse_transform(nums) if self.pre.scaler is not None else nums
        cat_idx = np.rint(cats).astype(int) if cats is not None and cats.size else np.zeros((n, 0), dtype=int)
        for i in range(n):
            out = []
            num_vals = []
            for j, v in enumerate(nums_inv[i]):
                if np.isnan(v):
                    val = None
                else:
                    if self.pre.num_cols[j] in ("ApplicantIncome", "CoapplicantIncome"):
                        val = int(max(self.min_income, min(self.max_income, round(v * 1.05))))
                    elif self.pre.num_cols[j] == "LoanAmount":
                        val = int(max(self.min_loan, min(self.max_loan, round(v * 0.95))))
                    elif self.pre.num_cols[j] == "Loan_Amount_Term":
                        term = int(round(v))
                        choices = [t for t in self.valid_terms]
                        diffs = [abs(term - t) for t in choices]
                        val = choices[int(np.argmin(diffs))]
                    elif self.pre.num_cols[j] == "Credit_History":
                        val = 1 if v >= 0.5 else 0
                    else:
                        if float(v).is_integer():
                            val = int(round(v))
                        else:
                            val = float(v)
                num_vals.append(val)
            cat_vals = []
            for j in range(cat_idx.shape[1] if cat_idx is not None else 0):
                cats_for_j = self.pre.encoder_categories_[j] if (self.pre.encoder_categories_ is not None and j < len(self.pre.encoder_categories_)) else None
                k = len(cats_for_j) if cats_for_j is not None else 3
                idx = int(cat_idx[i, j]) if j < cat_idx.shape[1] else 0
                idx = max(0, min(k - 1, idx))
                val = str(cats_for_j[idx]) if cats_for_j is not None else "UNK"
                if self.pre.cat_cols[j] == "Dependents":
                    if val == "3+" or val == "3":
                        val = "3"
                    val = val if val is not None else "0"
                cat_vals.append(val)
            out.extend(num_vals)
            out.extend(cat_vals)
            rows.append(out)
        return rows

    def sample(self, n):
        num_dim = len(self.pre.num_cols)
        cat_dim = len(self.pre.cat_cols)
        nums = self.rng.normal(loc=0.2, scale=1.2, size=(n, num_dim)) if num_dim > 0 else np.zeros((n, 0))
        cats = np.zeros((n, cat_dim))
        for j in range(cat_dim):
            k = len(self.pre.encoder_categories_[j]) if (self.pre.encoder_categories_ and j < len(self.pre.encoder_categories_)) else 3
            cats[:, j] = np.clip(np.abs(self.rng.randint(0, k+1, size=n)), 0, max(0, k-1))
        rows_body = self._postprocess(nums, cats)
        out = []
        for i, r in enumerate(rows_body):
            out.append([f"r{i+1:02d}"] + r)
        return out
