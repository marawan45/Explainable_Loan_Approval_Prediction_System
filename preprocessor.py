import joblib
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_FEATURES = ['home_ownership', 'loan_purpose']
NUMERIC_FEATURES = [
    'loan_amount', 'interest_rate', 'annual_income', 'dti',
    'credit_score', 'emp_length_yrs', 'num_credit_lines',
    'delinq_2yrs', 'months_since_last_delinq', 'pub_rec'
]

class LoanPreprocessor:

    def __init__(self):
        self.label_encoders = {}
        self.feature_names_out = None
        self.is_fitted = False

    def fit(self, df):
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le

        self.feature_names_out = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError('Call fit() before transform()')

        result = df[NUMERIC_FEATURES].copy()

        for col in CATEGORICAL_FEATURES:
            result[col] = self.label_encoders[col].transform(df[col])

        return result

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)