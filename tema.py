import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

# 3.2
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # Necesare pentru IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# end 3.2


# 3.3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# end 3.3




# Setari generale pentru grafice
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# Create directory for saving graphs
output_dir_3_1 = 'graphs_3.1'
output_dir_3_2 = 'graphs_3.2'
if not os.path.exists(output_dir_3_1):
    os.makedirs(output_dir_3_1)

if not os.path.exists(output_dir_3_2):
    os.makedirs(output_dir_3_2)

# Citirea Datelor

credit_risk_train = pd.read_csv('./tema2_Credit_Risk/credit_risk_train.csv')
diabet_train = pd.read_csv('./tema2_Diabet/Diabet_train.csv')



# 3.3
credit_risk_test = pd.read_csv('./tema2_Credit_Risk/credit_risk_test.csv')
diabet_test = pd.read_csv('./tema2_Diabet/Diabet_test.csv')

# end 3.3


# Identificarea Tipurilor de Atribute
# numerica, categorica, ordinala
def attribute_analysis(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    ordinal_cols = [col for col in df.columns if df[col].dtype == 'int64' and col not in numerical_cols]
    
    return numerical_cols, categorical_cols, ordinal_cols

credit_numerical_cols, credit_categorical_cols, credit_ordinal_cols = attribute_analysis(credit_risk_train)
diabet_numerical_cols, diabet_categorical_cols, diabet_ordinal_cols = attribute_analysis(diabet_train)


# Statistici pentru Atribute Numerice

def numerical_statistics(df, numerical_cols):
    stats = df[numerical_cols].describe().transpose()
    stats['missing_values'] = df[numerical_cols].isna().sum()
    return stats

credit_numerical_stats = numerical_statistics(credit_risk_train, credit_numerical_cols)
diabet_numerical_stats = numerical_statistics(diabet_train, diabet_numerical_cols)
print("Credit Risk Numerical Stats:\n", credit_numerical_stats)
print("Diabetes Numerical Stats:\n", diabet_numerical_stats)


# Boxplots pentru Atribute Numerice

def plot_boxplots(df, numerical_cols, output_dir, dataset_name):
    num_cols = len(numerical_cols)
    fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, num_cols * 5))
    
    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Values')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_boxplots.png')
    plt.close()

plot_boxplots(credit_risk_train, credit_numerical_cols, output_dir_3_1, 'credit_risk')
plot_boxplots(diabet_train, diabet_numerical_cols, output_dir_3_1, 'diabet')


# Histogramă pentru Atribute Categoriale și Ordinale

def plot_histograms(df, categorical_cols, output_dir, dataset_name):
    num_cols = len(categorical_cols)
    fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(10, num_cols * 5))
    
    for i, col in enumerate(categorical_cols):
        df[col].value_counts().plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_histograms.png')
    plt.close()

plot_histograms(credit_risk_train, credit_categorical_cols, output_dir_3_1, 'credit_risk')
plot_histograms(diabet_train, diabet_categorical_cols, output_dir_3_1, 'diabet')


# Număr de Valori Unice și Lipsă pentru Atribute Categoriale

def categorical_statistics(df, categorical_cols):
    stats = pd.DataFrame(columns=['num_unique', 'missing_values'])
    stats['num_unique'] = df[categorical_cols].nunique()
    stats['missing_values'] = df[categorical_cols].isna().sum()
    return stats

credit_categorical_stats = categorical_statistics(credit_risk_train, credit_categorical_cols)
diabet_categorical_stats = categorical_statistics(diabet_train, diabet_categorical_cols)
print("Credit Risk Categorical Stats:\n", credit_categorical_stats)
print("Diabetes Categorical Stats:\n", diabet_categorical_stats)

# 2. Analiza Echilibrului de Clase

def plot_class_balance(df, target_col, output_dir, dataset_name):
    sns.countplot(x=target_col, data=df)
    plt.title('Class Distribution')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/{dataset_name}_class_balance.png')
    plt.close()

plot_class_balance(credit_risk_train, 'loan_approval_status', output_dir_3_1, 'credit_risk')
plot_class_balance(diabet_train, 'Diabetes', output_dir_3_1, 'diabet')

# 3. Analiza Corelației Între Atribute

# Corelația pentru Atribute Numerice

def plot_correlation_matrix(df, numerical_cols, output_dir, dataset_name):
    corr_matrix = df[numerical_cols].corr(method='pearson')
    plt.matshow(corr_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title('Correlation Matrix', pad=12)
    plt.xticks(ticks=np.arange(len(numerical_cols)), labels=numerical_cols, rotation=90)
    plt.yticks(ticks=np.arange(len(numerical_cols)), labels=numerical_cols)
    plt.savefig(f'{output_dir}/{dataset_name}_correlation_matrix.png')
    plt.close()

plot_correlation_matrix(credit_risk_train, credit_numerical_cols, output_dir_3_1, 'credit_risk')
plot_correlation_matrix(diabet_train, diabet_numerical_cols, output_dir_3_1, 'diabet')


# Testul Chi-Pătrat pentru Atribute Categoriale

def chi2_test_of_independence(df, categorical_cols):
    results = {}
    for i in range(len(categorical_cols)):
        for j in range(i+1, len(categorical_cols)):
            col1 = categorical_cols[i]
            col2 = categorical_cols[j]
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            results[(col1, col2)] = p
    return results

credit_chi2_results = chi2_test_of_independence(credit_risk_train, credit_categorical_cols)
diabet_chi2_results = chi2_test_of_independence(diabet_train, diabet_categorical_cols)

print('\n')
print("Credit Risk Chi-Square Results:", credit_chi2_results)
print('\n')
print("Diabetes Chi-Square Results:", diabet_chi2_results)



# 3.2

# Imputarea valorilor lipsă
def imputare_valori_lipse(df, numerical_cols, categorical_cols):
    # Imputare univariată pentru atributele numerice (valoarea medie)
    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
    
    # Imputare univariată pentru atributele categorice (valoarea cea mai frecventă)
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    
    return df

credit_risk_train = imputare_valori_lipse(credit_risk_train, credit_numerical_cols, credit_categorical_cols)
diabet_train = imputare_valori_lipse(diabet_train, diabet_numerical_cols, diabet_categorical_cols)


# Tratarea valorilor extreme
def tratare_valori_extreme(df, numerical_cols):
    Q1 = df[numerical_cols].quantile(0.25)


    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    for col in numerical_cols:
        df[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])
        df[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df[col])
    
    return df

credit_risk_train = tratare_valori_extreme(credit_risk_train, credit_numerical_cols)
diabet_train = tratare_valori_extreme(diabet_train, diabet_numerical_cols)


# Eliminarea atributelor redundante
def elimina_atribute_redundante(df, redundant_cols):
    return df.drop(columns=redundant_cols)

credit_redundant_cols = []  # Adăugați aici coloanele redundante determinate din analiza corelației
diabet_redundant_cols = []  # Adăugați aici coloanele redundante determinate din analiza corelației

credit_risk_train = elimina_atribute_redundante(credit_risk_train, credit_redundant_cols)
diabet_train = elimina_atribute_redundante(diabet_train, diabet_redundant_cols)


# Standardizarea atributelor numerice
def standardizare_atribute(df, numerical_cols, method='standard'):
    if method == 'standard':
        scaler = StandardScaler() # Standardizare (mean = 0, standard deviation = 1).
    elif method == 'minmax':
        scaler = MinMaxScaler() # Normalizare (transformare în intervalul [0, 1]).
    elif method == 'robust':
        scaler = RobustScaler() # Scalare robustă (folosește mediane și interquartile range pentru a fi mai rezistentă la outlieri).
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

credit_risk_train = standardizare_atribute(credit_risk_train, credit_numerical_cols, method='standard')
diabet_train = standardizare_atribute(diabet_train, diabet_numerical_cols, method='standard')



# Generarea din nou a boxplot-urilor și histogramelor pentru a vedea cum arată datele după preprocesare
plot_boxplots(credit_risk_train, credit_numerical_cols, output_dir_3_2, 'credit_risk_preprocessed')
plot_boxplots(diabet_train, diabet_numerical_cols, output_dir_3_2, 'diabet_preprocessed')

plot_histograms(credit_risk_train, credit_categorical_cols, output_dir_3_2, 'credit_risk_preprocessed')
plot_histograms(diabet_train, diabet_categorical_cols, output_dir_3_2, 'diabet_preprocessed')

# Analiza echilibrului de clase după preprocesare
plot_class_balance(credit_risk_train, 'loan_approval_status', output_dir_3_2, 'credit_risk_preprocessed')
plot_class_balance(diabet_train, 'Diabetes', output_dir_3_2, 'diabet_preprocessed')

# Analiza corelației între atribute numerice după preprocesare
plot_correlation_matrix(credit_risk_train, credit_numerical_cols, output_dir_3_2, 'credit_risk_preprocessed')
plot_correlation_matrix(diabet_train, diabet_numerical_cols, output_dir_3_2, 'diabet_preprocessed')

print('\n')
credit_categorical_stats_after = categorical_statistics(credit_risk_train, credit_categorical_cols)
diabet_categorical_stats_after = categorical_statistics(diabet_train, diabet_categorical_cols)
print("Credit Risk Categorical Stats After Preprocessing:\n", credit_categorical_stats_after)
print("Diabetes Categorical Stats After Preprocessing:\n", diabet_categorical_stats_after)




# 3.3 - 1.1 RandomForest cu scikit-learn

# Convertirea variabilelor categoriale în variabile numerice folosind One-Hot Encoding

# Convertirea variabilelor categoriale pentru credit risk in variabile numerice binare
credit_risk_train = pd.get_dummies(credit_risk_train)
credit_risk_test = pd.get_dummies(credit_risk_test)

# Asigurarea că setul de date de antrenament și test au aceleași coloane
credit_risk_train, credit_risk_test = credit_risk_train.align(credit_risk_test, join='left', axis=1) # align pentru a asigura ca train si test au aceleasi coloane
credit_risk_test = credit_risk_test.fillna(0)  # fillna completeaza valorile lipsa din aliniere

# Convertirea variabilelor categoriale pentru diabet
diabet_train = pd.get_dummies(diabet_train)
diabet_test = pd.get_dummies(diabet_test)

# Asigurarea că setul de date de antrenament și test au aceleași coloane
diabet_train, diabet_test = diabet_train.align(diabet_test, join='left', axis=1)
diabet_test = diabet_test.fillna(0)


# Separați caracteristicile (features) de etichete (labels)   -- labels sunt iesirile si features sunt de intrare pentru a face predictii
# Split features and labels
X_credit_train = credit_risk_train.drop('loan_approval_status_Approved', axis=1) # elimina coloana tinta lasand doar caracteristicele
y_credit_train = credit_risk_train['loan_approval_status_Approved']

X_credit_test = credit_risk_test.drop('loan_approval_status_Approved', axis=1)
y_credit_test = credit_risk_test['loan_approval_status_Approved']

X_diabet_train = diabet_train.drop('Diabetes', axis=1)
y_diabet_train = diabet_train['Diabetes']

X_diabet_test = diabet_test.drop('Diabetes', axis=1)
y_diabet_test = diabet_test['Diabetes']



# Ajustarea hiperparametrilor pentru RandomForest
rf_params = {
    'n_estimators': 100, # Numarul de arbori in padurea aleatorie
    'max_depth': 10,  # Reduceți adâncimea maximă
    'min_samples_split': 10,  # Creșteți numărul minim de exemple pentru un split
    'min_samples_leaf': 5,  # Creșteți numărul minim de exemple într-o frunză
    'criterion': 'gini',  # Criteriu pentru masurarea calitatii unui split
    'class_weight': 'balanced',  # Ajustați ponderile claselor
    'max_features': 'sqrt',  # Limitați numărul de caracteristici folosite la fiecare split
    'random_state': 42
}

# Antrenarea modelului RandomForest pentru credit risk pe datele de antrenament
rf_credit = RandomForestClassifier(**rf_params)
rf_credit.fit(X_credit_train, y_credit_train)

# Antrenarea modelului RandomForest pentru diabet
rf_diabet = RandomForestClassifier(**rf_params)
rf_diabet.fit(X_diabet_train, y_diabet_train)


def evaluate_model(model, X_train, y_train, X_test, y_test, dataset_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"\nEvaluarea modelului pe setul de date {dataset_name}:")
    
    # Matricea de confuzie și raportul de clasificare pentru setul de antrenament
    print("\nMatricea de confuzie pentru setul de antrenament:")
    print(confusion_matrix(y_train, y_train_pred))
    print("\nRaportul de clasificare pentru setul de antrenament:")
    print(classification_report(y_train, y_train_pred))
    
    # Matricea de confuzie și raportul de clasificare pentru setul de test
    print("\nMatricea de confuzie pentru setul de test:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nRaportul de clasificare pentru setul de test:")
    print(classification_report(y_test, y_test_pred))
    
    
    # Acuratețea modelului
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Acuratețea pe setul de antrenament: {train_accuracy}")
    print(f"Acuratețea pe setul de test: {test_accuracy}")

# Evaluarea modelului pentru credit risk
evaluate_model(rf_credit, X_credit_train, y_credit_train, X_credit_test, y_credit_test, "Credit Risk")

# Evaluarea modelului pentru diabet
evaluate_model(rf_diabet, X_diabet_train, y_diabet_train, X_diabet_test, y_diabet_test, "Diabet")






# 3.3 - 1.2 manual plecand de la codul din laborator, sarcini de clasificare si regresie. Construieste mmultiple arbori de decizie
# si combina rezultatele pentru a face predictii mai bune

import random

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0): # construieste arborele de decizie recursiv, determina pragul de impartire folosind _best_criteria
        num_samples, num_features = X.shape
        if num_samples < self.min_samples_leaf or depth == self.max_depth:
            # returnam cel mai comun label si oprim recursivitatea
            return np.bincount(y).argmax()

        if len(set(y)) == 1:
            return y[0]

        best_feature, best_threshold = self._best_criteria(X, y, num_features)
        if best_feature is None:
            return np.bincount(y).argmax()

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return np.bincount(y).argmax()

        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return (best_feature, best_threshold, left, right)

    def _best_criteria(self, X, y, num_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature_idx in range(num_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column <= threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, split): # calculeaza castigul informatiei bazat pe entropia parintelui si a copiilor, pentru a masura cat de bine o impartire separa datele
        # entropia parinterului
        parent_entropy = self._entropy(y)
        # entropia copiilor
        left, right = y[split], y[~split]
        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left), len(right)
        e_left, e_right = self._entropy(left), self._entropy(right)
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # informatia castigata
        return parent_entropy - child_entropy

    def _entropy(self, y): # Calculeaaza entropia care este o masura a impuritatii unui set de date
        hist = np.bincount(y)
        ps = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            feature_idx, threshold, left, right = node
            if inputs[feature_idx] <= threshold:
                node = left
            else:
                node = right
        return node

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.criterion)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(preds, minlength=2).argmax() for preds in tree_preds.T])

# Antrenarea și evaluarea modelului RandomForest implementat manual
rf_custom = RandomForest(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, criterion='gini', random_state=42)

# Convert X_credit_train and y_credit_train to numpy arrays
X_credit_train_np = X_credit_train.to_numpy()
y_credit_train_np = y_credit_train.to_numpy()

# Diabetes dataset
X_diabet_train_np = X_diabet_train.to_numpy()
y_diabet_train_np = y_diabet_train.to_numpy()

# Antrenarea modelului
rf_custom.fit(X_credit_train_np, y_credit_train_np)




# Evaluarea modelului RandomForest implementat manual
def evaluate_model_custom(model, X_train, y_train, X_test, y_test, dataset_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"\nEvaluarea modelului pe setul de date {dataset_name}:")
    
    # Matricea de confuzie și raportul de clasificare pentru setul de antrenament
    print("\nMatricea de confuzie pentru setul de antrenament:")
    print(confusion_matrix(y_train, y_train_pred))
    print("\nRaportul de clasificare pentru setul de antrenament:")
    print(classification_report(y_train, y_train_pred))
    
    # Matricea de confuzie și raportul de clasificare pentru setul de test
    print("\nMatricea de confuzie pentru setul de test:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nRaportul de clasificare pentru setul de test:")
    print(classification_report(y_test, y_test_pred))
    
    # Acuratețea modelului
    train_accuracy = np.mean(y_train == y_train_pred)
    test_accuracy = np.mean(y_test == y_test_pred)
    
    print(f"Acuratețea pe setul de antrenament: {train_accuracy}")
    print(f"Acuratețea pe setul de test: {test_accuracy}")

# Convertim la numpy arrays
X_credit_test_np = X_credit_test.to_numpy()
y_credit_test_np = y_credit_test.to_numpy()

# Diabetes dataset
X_diabet_test_np = X_diabet_test.to_numpy()
y_diabet_test_np = y_diabet_test.to_numpy()

evaluate_model_custom(rf_custom, X_credit_train_np, y_credit_train_np, X_credit_test_np, y_credit_test_np, "Credit Risk - Laborator RandomForest")
evaluate_model_custom(rf_custom, X_diabet_train_np, y_diabet_train_np, X_diabet_test_np, y_diabet_test_np, "Diabetes - Laborator RandomForest")






# 3.3 2.2 MLP cu scikit-learn


# Setarea unui director pentru salvarea rezultatelor
output_dir_MLP = 'output_MLP'
if not os.path.exists(output_dir_MLP):
    os.makedirs(output_dir_MLP)

# Model MLP pentru credit. retea cu 2 straturi ascunse, 50 neuroni si 30 neuroni, antrenamentul se opreste anticipat daca nu exista imbunatatiri timp de 10 epoci consecutive
mlp_credit = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, learning_rate_init=0.1, random_state=42, validation_fraction=0.1, early_stopping=True, n_iter_no_change=10)
mlp_credit.fit(X_credit_train, y_credit_train)

# random_state=42, seed pentru reproducibilitatea rezultatelor
# Model MLP pentru diabet
mlp_diabet = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, learning_rate_init=0.1, random_state=42, validation_fraction=0.1, early_stopping=True, n_iter_no_change=10)
mlp_diabet.fit(X_diabet_train, y_diabet_train)

def evaluate_mlp(model, X_train, y_train, X_test, y_test, dataset_name):
    # Predictii
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcularea acurateții
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nEvaluarea modelului pe setul de date {dataset_name}:")
    
    # Matricea de confuzie și raportul de clasificare pentru setul de antrenament
    print("\nMatricea de confuzie pentru setul de antrenament:")
    print(confusion_matrix(y_train, y_train_pred))
    print("\nRaportul de clasificare pentru setul de antrenament:")
    print(classification_report(y_train, y_train_pred))
    
    # Matricea de confuzie și raportul de clasificare pentru setul de test
    print("\nMatricea de confuzie pentru setul de test:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nRaportul de clasificare pentru setul de test:")
    print(classification_report(y_test, y_test_pred))
    
    # Acuratețea modelului
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Acuratețea pe setul de antrenament: {train_accuracy}")
    print(f"Acuratețea pe setul de test: {test_accuracy}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_curve_, label='Train Loss')
    plt.title(f'Training Loss Curve - {dataset_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([train_accuracy] * len(model.loss_curve_), label='Train Accuracy', linestyle='--')
    plt.plot([test_accuracy] * len(model.loss_curve_), label='Test Accuracy', linestyle='--')
    plt.title(f'Accuracy - {dataset_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir_MLP}/{dataset_name}_performance.png')



# Evaluarea MLP pentru credit risk
evaluate_mlp(mlp_credit, X_credit_train, y_credit_train, X_credit_test, y_credit_test, "Credit Risk - MLP")

# Evaluarea MLP pentru diabet
evaluate_mlp(mlp_diabet, X_diabet_train, y_diabet_train, X_diabet_test, y_diabet_test, "Diabetes - MLP")
