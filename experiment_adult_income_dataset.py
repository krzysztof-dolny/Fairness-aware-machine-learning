# =========================== #
#        IMPORTS & SETUP      #
# =========================== #
import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from fairness.losses import fair_bce_loss, calculate_alpha
from fairness.metrics import accuracy_equality, statistical_parity, equal_opportunity, predictive_equality

# Set project root and output directory
project_root = os.getcwd()
sys.path.append(project_root)
output_dir = os.path.join(project_root, 'outputs')
os.makedirs(output_dir, exist_ok=True)


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================== #
#        CONFIG LOADER        #
# =========================== #
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================== #
#        DATA LOADING         #
# =========================== #
def load_data():
    dataset = fetch_openml("adult", version=2, as_frame=True)
    df = dataset.data.copy()
    df['target'] = dataset.target
    df['sex'] = LabelEncoder().fit_transform(df['sex'])

    for col, fill_value in {
        "native-country": "Unknown",
        "workclass": "Unemployed",
        "occupation": "Unemployed"
    }.items():
        if fill_value not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories([fill_value])
        df[col] = df[col].fillna(fill_value)

    selected_cols = ['age', 'workclass', 'education-num', 'marital-status', 'occupation',
                     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country']

    X = pd.DataFrame(OrdinalEncoder().fit_transform(df[selected_cols]), columns=selected_cols)
    y = pd.DataFrame(LabelEncoder().fit_transform(df['target']), columns=['target'])

    X_train, X_valtest, y_train, y_valtest, sex_train, sex_valtest = train_test_split(
        X, y, df['sex'], test_size=0.3, random_state=42, stratify=df['sex']
    )
    X_val, X_test, y_val, y_test, sex_val, sex_test = train_test_split(
        X_valtest, y_valtest, sex_valtest, test_size=0.5, random_state=42, stratify=sex_valtest
    )

    scaler = StandardScaler()
    to_tensor = lambda df: torch.tensor(df.values, dtype=torch.float32)

    return (
        torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32),
        torch.tensor(scaler.transform(X_val), dtype=torch.float32),
        torch.tensor(scaler.transform(X_test), dtype=torch.float32),
        to_tensor(y_train), to_tensor(y_val), to_tensor(y_test),
        to_tensor(sex_train), to_tensor(sex_val), to_tensor(sex_test)
    )


# =========================== #
#         MODELS              #
# =========================== #
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


def load_model(ml_algorithm, input_dim):
    return MLP(input_dim) if ml_algorithm == "MLP" else LogisticRegression(input_dim)


# =========================== #
#      TRAIN & TEST LOGIC     #
# =========================== #
def train_model(model, optimizer, epochs, patience, alpha, alpha_mode, fairness_mode,
                X_train, y_train, sex_train, X_val, y_val):
    best_val_acc, best_model_state, epochs_no_improve = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = fair_bce_loss(outputs, y_train, sex_train,
                             alpha=calculate_alpha(epoch, epochs, alpha, alpha_mode),
                             fairness_mode=fairness_mode)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = ((model(X_val) > 0.5).float() == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc, best_model_state, epochs_no_improve = val_acc, model.state_dict(), 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve > patience:
            break

    return best_model_state


def test_model(model, X_test, y_test, sex_test, fairness_mode):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        acc = ((outputs > 0.5).float() == y_test).float().mean().item()

        if fairness_mode == "AE":
            fairness_score, _ = accuracy_equality(outputs.squeeze(), y_test, sex_test)
        elif fairness_mode == "SP":
            fairness_score, _ = statistical_parity(outputs, sex_test)
        elif fairness_mode == "EOP":
            fairness_score, _ = equal_opportunity(outputs, y_test.squeeze(), sex_test)
        elif fairness_mode == "PE":
            fairness_score, _ = predictive_equality(outputs, y_test.squeeze(), sex_test)
        else:
            raise ValueError(f"Unsupported fairness mode: {fairness_mode}")

    return acc, fairness_score


# =========================== #
#        EXPERIMENT LOOP      #
# =========================== #
def run_experiments(config, X_train, y_train, sex_train, X_val, y_val, sex_val, X_test, y_test, sex_test):
    results = []
    best_score = best_acc = -float("inf")
    best_fairness = float("inf")
    best_model_score = best_model_acc = best_model_fair = None
    best_result_score = best_result_acc = best_result_fair = None

    for alpha in config["alpha_values"]:
        for alpha_mode in config["alpha_modes"]:
            print(f"\nRunning {config['ml_algorithm']}"
                  f" | alpha={alpha}, mode={alpha_mode}, fairness={config['fairness_mode']}")

            model = load_model(config["ml_algorithm"], X_train.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            best_model_state = train_model(model, optimizer, config["epochs"], config["patience"], alpha, alpha_mode,
                                           config["fairness_mode"], X_train, y_train, sex_train, X_val, y_val)
            model.load_state_dict(best_model_state)

            acc, fairness = test_model(model, X_test, y_test, sex_test, config["fairness_mode"])
            print(f"Test Accuracy: {acc:.3f}, Fairness ({config['fairness_mode']}): {fairness:.3f}")

            row = {
                "ml_algorithm": config["ml_algorithm"],
                "fairness_mode": config["fairness_mode"],
                "fairness_score": fairness,
                "test_accuracy": acc,
                "alpha_mode": alpha_mode,
                "alpha_value": alpha
            }
            results.append(row)

            score = acc - fairness
            if score > best_score:
                best_score, best_model_score, best_result_score = score, best_model_state, row
            if acc > best_acc or (acc == best_acc and fairness < best_result_acc["fairness_score"]):
                best_acc, best_model_acc, best_result_acc = acc, best_model_state, row
            if fairness < best_fairness or (fairness == best_fairness and acc > best_result_fair["test_accuracy"]):
                best_fairness, best_model_fair, best_result_fair = fairness, best_model_state, row

    print("\nBest score config:", best_result_score)
    print("\nBest accuracy config:", best_result_acc)
    print("\nBest fairness config:", best_result_fair)

    return save_results(results, output_dir, best_model_score, best_model_acc, best_model_fair)


# =========================== #
#    Additional Functions     #
# =========================== #
def save_results(results, output_dir, model_score, model_acc, model_fair):
    torch.save(model_score, os.path.join(output_dir, "best_model_score.pth"))
    torch.save(model_acc, os.path.join(output_dir, "best_model_acc.pth"))
    torch.save(model_fair, os.path.join(output_dir, "best_model_fair.pth"))

    df = pd.DataFrame(results)
    file = os.path.join(output_dir, "results.csv")
    df.to_csv(file, mode='a', header=not os.path.exists(file), index=False)
    return df


# =========================== #
#           MAIN              #
# =========================== #
def main():
    set_seed(123)
    config = load_config("config.yaml")
    X_train, X_val, X_test, y_train, y_val, y_test, sex_train, sex_val, sex_test = load_data()
    df = run_experiments(config, X_train, y_train, sex_train, X_val, y_val, sex_val, X_test, y_test, sex_test)
    print("\nFinal results:")
    print(df)


if __name__ == "__main__":
    main()
