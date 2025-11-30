import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

# ============================================================================
# DATA GENERATION / PREPROCESSING
# ============================================================================

def generate_synthetic_RA_data(n_patients=1000, n_features=21):
    np.random.seed(42)
    DAS28_CRP = np.random.normal(5.7, 1.2, n_patients)
    SJC28 = np.random.poisson(10, n_patients).astype(float)
    TJC28 = np.random.poisson(15, n_patients).astype(float)
    CRP = np.random.exponential(20, n_patients)
    ESR = np.random.exponential(30, n_patients)
    PtGA = np.random.uniform(40, 80, n_patients)
    PhGA = np.random.uniform(40, 75, n_patients)
    VAS_Pain = np.random.uniform(45, 85, n_patients)
    HAQ_DI = np.random.uniform(1.0, 2.5, n_patients)
    Age = np.random.normal(55, 12, n_patients)
    Sex = np.random.binomial(1, 0.75, n_patients)
    BMI = np.random.normal(27, 5, n_patients)
    Disease_Duration = np.random.exponential(5, n_patients)
    RF_Positive = np.random.binomial(1, 0.7, n_patients)
    AntiCCP_Positive = np.random.binomial(1, 0.65, n_patients)
    Prior_DMARDs = np.random.poisson(2, n_patients).astype(float)
    Prior_Biologics = np.random.poisson(1, n_patients).astype(float)
    MTX_Naive = np.random.binomial(1, 0.3, n_patients)
    Concomitant_MTX = np.random.binomial(1, 0.6, n_patients)
    Corticosteroid_Dose = np.random.exponential(5, n_patients)
    Smoking = np.random.choice([0,1,2], n_patients, p=[0.5,0.3,0.2])

    X = np.column_stack([
        DAS28_CRP, SJC28, TJC28, CRP, ESR,
        PtGA, PhGA, VAS_Pain, HAQ_DI,
        Age, Sex, BMI, Disease_Duration,
        RF_Positive, AntiCCP_Positive,
        Prior_DMARDs, Prior_Biologics,
        MTX_Naive, Concomitant_MTX, Corticosteroid_Dose,
        Smoking
    ])

    response_score = (
        -1.5*(DAS28_CRP - 5.7) +
        -0.8*(Disease_Duration - 5) +
        2.0*(AntiCCP_Positive*(CRP>12.3)) +
        -0.5*Prior_Biologics +
        1.0*MTX_Naive +
        0.5*(CRP<10) +
        np.random.randn(n_patients)*1.5
    )

    thresholds = np.percentile(response_score, [25,50,75])
    y = np.zeros(n_patients, dtype=int)
    y[response_score > thresholds[0]] = 1
    y[response_score > thresholds[1]] = 2
    y[response_score > thresholds[2]] = 3

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# ============================================================================
# NEURAL NETWORK CLASSES
# ============================================================================

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_init='xavier'):
        if weight_init == 'xavier':
            self.weights = np.random.randn(n_inputs, n_neurons)*np.sqrt(1./n_inputs)
        elif weight_init == 'he':
            self.weights = np.random.randn(n_inputs, n_neurons)*np.sqrt(2./n_inputs)
        else:
            self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights)+self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class Loss_CCE_Weighted(Loss):
    def __init__(self, class_weights=None):
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        neg_log_likelihoods = -np.log(correct_confidences)
        if self.class_weights is not None:
            weights = self.class_weights[y_true]
            neg_log_likelihoods *= weights
        return neg_log_likelihoods

def compute_class_weights(y):
    classes = np.unique(y)
    counts = np.bincount(y)
    weights = len(y)/(len(classes)*counts)
    print("Class weights:", weights)
    return weights

# ============================================================================
# TRAINING FUNCTION FOR ONE MODEL
# ============================================================================

def train_one_model(X_train, y_train, n_features, n_classes, class_weights, num_iterations=5000, learning_rate=0.5):

    dense1 = Layer_Dense(n_features, 128, weight_init='he')
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(128, 64, weight_init='he')
    activation2 = Activation_ReLU()
    dense3 = Layer_Dense(64, n_classes, weight_init='xavier')
    activation3 = Activation_Softmax()
    loss_function = Loss_CCE_Weighted(class_weights)

    momentum = 0.9
    v1w = np.zeros_like(dense1.weights)
    v1b = np.zeros_like(dense1.biases)
    v2w = np.zeros_like(dense2.weights)
    v2b = np.zeros_like(dense2.biases)
    v3w = np.zeros_like(dense3.weights)
    v3b = np.zeros_like(dense3.biases)

    for it in range(num_iterations):
        dense1.forward(X_train)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)

        dscores = activation3.output.copy()
        dscores[range(len(X_train)), y_train] -= 1
        dscores /= len(X_train)
        dscores *= class_weights[y_train].reshape(-1,1)

        dW3 = activation2.output.T.dot(dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)
        dh2 = dscores.dot(dense3.weights.T)
        dh2[activation2.output <= 0] = 0
        dW2 = activation1.output.T.dot(dh2)
        db2 = np.sum(dh2, axis=0, keepdims=True)
        dh1 = dh2.dot(dense2.weights.T)
        dh1[activation1.output <= 0] = 0
        dW1 = X_train.T.dot(dh1)
        db1 = np.sum(dh1, axis=0, keepdims=True)

        v1w = momentum*v1w - learning_rate*dW1; dense1.weights += v1w
        v1b = momentum*v1b - learning_rate*db1; dense1.biases += v1b
        v2w = momentum*v2w - learning_rate*dW2; dense2.weights += v2w
        v2b = momentum*v2b - learning_rate*db2; dense2.biases += v2b
        v3w = momentum*v3w - learning_rate*dW3; dense3.weights += v3w
        v3b = momentum*v3b - learning_rate*db3; dense3.biases += v3b

    return dense1, dense2, dense3

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    X, y = generate_synthetic_RA_data(1000)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    print(f"Dataset shape: {X.shape}, Classes: {n_classes}")

    # 5-Fold Stratified Cross-Validation
    print("\nRunning 5-Fold Stratified CV...")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n=== Fold {fold+1} ===")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        class_weights = compute_class_weights(y_tr)

        dense1, dense2, dense3 = train_one_model(X_tr, y_tr, n_features, n_classes, class_weights)

        dense1.forward(X_val)
        a1 = Activation_ReLU(); a1.forward(dense1.output)
        dense2.forward(a1.output)
        a2 = Activation_ReLU(); a2.forward(dense2.output)
        dense3.forward(a2.output)
        a3 = Activation_Softmax(); a3.forward(dense3.output)

        preds = np.argmax(a3.output, axis=1)
        acc = np.mean(preds==y_val)
        fold_accs.append(acc)
        print(f"Fold {fold+1} accuracy: {acc:.4f}")

    print("\n=== CV Summary ===")
    print("Fold accuracies:", [f"{a:.4f}" for a in fold_accs])
    print(f"Mean CV accuracy: {np.mean(fold_accs):.4f}, Std: {np.std(fold_accs):.4f}")

    # Train final model on full train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    class_weights = compute_class_weights(y_train)
    dense1, dense2, dense3 = train_one_model(X_train, y_train, n_features, n_classes, class_weights, num_iterations=3000)

    # Evaluate on test set
    dense1.forward(X_test)
    a1 = Activation_ReLU(); a1.forward(dense1.output)
    dense2.forward(a1.output)
    a2 = Activation_ReLU(); a2.forward(dense2.output)
    dense3.forward(a2.output)
    a3 = Activation_Softmax(); a3.forward(dense3.output)

    test_preds = np.argmax(a3.output, axis=1)
    test_acc = np.mean(test_preds==y_test)
    print(f"\nTest set accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=['Non-responder','ACR20','ACR50','ACR70']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_preds)
    print(cm)

    # Optional: plot confusion matrix
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(4), ['Non-resp','ACR20','ACR50','ACR70'])
    plt.yticks(np.arange(4), ['Non-resp','ACR20','ACR50','ACR70'])
    for i in range(4):
        for j in range(4):
            plt.text(j,i,cm[i,j],ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Save final model
    np.savez("ra_model_kfold.npz",
             dense1_w=dense1.weights, dense1_b=dense1.biases,
             dense2_w=dense2.weights, dense2_b=dense2.biases,
             dense3_w=dense3.weights, dense3_b=dense3.biases)
    print("\nFinal model saved as ra_model_kfold.npz")
