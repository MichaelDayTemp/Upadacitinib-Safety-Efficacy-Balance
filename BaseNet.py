import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

np.random.seed(42)

# ============================================================================
# DATA LOADING AND PREPROCESSING FOR RA TREATMENT RESPONSE
# ============================================================================

def load_and_preprocess_RA_data(csv_path):
    """
    Load RA patient data from CSV and preprocess for neural network
    
    Expected columns:
    - DAS28_CRP: Disease Activity Score
    - SJC28: Swollen Joint Count
    - TJC28: Tender Joint Count
    - CRP: C-Reactive Protein (mg/L)
    - ESR: Erythrocyte Sedimentation Rate
    - PtGA: Patient Global Assessment
    - PhGA: Physician Global Assessment
    - VAS_Pain: Pain score
    - HAQ_DI: Health Assessment Questionnaire
    - Age: Patient age
    - Sex: 0=Male, 1=Female
    - BMI: Body Mass Index
    - Disease_Duration: Years since diagnosis
    - RF_Positive: Rheumatoid Factor (0/1)
    - AntiCCP_Positive: Anti-CCP antibodies (0/1)
    - Prior_DMARDs: Number of prior DMARD failures
    - Prior_Biologics: Number of prior biologic failures
    - MTX_Naive: 1 if MTX naive, 0 otherwise
    - Concomitant_MTX: 1 if on MTX, 0 otherwise
    - Corticosteroid_Dose: mg/day
    - Smoking: 0=Never, 1=Former, 2=Current
    - ACR_Response: Target variable (0=Non-responder, 1=ACR20, 2=ACR50, 3=ACR70)
    """
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Define feature columns (adjust based on actual dataset)
    feature_columns = [
        'DAS28_CRP', 'SJC28', 'TJC28', 'CRP', 'ESR',
        'PtGA', 'PhGA', 'VAS_Pain', 'HAQ_DI',
        'Age', 'Sex', 'BMI', 'Disease_Duration',
        'RF_Positive', 'AntiCCP_Positive',
        'Prior_DMARDs', 'Prior_Biologics',
        'MTX_Naive', 'Concomitant_MTX', 'Corticosteroid_Dose',
        'Smoking'
    ]
    
    # Target variable
    target_column = 'ACR_Response'
    
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Handle missing values with median imputation
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler, imputer, feature_columns


# Alternative: Generate synthetic RA data for testing
def generate_synthetic_RA_data(n_patients=1000, n_features=21):
    """
    Generate synthetic RA patient data for testing the model
    This simulates the distribution of real clinical data with BALANCED classes
    """
    np.random.seed(42)
    
    # Generate features with clinical realistic distributions (generated from claude and template NIH data)
    DAS28_CRP = np.random.normal(5.7, 1.2, n_patients)  # Mean from trials
    SJC28 = np.random.poisson(10, n_patients).astype(float)
    TJC28 = np.random.poisson(15, n_patients).astype(float)
    CRP = np.random.exponential(20, n_patients)  # Skewed distribution
    ESR = np.random.exponential(30, n_patients)
    PtGA = np.random.uniform(40, 80, n_patients)  # VAS 0-100
    PhGA = np.random.uniform(40, 75, n_patients)
    VAS_Pain = np.random.uniform(45, 85, n_patients)
    HAQ_DI = np.random.uniform(1.0, 2.5, n_patients)
    Age = np.random.normal(55, 12, n_patients)
    Sex = np.random.binomial(1, 0.75, n_patients)  # 75% female
    BMI = np.random.normal(27, 5, n_patients)
    Disease_Duration = np.random.exponential(5, n_patients)
    RF_Positive = np.random.binomial(1, 0.7, n_patients)
    AntiCCP_Positive = np.random.binomial(1, 0.65, n_patients)
    Prior_DMARDs = np.random.poisson(2, n_patients).astype(float)
    Prior_Biologics = np.random.poisson(1, n_patients).astype(float)
    MTX_Naive = np.random.binomial(1, 0.3, n_patients)
    Concomitant_MTX = np.random.binomial(1, 0.6, n_patients)
    Corticosteroid_Dose = np.random.exponential(5, n_patients)
    Smoking = np.random.choice([0, 1, 2], n_patients, p=[0.5, 0.3, 0.2])
    
    # Stack all features
    X = np.column_stack([
        DAS28_CRP, SJC28, TJC28, CRP, ESR,
        PtGA, PhGA, VAS_Pain, HAQ_DI,
        Age, Sex, BMI, Disease_Duration,
        RF_Positive, AntiCCP_Positive,
        Prior_DMARDs, Prior_Biologics,
        MTX_Naive, Concomitant_MTX, Corticosteroid_Dose,
        Smoking
    ])
    
    # Generate response labels with STRONGER signal and BALANCED classes
    # Response influenced by baseline disease activity and other factors
    response_score = (
        -1.5 * (DAS28_CRP - 5.7) +  # Standardized, stronger weight
        -0.8 * (Disease_Duration - 5) +  # Standardized
        2.0 * (AntiCCP_Positive * (CRP > 12.3)) +  # Much stronger predictor
        -0.5 * Prior_Biologics +
        1.0 * MTX_Naive +
        0.5 * (CRP < 10) +  # Low inflammation predicts better response
        np.random.randn(n_patients) * 1.5  # Less random noise
    )
    
    # Convert to categorical with BALANCED distribution
    # Use percentiles to ensure balanced classes
    percentiles = [25, 50, 75]
    thresholds = np.percentile(response_score, percentiles)
    
    y = np.zeros(n_patients, dtype=int)
    y[response_score > thresholds[0]] = 1  # ACR20 (top 75%)
    y[response_score > thresholds[1]] = 2  # ACR50 (top 50%)
    y[response_score > thresholds[2]] = 3  # ACR70 (top 25%)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


# ============================================================================
# NEURAL NETWORK CLASSES (unchanged except for Xavier/He initialization)
# ============================================================================

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_init='xavier'):
        # Use Xavier/He initialization instead of tiny random weights
        if weight_init == 'xavier':
            # Xavier initialization for sigmoid/tanh/softmax
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)
        elif weight_init == 'he':
            # He initialization for ReLU
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        else:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        
        self.biases = np.zeros((1, n_neurons))
        self.inputs = n_inputs
        self.neurons = n_neurons
    
    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# ============================================================================
# WEIGHTED LOSS FOR CLASS IMBALANCE
# ============================================================================

class Loss_CCE_Weighted(Loss):
    def __init__(self, class_weights=None):
        self.class_weights = class_weights
    
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        neg_log_likelihoods = -np.log(correct_confidences)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            if len(y_true.shape) == 1:
                weights = self.class_weights[y_true]
            else:
                weights = np.sum(self.class_weights * y_true, axis=1)
            neg_log_likelihoods *= weights
        
        return neg_log_likelihoods


# ============================================================================
# COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================================

def compute_class_weights(y):
    """
    Compute class weights inversely proportional to class frequencies
    """
    classes = np.unique(y)
    class_counts = np.bincount(y)
    total_samples = len(y)
    
    # Inverse frequency weighting
    weights = total_samples / (len(classes) * class_counts)
    
    print("\nClass Distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} samples ({count/total_samples*100:.1f}%), weight: {weights[i]:.3f}")
    
    return weights


# ============================================================================
# MAIN TRAINING PIPELINE FOR RA DATA
# ============================================================================

# Load data (use one of these options)
# Option 1: Load from CSV
# X, y, scaler, imputer, feature_names = load_and_preprocess_RA_data('ra_patient_data.csv')

# Option 2: Generate synthetic data for testing
X, y = generate_synthetic_RA_data(n_patients=1000)
n_features = X.shape[1]
n_classes = len(np.unique(y))  # Should be 4 (0, 1, 2, 3 for ACR responses)

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {n_features}")
print(f"Number of classes: {n_classes}")
print(f"Class distribution: {np.bincount(y)}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize network architecture

dense1 = Layer_Dense(n_features, 128, weight_init='he')  # Input: 21 features, Hidden: 128 neurons
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 64, weight_init='he')  # Additional hidden layer
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, n_classes, weight_init='xavier')  # Output: 4 classes
activation3 = Activation_Softmax()

# Compute class weights to handle imbalance
class_weights = compute_class_weights(y_train)
loss_function = Loss_CCE_Weighted(class_weights=class_weights)

# Training hyperparameters
learning_rate = 0.001  # Higher initial learning rate
num_iterations = 10000  # More iterations
num_examples = X_train.shape[0]
decay_rate = 0.95  # Learning rate decay
decay_step = 1000

# Add momentum for faster convergence
momentum = 0.9
velocity_w1 = np.zeros_like(dense1.weights)
velocity_b1 = np.zeros_like(dense1.biases)
velocity_w2 = np.zeros_like(dense2.weights)
velocity_b2 = np.zeros_like(dense2.biases)
velocity_w3 = np.zeros_like(dense3.weights)
velocity_b3 = np.zeros_like(dense3.biases)

# Tracking best model
best_loss = float('inf')
best_accuracy = 0
best_weights = None

print("\nStarting training...")
print("=" * 60)

# Training loop
for iteration in range(num_iterations):
    # Learning rate decay
    if iteration % decay_step == 0 and iteration > 0:
        learning_rate *= decay_rate
    
    # Forward pass
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    # Calculate loss
    loss = loss_function.calculate(activation3.output, y_train)
    
    # Calculate accuracy
    predictions = np.argmax(activation3.output, axis=1)
    accuracy = np.mean(predictions == y_train)
    
    # Calculate per-class accuracy to detect "stuck" behavior
    class_accuracies = []
    for c in range(n_classes):
        mask = y_train == c
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == c)
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    # Print progress
    if iteration % 500 == 0:
        print(f"Iteration {iteration:5d} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | LR: {learning_rate:.6f}")
        print(f"  Per-class accuracy: {[f'{acc:.3f}' for acc in class_accuracies]}")
        print(f"  Predictions: {np.bincount(predictions, minlength=n_classes)}")
    
    # Save best model
    if accuracy > best_accuracy:  # Changed from loss to accuracy
        best_loss = loss
        best_accuracy = accuracy
        best_iteration = iteration
        best_weights = {
            'dense1_w': dense1.weights.copy(),
            'dense1_b': dense1.biases.copy(),
            'dense2_w': dense2.weights.copy(),
            'dense2_b': dense2.biases.copy(),
            'dense3_w': dense3.weights.copy(),
            'dense3_b': dense3.biases.copy()
        }
    
    # Backpropagation
    dscores = activation3.output.copy()
    dscores[range(num_examples), y_train] -= 1
    dscores /= num_examples
    
    # Apply class weights to gradients
    if class_weights is not None:
        weight_matrix = class_weights[y_train].reshape(-1, 1)
        dscores *= weight_matrix
    
    # Gradient for output layer
    dW3 = np.dot(activation2.output.T, dscores)
    db3 = np.sum(dscores, axis=0, keepdims=True)
    
    # Add L2 regularization
    reg_lambda = 0.001
    dW3 += reg_lambda * dense3.weights
    
    # Backprop into second hidden layer
    dhidden2 = np.dot(dscores, dense3.weights.T)
    dhidden2[activation2.output <= 0] = 0  # ReLU gradient
    
    # Gradient for second hidden layer
    dW2 = np.dot(activation1.output.T, dhidden2)
    db2 = np.sum(dhidden2, axis=0, keepdims=True)
    dW2 += reg_lambda * dense2.weights
    
    # Backprop into first hidden layer
    dhidden1 = np.dot(dhidden2, dense2.weights.T)
    dhidden1[activation1.output <= 0] = 0  # ReLU gradient
    
    # Gradient for first hidden layer
    dW1 = np.dot(X_train.T, dhidden1)
    db1 = np.sum(dhidden1, axis=0, keepdims=True)
    dW1 += reg_lambda * dense1.weights
    
    # Update weights with momentum
    velocity_w1 = momentum * velocity_w1 - learning_rate * dW1
    velocity_b1 = momentum * velocity_b1 - learning_rate * db1
    velocity_w2 = momentum * velocity_w2 - learning_rate * dW2
    velocity_b2 = momentum * velocity_b2 - learning_rate * db2
    velocity_w3 = momentum * velocity_w3 - learning_rate * dW3
    velocity_b3 = momentum * velocity_b3 - learning_rate * db3
    
    dense1.weights += velocity_w1
    dense1.biases += velocity_b1
    dense2.weights += velocity_w2
    dense2.biases += velocity_b2
    dense3.weights += velocity_w3
    dense3.biases += velocity_b3

print("=" * 60)
print(f"\nTraining completed!")
print(f"Best loss: {best_loss:.4f} at iteration {best_iteration}")
print(f"Best training accuracy: {best_accuracy:.4f}")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

# Load best weights
dense1.weights = best_weights['dense1_w']
dense1.biases = best_weights['dense1_b']
dense2.weights = best_weights['dense2_w']
dense2.biases = best_weights['dense2_b']
dense3.weights = best_weights['dense3_w']
dense3.biases = best_weights['dense3_b']

# Evaluate on test set
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

test_predictions = np.argmax(activation3.output, axis=1)
test_accuracy = np.mean(test_predictions == y_test)

print(f"\nTest Set Accuracy: {test_accuracy:.4f}")

# ============================================================================
# DETAILED EVALUATION METRICS
# ============================================================================

from sklearn.metrics import classification_report, confusion_matrix

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_test, 
    test_predictions,
    target_names=['Non-responder', 'ACR20', 'ACR50', 'ACR70']
))

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = confusion_matrix(y_test, test_predictions)
print(cm)
print("\nRows = Actual, Columns = Predicted")
print("Classes: 0=Non-responder, 1=ACR20, 2=ACR50, 3=ACR70")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot training history (if you track it)
# Since we don't track history in the loop, this is optional

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - RA Treatment Response Prediction')
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, ['Non-resp', 'ACR20', 'ACR50', 'ACR70'])
plt.yticks(tick_marks, ['Non-resp', 'ACR20', 'ACR50', 'ACR70'])

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ============================================================================
# SAVE MODEL FOR DEPLOYMENT
# ============================================================================

def save_model(filename='ra_response_model.npz'):
    """Save trained model weights and biases"""
    np.savez(
        filename,
        dense1_weights=dense1.weights,
        dense1_biases=dense1.biases,
        dense2_weights=dense2.weights,
        dense2_biases=dense2.biases,
        dense3_weights=dense3.weights,
        dense3_biases=dense3.biases
    )
    print(f"\nModel saved to {filename}")

def load_model(filename='ra_response_model.npz'):
    """Load trained model weights and biases"""
    data = np.load(filename)
    
    # Recreate layers with loaded weights
    n_features = data['dense1_weights'].shape[0]
    hidden1_size = data['dense1_weights'].shape[1]
    hidden2_size = data['dense2_weights'].shape[1]
    n_classes = data['dense3_weights'].shape[1]
    
    dense1 = Layer_Dense(n_features, hidden1_size)
    dense1.weights = data['dense1_weights']
    dense1.biases = data['dense1_biases']
    
    dense2 = Layer_Dense(hidden1_size, hidden2_size)
    dense2.weights = data['dense2_weights']
    dense2.biases = data['dense2_biases']
    
    dense3 = Layer_Dense(hidden2_size, n_classes)
    dense3.weights = data['dense3_weights']
    dense3.biases = data['dense3_biases']
    
    print(f"Model loaded from {filename}")
    return dense1, dense2, dense3

# Save the trained model
save_model()

print("\n" + "=" * 60)
print("Training pipeline complete!")
print("=" * 60)
