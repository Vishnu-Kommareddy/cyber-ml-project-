import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# Define standard NSL-KDD column mappings
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
           "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "target", "difficulty_level"]

# Look for the local file path
local_path = "data/KDDTrain+.txt"

if not os.path.exists(local_path):
    print(f"❌ Error: Could not find the dataset file at '{local_path}'.")
    print("Please ensure you downloaded the text file and placed it in a 'data' folder inside your project directory.")
else:
    print("🛰️ Step 1: Loading local network intrusion dataset...")
    df = pd.read_csv(local_path, names=columns, header=None)
    print(f"Dataset loaded successfully! Shape: {df.shape}")

    print("\n🧹 Step 2: Preprocessing data...")
    # Target is 'normal' or a specific attack type. Convert to binary anomaly detection flag.
    df['label'] = df['target'].apply(lambda x: 0 if x == 'normal' else 1)

    # Separate features and target label, dropping string identifier columns
    X = df.drop(['target', 'difficulty_level', 'label'], axis=1)
    y = df['label']

    # One-hot encode features like protocol type ('tcp', 'udp', 'icmp')
    X = pd.get_dummies(X)

    # Split into train/test groups
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🤖 Step 3: Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n📊 Step 4: Evaluating Model Performance...")
    predictions = model.predict(X_test)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, predictions))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions))