"""
Train XGBoost model for football match prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(filepath='../data/processed/final_dataset_with_features.csv'):
    """
    Load the dataset with features
    """
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(filepath)
    
    print(f"✅ Loaded {len(df)} matches")
    print(f"✅ Total columns: {len(df.columns)}")
    
    return df


def prepare_data(df):
    """
    Prepare data for XGBoost training
    """
    print("\n" + "="*60)
    print("PREPARING DATA FOR TRAINING")
    print("="*60)
    
    # Select feature columns
    feature_cols = [
        'home_form_L5',
        'home_avg_goals_scored_L5',
        'home_avg_goals_conceded_L5',
        'home_goal_diff_L5',
        'home_avg_shots_L5',
        'home_avg_shots_on_target_L5',
        'home_shot_accuracy_L5',
        'home_avg_corners_L5',
        'home_h2h_win_rate_L3',
        'away_form_L5',
        'away_avg_goals_scored_L5',
        'away_avg_goals_conceded_L5',
        'away_goal_diff_L5',
        'away_avg_shots_L5',
        'away_avg_shots_on_target_L5',
        'away_shot_accuracy_L5',
        'away_avg_corners_L5',
        'away_h2h_win_rate_L3',
    ]
    
    print(f"\n📋 Features to use: {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i}. {col}")
    
    # Remove rows without sufficient history
    df_model = df[df['home_form_L5'] > 0].copy()
    
    print(f"\n📊 Matches with sufficient history: {len(df_model)}")
    print(f"📊 Matches removed (insufficient history): {len(df) - len(df_model)}")
    
    # Features and target
    X = df_model[feature_cols]
    y = df_model['FTR']
    
    # Fill any remaining missing values
    if X.isnull().sum().sum() > 0:
        print("\n⚠️  Warning: Found missing values. Filling with 0...")
        X = X.fillna(0)
    
    # XGBoost requires numeric labels (0, 1, 2)
    # Map: A=0, D=1, H=2
    label_map = {'A': 0, 'D': 1, 'H': 2}
    y_numeric = y.map(label_map)
    
    print(f"\n🔢 Converting labels to numeric:")
    print(f"   A (Away Win) → 0")
    print(f"   D (Draw) → 1")
    print(f"   H (Home Win) → 2")
    
    # Time-based split (80% train, 20% test)
    split_index = int(len(X) * 0.8)
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y_numeric[:split_index]
    y_test = y_numeric[split_index:]
    y_test_original = y[split_index:]  # Keep original labels for display
    
    print(f"\n📊 Training set: {len(X_train)} matches ({len(X_train)/len(X)*100:.1f}%)")
    print(f"📊 Test set: {len(X_test)} matches ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n📈 Target distribution in training set:")
    train_dist = y_train.value_counts(normalize=True).sort_index()
    for label, pct in train_dist.items():
        outcome_name = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}.get(label)
        print(f"   {outcome_name}: {pct:.1%}")
    
    return X_train, X_test, y_train, y_test, y_test_original, feature_cols


def train_xgboost(X_train, y_train, X_test, y_test):

    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    
    print("\n🚀 XGBoost Configuration:")
    print("   - Objective: multi:softmax (3-class classification)")
    print("   - Number of classes: 3")
    print("   - Max depth: 3")
    print("   - Learning rate: 0.1")
    print("   - Number of estimators: 200")
    print("   - Min child weight: 5")
    print("   - Subsample: 0.8")
    print("   - Colsample bytree: 0.8")
    print("   - Gamma: 1")
    print("   - Reg alpha: 0.1 (L1 regularization)")
    print("   - Reg lambda: 1 (L2 regularization)")
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train)
    total = len(y_train)

    print(f"\n📊 Class distribution in training set:")
    for label, count in enumerate(class_counts):
        outcome_name = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}.get(label)
        pct = count / total * 100
        print(f"   {outcome_name}: {count} ({pct:.1f}%)")
    
    # Calculate CONSERVATIVE weights using square root
    class_weights = {}
    for i in range(3):  # 3 classes: 0=Away, 1=Draw, 2=Home
        # Use square root for more conservative weighting
        weight = np.sqrt(total / (3 * class_counts[i]))
        class_weights[i] = weight
    
    # 🆕 MODERATE boost for draws (only 10% increase)
    class_weights[1] = class_weights[1] * 1.28
    
    print(f"\n⚖️  Class weights (conservative with sqrt):")
    for label, weight in class_weights.items():
        outcome_name = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}.get(label)
        boost_marker = " 🚀 +10% boost" if label == 1 else ""
        print(f"   {outcome_name}: {weight:.3f}{boost_marker}")
    # Convert to sample weights
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    # Initialize XGBoost Classifier
    model = xgb.XGBClassifier(
            objective='multi:softmax',    # Multi-class classification
            num_class=3,                   # 3 outcomes: A, D, H
            max_depth=3,                   # Shallow trees (prevent overfitting)
            learning_rate=0.01,             # Step size
            n_estimators=300,              # Number of trees
            min_child_weight=6,            # Minimum sum of weights in a child
            subsample=0.8,                 # Fraction of samples per tree
            colsample_bytree=0.8,          # Fraction of features per tree
            gamma=0.3,                       # Minimum loss reduction for split
            reg_alpha=0.3,                 # L1 regularization
            reg_lambda=1.2,                  # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
    
    print("\n🔄 Training the model...")
    print("   This may take 1-2 minutes...\n")
    
    # Train with sample weights and evaluation
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    print("✅ Training complete!\n")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, y_test_original, feature_names):

    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    gap = train_acc - test_acc
    
    print(f"\n📊 ACCURACY SCORES:")
    print(f"   Training Accuracy: {train_acc:.1%}")
    print(f"   Test Accuracy: {test_acc:.1%}")
    print(f"   Gap: {gap:.1%}")
    
    if gap < 0.05:
        print(f"\n✅ Excellent! Gap < 5% - minimal overfitting")
    elif gap < 0.10:
        print(f"\n✅ Good! Gap < 10% - acceptable overfitting")
    elif gap < 0.15:
        print(f"\n⚠️  Warning: Gap 10-15% - some overfitting")
    else:
        print(f"\n❌ Bad: Gap > 15% - severe overfitting")
    
    # Classification report (convert back to original labels for display)
    y_test_labels = y_test_original
    y_pred_labels = pd.Series(y_pred_test).map({0: 'A', 1: 'D', 2: 'H'})
    
    print(f"\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("="*60)
    
    target_names = ['Away Win', 'Draw', 'Home Win']
    print(classification_report(y_test_labels, y_pred_labels, target_names=target_names))
    
    # Confusion matrix
    print("="*60)
    print("CONFUSION MATRIX (Test Set)")
    print("="*60)
    
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=['A', 'D', 'H'])
    
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              A    D    H")
    print("Actual  A  ", cm[0])
    print("        D  ", cm[1])
    print("        H  ", cm[2])
    
    print("\nInterpretation:")
    print("  - Diagonal values (A-A, D-D, H-H) = Correct predictions")
    print("  - Off-diagonal values = Incorrect predictions")
    
    # Calculate recall for each class
    print("\n📊 Recall by outcome:")
    for i, outcome in enumerate(['Away Win', 'Draw', 'Home Win']):
        recall = cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"   {outcome}: {recall:.1%}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    # XGBoost provides feature importance scores
    importance_scores = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    # Prediction distribution
    print("\n" + "="*60)
    print("PREDICTION BREAKDOWN (Test Set)")
    print("="*60)
    
    pred_dist = y_pred_labels.value_counts(normalize=True).sort_index()
    actual_dist = y_test_labels.value_counts(normalize=True).sort_index()
    
    print("\n                 Predicted    Actual    Difference")
    for outcome in ['A', 'D', 'H']:
        outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}.get(outcome)
        pred_pct = pred_dist.get(outcome, 0)
        actual_pct = actual_dist.get(outcome, 0)
        diff = pred_pct - actual_pct
        print(f"   {outcome_name:12s}  {pred_pct:6.1%}      {actual_pct:6.1%}    {diff:+6.1%}")
    
    return feature_importance, y_pred_labels


def save_model(model, feature_names, filepath='../models/xgboost_model.pkl'):
    """
    Save the trained XGBoost model
    """
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'model_type': 'XGBoostClassifier',
        'n_features': len(feature_names),
        'label_map': {'A': 0, 'D': 1, 'H': 2},
        'reverse_label_map': {0: 'A', 1: 'D', 2: 'H'}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    file_size = os.path.getsize(filepath) / 1024
    
    print(f"✅ Model saved to: {filepath}")
    print(f"✅ File size: {file_size:.2f} KB")
    print(f"✅ Model type: XGBoost Classifier")
    print(f"✅ Number of features: {len(feature_names)}")


def plot_feature_importance(feature_importance, top_n=10):

    print("\n" + "="*60)
    print("GENERATING FEATURE IMPORTANCE PLOT")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    
    top_features = feature_importance.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features (XGBoost)')
    plt.gca().invert_yaxis()
    
    os.makedirs('models', exist_ok=True)
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
    
    print("✅ Feature importance plot saved to: models/feature_importance.png")
    
    plt.close()


def plot_confusion_matrix(y_test, y_pred):

    print("\n" + "="*60)
    print("GENERATING CONFUSION MATRIX PLOT")
    print("="*60)
    
    cm = confusion_matrix(y_test, y_pred, labels=['A', 'D', 'H'])
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Away Win', 'Draw', 'Home Win'],
                yticklabels=['Away Win', 'Draw', 'Home Win'])
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - XGBoost Model')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    print("✅ Confusion matrix plot saved to: models/confusion_matrix.png")
    
    plt.close()


def plot_training_history(model):

    try:
        results = model.evals_result()
        
        if results:
            print("\n" + "="*60)
            print("GENERATING TRAINING HISTORY PLOT")
            print("="*60)
            
            epochs = len(results['validation_0']['mlogloss'])
            x_axis = range(0, epochs)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
            ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
            ax.legend()
            ax.set_ylabel('Log Loss')
            ax.set_xlabel('Iteration')
            ax.set_title('XGBoost Training History')
            
            plt.tight_layout()
            plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
            
            print("✅ Training history plot saved to: models/training_history.png")
            plt.close()
    except:
        pass


def main():

    print("\n" + "="*60)
    print("FOOTBALL MATCH PREDICTION - XGBOOST TRAINING")
    print("="*60)
    print("\n🚀 Using: XGBoost Classifier")
    print("🎯 Task: Predict match outcome (Home Win / Draw / Away Win)\n")
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, y_test_original, feature_names = prepare_data(df)
    
    # Step 3: Train XGBoost model
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Step 4: Evaluate model
    feature_importance, y_pred_labels = evaluate_model(
        model, X_train, y_train, X_test, y_test, y_test_original, feature_names
    )
    
    # Step 5: Save model
    save_model(model, feature_names)
    
    # Step 6: Create visualizations
    plot_feature_importance(feature_importance, top_n=10)
    plot_confusion_matrix(y_test_original, y_pred_labels)
    plot_training_history(model)
    
    # Final summary
    test_acc = accuracy_score(y_test_original, y_pred_labels)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    print(f"\n📊 Final Test Accuracy: {test_acc:.1%}")
    print(f"📁 Model saved to: models/xgboost_model.pkl")
    print(f"📊 Plots saved to: models/")
    
    print("\n🎉 Your XGBoost model is ready to make predictions!")
    print("\nNext steps:")
    print("   1. Run predictions: python src/predict.py")
    print("   2. Launch Streamlit app: streamlit run app.py")
    
    # Performance assessment
    print("\n" + "="*60)
    print("PERFORMANCE ASSESSMENT")
    print("="*60)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    gap = train_acc - test_acc
    
    if test_acc >= 0.50 and gap < 0.05:
        print("🏆 EXCELLENT! Your model meets professional standards!")
    elif test_acc >= 0.48 and gap < 0.10:
        print("✅ GOOD! Your model performs well!")
    elif test_acc >= 0.45 and gap < 0.15:
        print("⚠️  ACCEPTABLE. Consider tuning hyperparameters.")
    else:
        print("❌ NEEDS IMPROVEMENT. Try adjusting hyperparameters.")
    
    print(f"\nBenchmarks:")
    print(f"   Random guessing: 33.3%")
    print(f"   Your model: {test_acc:.1%}")
    print(f"   Professional models: 50-55%")


if __name__ == "__main__":
    main()
