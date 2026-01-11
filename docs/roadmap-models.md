# Model Improvements Roadmap

## Current Model: XGBoost Classifier
- **Type:** Gradient Boosting Decision Tree
- **Target:** 3-class (Home Win, Draw, Away Win)
- **Current Accuracy:** ~50-60% (typical for football prediction)

---

## Recommended Model Enhancements

### 1. Ensemble Model Approach
**Priority:** High  
**Complexity:** Medium  
**Expected Improvement:** +5-10% accuracy

Combine multiple models for more robust predictions.

```python
# Create: models/ensemble_predictor.py
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import numpy as np

def create_ensemble_model():
    """Create ensemble of multiple classifiers"""
    
    # Individual models
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42
    )
    
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    # Voting ensemble (soft voting for probabilities)
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft',
        weights=[2, 1.5, 1, 0.5]  # Higher weight for XGB
    )
    
    return ensemble

# Usage in premier-league-predictions.py
model = create_ensemble_model()
model.fit(X_train, y_train)
```

---

### 2. Neural Network with PyTorch
**Priority:** Medium  
**Complexity:** High  
**Expected Improvement:** +3-7% accuracy

Deep learning approach for complex pattern recognition.

```python
# Create: models/neural_predictor.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

class FootballNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super(FootballNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 3))  # 3 output classes
        layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_neural_model(X_train, y_train, epochs=100, batch_size=32):
    """Train neural network model"""
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y_train.values)
    
    # Create model
    model = FootballNet(input_size=X_scaled.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model, scaler

# Prediction function
def predict_neural(model, scaler, X_new):
    """Make predictions with neural network"""
    model.eval()
    with torch.no_grad():
        X_scaled = scaler.transform(X_new)
        X_tensor = torch.FloatTensor(X_scaled)
        predictions = model(X_tensor)
        return predictions.numpy()
```

**Requirements:**
```bash
pip install torch torchvision
```

---

### 3. Poisson Regression for Goal Prediction
**Priority:** Medium  
**Complexity:** Low  
**Expected Improvement:** Better for goal-based betting

Predict exact scorelines using Poisson distribution.

```python
# Create: models/poisson_predictor.py
from scipy.stats import poisson
import numpy as np
import pandas as pd

def estimate_goals(home_attack, home_defense, away_attack, away_defense, league_avg=1.4):
    """
    Estimate expected goals using team strengths
    
    Args:
        home_attack: Home team attacking strength
        home_defense: Home team defensive strength
        away_attack: Away team attacking strength
        away_defense: Away team defensive strength
        league_avg: League average goals per match
    """
    home_expected = league_avg * home_attack * away_defense
    away_expected = league_avg * away_attack * home_defense
    
    return home_expected, away_expected

def poisson_scoreline_probabilities(home_exp, away_exp, max_goals=5):
    """Calculate probability matrix for all scorelines"""
    
    scoreline_probs = np.zeros((max_goals + 1, max_goals + 1))
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_home = poisson.pmf(home_goals, home_exp)
            prob_away = poisson.pmf(away_goals, away_exp)
            scoreline_probs[home_goals, away_goals] = prob_home * prob_away
    
    return scoreline_probs

def predict_match_outcome(scoreline_probs):
    """Convert scoreline probabilities to match outcome probabilities"""
    
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    rows, cols = scoreline_probs.shape
    
    for home_goals in range(rows):
        for away_goals in range(cols):
            prob = scoreline_probs[home_goals, away_goals]
            
            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob
    
    return home_win_prob, draw_prob, away_win_prob

# Full prediction pipeline
def predict_with_poisson(home_team, away_team, team_stats):
    """
    Predict match using Poisson regression
    
    Args:
        home_team: Home team name
        away_team: Away team name
        team_stats: DataFrame with team statistics
    """
    
    # Get team stats
    home_data = team_stats[team_stats['Team'] == home_team].iloc[0]
    away_data = team_stats[team_stats['Team'] == away_team].iloc[0]
    
    # Calculate expected goals
    home_exp, away_exp = estimate_goals(
        home_attack=home_data['HomeGoalsAve'],
        home_defense=1 / (home_data['AwayGoalsAve'] + 0.1),  # Inverse for defense
        away_attack=away_data['AwayGoalsAve'],
        away_defense=1 / (away_data['HomeGoalsAve'] + 0.1)
    )
    
    # Get scoreline probabilities
    scorelines = poisson_scoreline_probabilities(home_exp, away_exp)
    
    # Convert to match outcome
    probs = predict_match_outcome(scorelines)
    
    return {
        'HomeWinProb': probs[0],
        'DrawProb': probs[1],
        'AwayWinProb': probs[2],
        'ExpectedHomeGoals': home_exp,
        'ExpectedAwayGoals': away_exp,
        'MostLikelyScore': np.unravel_index(scorelines.argmax(), scorelines.shape)
    }
```

---

### 4. Time Series LSTM for Momentum
**Priority:** Low  
**Complexity:** High  
**Expected Improvement:** Captures temporal patterns

```python
# Create: models/lstm_predictor.py
import torch
import torch.nn as nn
import numpy as np

class FootballLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(FootballLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

def prepare_sequence_data(df, sequence_length=5):
    """
    Prepare time series sequences for LSTM
    Each sequence is last N matches for a team
    """
    sequences = []
    labels = []
    
    teams = df['HomeTeam'].unique()
    
    for team in teams:
        team_matches = df[
            (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
        ].sort_values('MatchDate')
        
        for i in range(len(team_matches) - sequence_length):
            seq = team_matches.iloc[i:i+sequence_length]
            target = team_matches.iloc[i+sequence_length]
            
            # Extract features for sequence
            seq_features = extract_match_features(seq, team)
            sequences.append(seq_features)
            
            # Get label
            if target['HomeTeam'] == team:
                label = 0 if target['FullTimeResult'] == 'H' else (1 if target['FullTimeResult'] == 'D' else 2)
            else:
                label = 2 if target['FullTimeResult'] == 'A' else (1 if target['FullTimeResult'] == 'D' else 0)
            
            labels.append(label)
    
    return np.array(sequences), np.array(labels)
```

---

### 5. Hyperparameter Optimization
**Priority:** High  
**Complexity:** Low  
**Expected Improvement:** +2-5% accuracy

```python
# Create: optimize_model.py
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
import numpy as np

def optimize_xgboost(X_train, y_train):
    """Find best hyperparameters for XGBoost"""
    
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb = XGBClassifier(random_state=42)
    
    # Randomized search (faster than grid search)
    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=param_grid,
        n_iter=50,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.3f}")
    
    return random_search.best_estimator_

# Usage
best_model = optimize_xgboost(X_train, y_train)
```

---

## Model Comparison Framework

```python
# Create: compare_models.py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def compare_all_models(X_train, X_test, y_train, y_test):
    """Compare performance of different models"""
    
    models = {
        'XGBoost': XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Ensemble': create_ensemble_model()
    }
    
    results = []
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Predictions': y_pred
        })
    
    # Display comparison
    comparison_df = pd.DataFrame(results)[['Model', 'Accuracy']]
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    return comparison_df
```

---

## Recommended Next Steps

1. **Immediate:** Optimize current XGBoost hyperparameters
2. **Week 1:** Implement ensemble model
3. **Week 2:** Add Poisson regression for goal predictions
4. **Month 2:** Experiment with neural networks
5. **Month 3:** Build model comparison dashboard
