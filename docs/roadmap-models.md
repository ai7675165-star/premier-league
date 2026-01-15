# Model Improvements Roadmap

## Current Model: XGBoost Classifier
- **Type:** Gradient Boosting Decision Tree
- **Target:** 3-class (Home Win, Draw, Away Win)
- **Current Accuracy:** ~50-60% (typical for football prediction)

---

## Recommended Model Enhancements

### 1. Ensemble Model Approach ✅ **COMPLETED**
**Priority:** High  
**Complexity:** Medium  
**Expected Improvement:** +5-10% accuracy  
**Actual Improvement:** +3.5% accuracy, -0.038 MAE

Combine multiple models for more robust predictions. Implemented as VotingClassifier with XGBoost, Random Forest, Gradient Boosting, and Logistic Regression using soft voting with weighted probabilities.

**Features:**
- Pre-trained nightly via automated pipeline
- Session state persistence for immediate availability
- Simple ensemble for fast loading vs full ensemble for accuracy

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

### 2. Neural Network with PyTorch ✅ **COMPLETED**
**Priority:** Medium  
**Complexity:** High  
**Expected Improvement:** +3-7% accuracy  
**Actual Improvement:** +4.9% accuracy vs XGBoost baseline

Deep learning approach using PyTorch with 3-layer neural network (128→64→32 neurons), batch normalization, and dropout regularization. Successfully implemented and integrated into the model comparison framework with UI button activation.

**Features:**
- Pre-trained nightly via automated pipeline
- Session state persistence of trained models
- 50 epochs with batch normalization and dropout
- On-demand retraining available via UI button
- Automatic integration with model comparison dashboard

**Architecture:**
- Input layer → 128 neurons → 64 neurons → 32 neurons → 3 outputs
- ReLU activation, batch normalization, 30% dropout
- Cross-entropy loss with Adam optimizer

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

### 5. Hyperparameter Optimization ✅ **COMPLETED**
**Priority:** High  
**Complexity:** Low  
**Expected Improvement:** +2-5% accuracy  
**Actual Improvement:** +0.87% accuracy, -0.023 MAE

Implemented RandomizedSearchCV for XGBoost hyperparameter optimization with UI button integration. Users can now trigger expensive hyperparameter optimization on-demand rather than on every app startup, improving performance while maintaining access to advanced features.

**Features:**
- Pre-trained nightly via automated pipeline
- On-demand re-optimization available via UI button
- Session state persistence of optimization results
- Reduced search space (10 iterations × 3-fold CV) for reasonable runtime
- Real-time progress indicators and status updates
- Integration with model comparison dashboard

**Best Parameters Found:**
- `subsample`: 0.8
- `n_estimators`: 100  
- `min_child_weight`: 5
- `max_depth`: 3
- `learning_rate`: 0.1
- `gamma`: 0
- `colsample_bytree`: 0.8

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

1. ✅ **COMPLETED:** Implement ensemble model (+3.5% accuracy improvement)
2. ✅ **COMPLETED:** Experiment with neural networks (+4.9% accuracy vs XGBoost baseline) - Now with UI button activation
3. ✅ **COMPLETED:** Optimize current XGBoost hyperparameters (+0.87% accuracy improvement) - Now with UI button activation
4. **Week 1:** Add Poisson regression for goal predictions
5. **Month 1:** Build comprehensive model comparison dashboard
