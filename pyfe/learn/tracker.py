"""
Experiment tracking and database management for ML experiments.

Stores results in SQLite database for later analysis and comparison.
"""

from typing import Dict, List, Optional, Any
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd


class ExperimentTracker:
    """
    Track ML experiments in SQLite database.
    
    Stores:
    - Experiment metadata
    - Model configurations
    - Performance metrics
    - Feature selections
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        experiment_name: str = 'experiment'
    ):
        """
        Initialize tracker.
        
        Args:
            db_path: Path to SQLite database (default: ./experiments.db)
            experiment_name: Name for this experiment
        """
        if db_path is None:
            db_path = './experiments.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.experiment_id = None
        
        # Initialize database
        self._init_db()
        
        # Create experiment record
        self._create_experiment()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                selector TEXT,
                feature_count INTEGER,
                estimator TEXT,
                problem_type TEXT,
                tuned BOOLEAN,
                params TEXT,
                metrics TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _create_experiment(self):
        """Create experiment record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiments (name, created_at, metadata)
            VALUES (?, ?, ?)
        """, (
            self.experiment_name,
            datetime.now().isoformat(),
            json.dumps({})
        ))
        
        self.experiment_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
    
    def log_result(self, result: Dict[str, Any]):
        """
        Log a single result.
        
        Args:
            result: Dictionary with result data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract metrics
        metrics = {}
        for key, value in result.items():
            if key.startswith('test_') or key.startswith('train_'):
                metrics[key] = value
        
        cursor.execute("""
            INSERT INTO results (
                experiment_id, selector, feature_count, estimator,
                problem_type, tuned, params, metrics, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.experiment_id,
            result.get('selector'),
            result.get('k'),
            result.get('estimator'),
            result.get('problem_type'),
            result.get('tuned', False),
            json.dumps(result.get('params', {})),
            json.dumps(metrics),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_results(
        self,
        experiment_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get results as DataFrame.
        
        Args:
            experiment_id: Specific experiment ID (None = current experiment)
            
        Returns:
            DataFrame with results
        """
        if experiment_id is None:
            experiment_id = self.experiment_id
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM results
            WHERE experiment_id = ?
            ORDER BY result_id
        """
        
        df = pd.read_sql_query(query, conn, params=(experiment_id,))
        conn.close()
        
        # Parse JSON fields
        if not df.empty:
            df['params'] = df['params'].apply(json.loads)
            df['metrics'] = df['metrics'].apply(json.loads)
            
            # Expand metrics into columns
            metrics_df = pd.json_normalize(df['metrics'])
            df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
        
        return df
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments in database."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT 
                e.experiment_id,
                e.name,
                e.created_at,
                COUNT(r.result_id) as n_results
            FROM experiments e
            LEFT JOIN results r ON e.experiment_id = r.experiment_id
            GROUP BY e.experiment_id
            ORDER BY e.created_at DESC
        """, conn)
        conn.close()
        
        return df
    
    def get_best_result(
        self,
        metric: str = 'test_accuracy_mean',
        experiment_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get best result by metric.
        
        Args:
            metric: Metric to optimize
            experiment_id: Experiment ID (None = current)
            
        Returns:
            Dictionary with best result
        """
        df = self.get_results(experiment_id)
        
        if df.empty:
            raise ValueError("No results found")
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        best_idx = df[metric].idxmax()
        return df.loc[best_idx].to_dict()
    
    def export_to_csv(
        self,
        output_path: str,
        experiment_id: Optional[int] = None
    ):
        """
        Export results to CSV.
        
        Args:
            output_path: Path to save CSV
            experiment_id: Experiment ID (None = current)
        """
        df = self.get_results(experiment_id)
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")


def init_experiment_db(db_path: str = './experiments.db'):
    """
    Initialize experiment database.
    
    Args:
        db_path: Path to database file
    """
    tracker = ExperimentTracker(db_path=db_path, experiment_name='init')
    print(f"Database initialized at: {db_path}")


def load_experiment_results(
    db_path: str = './experiments.db',
    experiment_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Load results from database.
    
    Args:
        db_path: Path to database
        experiment_id: Specific experiment ID (None = latest)
        
    Returns:
        DataFrame with results
    """
    tracker = ExperimentTracker(db_path=db_path, experiment_name='temp')
    
    if experiment_id is None:
        # Get latest experiment
        experiments = tracker.list_experiments()
        if experiments.empty:
            raise ValueError("No experiments found in database")
        experiment_id = experiments.iloc[0]['experiment_id']
    
    return tracker.get_results(experiment_id)


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker(
        db_path='./test_experiments.db',
        experiment_name='test_run'
    )
    
    # Log some dummy results
    for i in range(3):
        result = {
            'selector': 'anova',
            'k': 10 * (i + 1),
            'estimator': 'RandomForest',
            'problem_type': 'classification',
            'tuned': True,
            'params': {'n_estimators': 100},
            'test_accuracy_mean': 0.85 + i * 0.02,
            'test_accuracy_std': 0.03,
            'test_f1_mean': 0.83 + i * 0.02,
            'test_f1_std': 0.04,
        }
        tracker.log_result(result)
    
    # Get results
    results = tracker.get_results()
    print("\nResults:")
    print(results[['selector', 'feature_count', 'estimator', 'test_accuracy_mean']])
    
    # Get best
    best = tracker.get_best_result('test_accuracy_mean')
    print(f"\nBest Result:")
    print(f"  Selector: {best['selector']}")
    print(f"  Features: {best['feature_count']}")
    print(f"  Estimator: {best['estimator']}")
    print(f"  Accuracy: {best['test_accuracy_mean']:.4f}")
