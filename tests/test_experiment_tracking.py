"""Tests for experiment tracking and model versioning."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from training.utils.experiment_tracking import (
    ExperimentTracker,
    MetricsRecord,
    RunMetadata,
    get_git_info,
    set_global_seed,
)


class TestSetGlobalSeed:
    """Tests for seed setting."""
    
    def test_set_seed_numpy(self) -> None:
        """Test that numpy seed is set."""
        import numpy as np
        
        set_global_seed(42)
        
        # Generate random numbers
        a = np.random.rand(5)
        
        # Reset seed
        set_global_seed(42)
        
        # Should get same numbers
        b = np.random.rand(5)
        
        assert np.allclose(a, b)
    
    def test_set_seed_random(self) -> None:
        """Test that random module seed is set."""
        import random
        
        set_global_seed(42)
        a = [random.random() for _ in range(5)]
        
        set_global_seed(42)
        b = [random.random() for _ in range(5)]
        
        assert a == b


class TestRunMetadata:
    """Tests for RunMetadata."""
    
    def test_create_generates_id(self) -> None:
        """Test that create generates unique run ID."""
        meta1 = RunMetadata.create()
        meta2 = RunMetadata.create()
        
        assert meta1.run_id != meta2.run_id
    
    def test_create_with_name(self) -> None:
        """Test creating metadata with custom name."""
        meta = RunMetadata.create(run_name="my_experiment")
        
        assert meta.run_name == "my_experiment"
    
    def test_complete(self) -> None:
        """Test completing a run."""
        meta = RunMetadata.create()
        assert meta.status == "running"
        assert meta.end_time is None
        
        meta.complete()
        
        assert meta.status == "completed"
        assert meta.end_time is not None
    
    def test_fail(self) -> None:
        """Test failing a run."""
        meta = RunMetadata.create()
        
        meta.fail("Test error")
        
        assert meta.status == "failed"
        assert meta.end_time is not None
        assert meta.environment["error"] == "Test error"


class TestMetricsRecord:
    """Tests for MetricsRecord."""
    
    def test_to_dict_excludes_none(self) -> None:
        """Test that to_dict excludes None values."""
        record = MetricsRecord(
            epoch=1,
            train_loss=2.5,
            val_loss=None,  # Should be excluded
        )
        
        result = record.to_dict()
        
        assert "epoch" in result
        assert "train_loss" in result
        assert "val_loss" not in result
    
    def test_to_dict_flattens_additional(self) -> None:
        """Test that additional metrics are flattened."""
        record = MetricsRecord(
            epoch=1,
            train_loss=2.5,
            additional={"custom_metric": 0.95},
        )
        
        result = record.to_dict()
        
        assert "custom_metric" in result
        assert result["custom_metric"] == 0.95
        assert "additional" not in result


class TestExperimentTracker:
    """Tests for ExperimentTracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                output_dir=tmpdir,
                run_name="test_run",
                seed=42,
            )
            yield tracker
    
    def test_initialization_creates_directories(self) -> None:
        """Test that tracker creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            assert (Path(tmpdir) / "logs").exists()
            assert (Path(tmpdir) / "checkpoints").exists()
            assert (Path(tmpdir) / "artifacts").exists()
    
    def test_log_config(self) -> None:
        """Test logging configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            config = {"batch_size": 32, "epochs": 10}
            tracker.log_config(config)
            
            # Check file was created
            config_path = Path(tmpdir) / "config_snapshot.json"
            assert config_path.exists()
            
            # Check content
            with open(config_path) as f:
                saved = json.load(f)
            
            assert saved["batch_size"] == 32
            assert "_config_hash" in saved
    
    def test_log_metrics(self) -> None:
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            tracker.log_metrics(epoch=1, train_loss=2.5, val_loss=2.8)
            tracker.log_metrics(epoch=2, train_loss=2.2, val_loss=2.5)
            
            # Check CSV file
            csv_path = Path(tmpdir) / "metrics.csv"
            assert csv_path.exists()
            
            content = csv_path.read_text()
            assert "epoch" in content
            assert "train_loss" in content
    
    def test_log_bleu_scores(self) -> None:
        """Test logging BLEU scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            tracker.log_bleu_scores(
                bleu1=0.65, bleu2=0.45, bleu3=0.35, bleu4=0.25
            )
            
            summary = tracker.get_summary()
            
            assert summary["bleu_scores"]["test_bleu1"] == 0.65
            assert summary["bleu_scores"]["test_bleu4"] == 0.25
    
    def test_finalize(self) -> None:
        """Test finalizing experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            tracker.log_config({"test": True})
            tracker.log_metrics(epoch=1, train_loss=2.5)
            
            summary_path = tracker.finalize()
            
            assert Path(summary_path).exists()
            
            with open(summary_path) as f:
                summary = json.load(f)
            
            assert summary["metadata"]["status"] == "completed"
    
    def test_finalize_with_error(self) -> None:
        """Test finalizing experiment with error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            summary_path = tracker.finalize(error="Test error")
            
            with open(summary_path) as f:
                summary = json.load(f)
            
            assert summary["metadata"]["status"] == "failed"
    
    def test_log_artifact_json(self) -> None:
        """Test logging JSON artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            
            data = {"key": "value", "number": 42}
            path = tracker.log_artifact("test_data", data, artifact_type="json")
            
            assert path.exists()
            assert path.suffix == ".json"
    
    def test_properties(self) -> None:
        """Test tracker properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir, run_name="test")
            
            assert tracker.run_id is not None
            assert tracker.output_dir == Path(tmpdir)
            assert tracker.logs_dir == Path(tmpdir) / "logs"
            assert tracker.checkpoints_dir == Path(tmpdir) / "checkpoints"
            assert tracker.artifacts_dir == Path(tmpdir) / "artifacts"
