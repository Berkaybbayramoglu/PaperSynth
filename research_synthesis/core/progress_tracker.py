"""Thread-safe JSON-backed progress logging across pipeline stages."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ProgressTracker:
    """Append-only progress log written to disk after each event."""

    def __init__(self, log_file: str = "synthesis_progress.json"):
        self.log_file = Path(log_file)
        self.lock = asyncio.Lock()
        self.progress: Dict[str, Any] = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "expected_papers": 0,
            "received_papers": 0,
            "stages": [],
            "papers": [],
            "metrics": {},
            "order_verifications": [],
        }

    def __del__(self):
        try:
            if hasattr(self, "progress") and self.progress:
                self._save_progress()
        except Exception:
            pass

    async def log_stage(self, stage_name: str, status: str, details: Optional[Dict] = None) -> None:
        async with self.lock:
            self.progress["stages"].append({
                "stage": stage_name,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details or {},
            })
            self._save_progress()
            print(f"Progress Tracker: {stage_name} - {status}")

    async def log_paper(self, paper_title: str, status: str, details: Optional[Dict] = None) -> None:
        async with self.lock:
            self.progress["papers"].append({
                "paper": paper_title,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "details": details or {},
            })
            self._save_progress()

    async def log_metric(self, metric_name: str, value: Any) -> None:
        async with self.lock:
            self.progress["metrics"][metric_name] = value
            self._save_progress()

    async def set_expected_papers(self, count: int) -> None:
        async with self.lock:
            self.progress["expected_papers"] = count
            self._save_progress()
            print(f"Progress Tracker: Expected papers set to {count}")

    async def set_received_papers(self, count: int) -> None:
        async with self.lock:
            self.progress["received_papers"] = count
            expected = self.progress["expected_papers"]
            self._save_progress()
            if count < expected:
                print(f"Progress Tracker: Paper count mismatch - Expected {expected}, received {count}")
            elif count > expected:
                print(f"Progress Tracker: More papers than expected - Expected {expected}, received {count}")
            else:
                print(f"Progress Tracker: All {count} papers received successfully")

    async def log_order_verification(self, stage_name: str, verification_result: dict) -> None:
        async with self.lock:
            entry = {
                "stage": stage_name,
                "verified": verification_result["verified"],
                "timestamp": datetime.now().isoformat(),
                "details": verification_result["details"],
            }
            if verification_result["mismatches"]:
                entry["mismatches"] = verification_result["mismatches"][:10]
            self.progress["order_verifications"].append(entry)
            self._save_progress()

    def _save_progress(self) -> None:
        temp_file = self.log_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, indent=2)
            temp_file.replace(self.log_file)
        except Exception as e:
            print(f"Progress Tracker: Save error: {e}")
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass

    def get_summary(self) -> str:
        completed = sum(1 for s in self.progress["stages"] if s["status"] == "completed")
        failed = sum(1 for s in self.progress["stages"] if s["status"] == "failed")
        total_papers = len(self.progress["papers"])
        order_verified = (
            all(v["verified"] for v in self.progress["order_verifications"])
            if self.progress["order_verifications"]
            else True
        )

        expected = self.progress.get("expected_papers", 0)
        received = self.progress.get("received_papers", 0)
        if expected > 0 and expected == received:
            paper_status = "All papers received"
        elif expected > 0:
            paper_status = f"{received}/{expected} papers"
        else:
            paper_status = f"{total_papers} papers"

        return (
            f"Progress Summary:\n"
            f"- Session: {self.progress['session_id']}\n"
            f"- Stages: {completed} completed, {failed} failed\n"
            f"- Papers: {paper_status} ({total_papers} tracked)\n"
            f"- Metrics: {len(self.progress['metrics'])} tracked\n"
            f"- Order Verifications: {len(self.progress['order_verifications'])} checks, "
            f"{'all passed' if order_verified else 'some failed'}\n"
        )
