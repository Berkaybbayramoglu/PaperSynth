"""Disk-backed cache so identical papers don't get re-analyzed each run."""
import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class MemoryAgent:
    """JSON cache keyed by ``paper_title + content_hash``.

    Used by :class:`AnalyzerAgent` to skip the LLM call when the same paper
    has been analyzed before. Saves are atomic (write-temp-then-rename) so an
    interrupted run can't corrupt the cache file.
    """

    def __init__(self, cache_file: str = "analysis_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict = self._load_cache()
        self.lock: Optional[asyncio.Lock] = None

    def __del__(self):
        try:
            if hasattr(self, "cache") and self.cache:
                self._save_cache()
        except Exception:
            pass

    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    print(f"\nMemory Agent: Loaded cache from {self.cache_file}")
                    return json.load(f)
            except Exception as e:
                print(f"\nMemory Agent: Cache load error: {e}")
                return {}
        return {}

    def _save_cache(self) -> None:
        temp_file = self.cache_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            temp_file.replace(self.cache_file)
            print(f"Memory Agent: Cache saved to {self.cache_file}")
        except Exception as e:
            print(f"Memory Agent: Cache save error: {e}")
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass

    async def _ensure_lock(self) -> None:
        if self.lock is None:
            self.lock = asyncio.Lock()

    async def get_analysis(self, paper_title: str, paper_hash: str) -> Optional[Dict]:
        await self._ensure_lock()
        async with self.lock:
            key = f"{paper_title}_{paper_hash}"
            if key in self.cache:
                print(f"Memory Agent: Using cached analysis for {paper_title[:50]}")
                return self.cache[key]
            return None

    async def store_analysis(self, paper_title: str, paper_hash: str, analysis: str) -> None:
        await self._ensure_lock()
        async with self.lock:
            key = f"{paper_title}_{paper_hash}"
            self.cache[key] = {
                "paper_title": paper_title,
                "paper_hash": paper_hash,
                "analysis": analysis,
                "cached_at": datetime.now().isoformat(),
            }
            self._save_cache()
            print(f"Memory Agent: Cached analysis for {paper_title[:50]}")

    async def clear_cache(self) -> None:
        await self._ensure_lock()
        async with self.lock:
            self.cache = {}
            self._save_cache()
            print("Memory Agent: Cache cleared")

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
