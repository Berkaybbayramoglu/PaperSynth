"""Order verification agent: ensures pipeline stages don't reorder papers."""
from pathlib import Path
from typing import Dict, List

from semantic_kernel import Kernel


class OrderVerificationAgent:
    """Checks that processed data still matches the order of the original paper paths."""

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id

    async def verify_order(
        self, original_paths: List[str], processed_data: List, stage_name: str,
    ) -> Dict:
        print(f"\nOrderVerification: Checking {stage_name}...")

        if len(original_paths) != len(processed_data):
            return {
                "verified": False,
                "mismatches": ["Length mismatch"],
                "details": f"Expected {len(original_paths)} items, got {len(processed_data)}",
            }

        mismatches: List[str] = []
        for i, (path, data) in enumerate(zip(original_paths, processed_data)):
            expected_name = Path(path).stem.lower()
            data_str = str(data).lower() if data else ""
            filename_found = expected_name in data_str
            position_marker = f"paper {i + 1}" in data_str or f"paper_{i + 1}" in data_str
            if not (filename_found or position_marker):
                mismatches.append(
                    f"Position {i}: Expected '{Path(path).name}' (paper {i + 1}) not found in data"
                )

        if mismatches:
            details = f"Found {len(mismatches)} order mismatches:\n" + "\n".join(mismatches[:5])
            print(f"   Order verification FAILED: {len(mismatches)} mismatches")
            return {"verified": False, "mismatches": mismatches, "details": details}

        print(f"   Order verified: All {len(original_paths)} items in correct order")
        return {
            "verified": True,
            "mismatches": [],
            "details": f"All {len(original_paths)} items verified in correct order",
        }
