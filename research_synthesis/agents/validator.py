"""Validation agent: cross-checks the report against the synthesis to catch hallucinations."""
from typing import Any, Dict

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory


class ValidationAgent:
    """Flags claims in the report that aren't supported by the synthesis text."""

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)

    async def validate(self, report: str, synthesis: str) -> Dict[str, Any]:
        print("\nValidation Agent: Checking for fabricated information...")

        history = ChatHistory()
        history.add_system_message(
            "You are a fact-checking validator. Compare the REPORT against the SYNTHESIS.\n\n"
            "Your ONLY job: Find any information in the report that is NOT in the synthesis.\n\n"
            "CHECK FOR:\n"
            "1. Methods/algorithms named in report but not in synthesis\n"
            "2. Metrics/numbers in report but not in synthesis\n"
            "3. Technologies/tools in report but not in synthesis\n"
            "4. Paper details in report but not in synthesis\n"
            "5. Any claims, results, or findings not supported by synthesis\n\n"
            "CRITICAL: Flag ANY fabricated or hallucinated content.\n\n"
            "If report only contains information from synthesis: VALID\n"
            "If report contains ANY unsupported information: INVALID\n\n"
            "Respond: VALID or INVALID with list of fabricated items."
        )
        history.add_user_message(
            f"SYNTHESIS (source of truth):\n{synthesis[:8000]}\n\n"
            f"REPORT (check against synthesis):\n{report[:8000]}\n\n"
            f"Does report contain ANY information not in synthesis?"
        )

        settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id, max_tokens=800, temperature=0.1,
        )
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history, settings=settings, kernel=self.kernel,
        )
        validation = result[0].content if result else "Validation failed"
        is_valid = "VALID" in validation.upper() and "INVALID" not in validation.upper()
        print(
            "Validation Agent: "
            + ("VALID - No hallucinations detected" if is_valid else "INVALID - Hallucinations found")
        )
        return {
            "valid": is_valid,
            "feedback": validation,
            "issues_found": [] if is_valid else validation.split("\n"),
        }
