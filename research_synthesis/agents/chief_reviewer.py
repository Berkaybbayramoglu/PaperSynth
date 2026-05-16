"""Chief reviewer agent: final academic-quality gate after the first reviewer approves."""
from typing import Any, Dict

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory


class ChiefReviewerAgent:
    """Final approval pass: rejects reports that are too brief or repetitive."""

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)

    async def final_review(self, report: str) -> Dict[str, Any]:
        print("\nChief Reviewer: Final review...")

        history = ChatHistory()
        history.add_system_message(
            "You are the final quality reviewer ensuring COMPREHENSIVE COVERAGE. Verify:\n"
            "1. ALL papers are covered with substantial detail (not superficial)\n"
            "2. AI/Technology role is thoroughly explained across papers\n"
            "3. Concepts, use cases, techniques are comprehensively described\n"
            "4. Report is LONG enough (15-25 pages worth) with rich detail\n"
            "5. Methodologies and findings are accurately and fully described\n"
            "6. Report is practical, useful, and insightful\n"
            "7. Synthesis integrates findings across papers effectively\n"
            "8. NO excessive repetition - varied language throughout\n"
            "9. Balanced tone - not overusing negative phrases like 'DOES NOT'\n\n"
            "REJECT if report is too brief, lacks comprehensive detail, or has excessive repetition.\n"
            "Respond with: APPROVE or REJECT followed by high-level feedback."
        )
        history.add_user_message(f"Final review of this research report:\n\n{report}")

        settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id, max_tokens=1000, temperature=0.2,
        )
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history, settings=settings, kernel=self.kernel,
        )
        review_text = result[0].content if result else "Final review failed"
        approved = "APPROVE" in review_text.upper()
        print(f"Chief Reviewer: {'FINAL APPROVAL' if approved else 'REJECTED'}")
        return {"approved": approved, "feedback": review_text, "level": "chief_reviewer"}
