"""Reviewer agent: first-pass technical quality check."""
from typing import Any, Dict

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory


class ReviewerAgent:
    """First-level review: APPROVE/REJECT based on technical completeness."""

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)

    async def review(self, report: str) -> Dict[str, Any]:
        print("\nReviewer: Reviewing report...")

        history = ChatHistory()
        history.add_system_message(
            "You are an EXTREMELY STRICT reviewer checking for COMPLETE DETAIL.\n\n"
            "Check EVERY paper section individually:\n\n"
            "FOR EACH PAPER CHECK:\n"
            "1. Are ALL methods listed by EXACT name? (Random Forest, XGBoost, ResNet-50, etc.)\n"
            "2. Are method parameters included? (number of trees, learning rate, layers, etc.)\n"
            "3. Are ALL results listed with EXACT metrics? (95.3% accuracy, not 'high accuracy')\n"
            "4. Are ALL technologies/tools/platforms mentioned by name?\n"
            "5. Are dataset details complete? (exact size, source, features)\n"
            "6. Are experiment details complete? (what was tested, how)\n"
            "7. Are ALL comparisons with baselines included?\n"
            "8. Can a reader fully understand this paper WITHOUT reading the original?\n\n"
            "GENERAL CHECKS:\n"
            "9. Each paper has 5+ paragraphs of detailed content?\n"
            "10. Integrated synthesis section compares ALL papers with specific details?\n"
            "11. No vague language like 'various methods', 'good results', 'machine learning'?\n"
            "12. No markdown symbols (**, |, ---, ###)?\n"
            "13. Report is 20+ pages of dense, detailed content?\n"
            "14. NO EXCESSIVE REPETITION - same sentence/paragraph repeated multiple times?\n"
            "15. VARIED LANGUAGE - not overusing 'DOES NOT', 'THE PAPER DOES NOT', 'LACKS'?\n\n"
            "REJECT if you find:\n"
            "- ANY generic/vague descriptions\n"
            "- Missing specific metrics or method names\n"
            "- Papers described in less than 4-5 paragraphs\n"
            "- Any detail that seems incomplete\n"
            "- Same sentences or paragraphs repeated multiple times\n"
            "- Excessive use of negative phrases like 'DOES NOT' (more than 5-10 times)\n\n"
            "Respond: APPROVE or REJECT with LIST of specific missing details."
        )
        history.add_user_message(f"Review this research report:\n\n{report}")

        settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id, max_tokens=1000, temperature=0.2,
        )
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history, settings=settings, kernel=self.kernel,
        )
        review_text = result[0].content if result else "Review failed"
        approved = "APPROVE" in review_text.upper()
        print(f"Reviewer: {'APPROVED' if approved else 'REJECTED'}")
        return {"approved": approved, "feedback": review_text, "level": "reviewer"}
