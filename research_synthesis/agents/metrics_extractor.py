"""Metrics extractor agent: pulls numeric results out of the per-paper analyses."""
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory


class MetricsExtractorAgent:
    """Produces a flat ``Header: Value`` style metrics summary for the final report."""

    def __init__(self, kernel: Kernel, service_id: str):
        self.kernel = kernel
        self.service_id = service_id
        self.chat_service = kernel.get_service(service_id)

    async def extract_metrics(self, analysis: str) -> str:
        print("\nMetrics Extractor: Extracting numerical data...")

        history = ChatHistory()
        history.add_system_message(
            "You are a metrics extraction specialist. Extract ALL numerical data from papers.\n\n"
            "For EACH paper, extract:\n"
            "1. All accuracy/precision/recall/F1 scores\n"
            "2. All percentage improvements\n"
            "3. All cost reductions or efficiency gains\n"
            "4. Dataset sizes and characteristics\n"
            "5. Model parameters (trees, layers, learning rates, etc.)\n"
            "6. Processing times, computational costs\n"
            "7. Any other numerical results\n\n"
            "Format as plain text summary (NO markdown, NO tables with pipes):\n\n"
            "METRICS SUMMARY\n\n"
            "Paper 1 - Title\n"
            "- Accuracy: 95.3%\n"
            "- Precision: 92.1%\n"
            "- Dataset: 10,000 samples\n"
            "- Model: Random Forest (100 trees, max_depth=10)\n\n"
            "Paper 2 - Title\n"
            "...\n\n"
            "Write plain text format. Be thorough - include EVERY metric."
        )
        history.add_user_message(f"Extract all metrics from these paper analyses:\n\n{analysis[:10000]}")

        settings = self.chat_service.get_prompt_execution_settings_class()(
            service_id=self.service_id, max_tokens=2000, temperature=0.1,
        )
        result = await self.chat_service.get_chat_message_contents(
            chat_history=history, settings=settings, kernel=self.kernel,
        )
        metrics = result[0].content if result else "Metrics extraction failed"
        print("Metrics Extractor: Extraction complete")
        return metrics
