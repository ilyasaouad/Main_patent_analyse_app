from .document_reader_agent.agent import DocumentReaderAgent

class ClaimsAnalystAgent:
    def run(self, state):
        return {"next_step": "prior_art"}

class PriorArtSearchAgent:
    def run(self, state):
        return {"next_step": "novelty"}

class NoveltyAgent:
    def run(self, state):
        return {"next_step": "inventive_step"}

class InventiveStepAgent:
    def run(self, state):
        return {"next_step": "industrial"}

class IndustrialApplicabilityAgent:
    def run(self, state):
        return {"next_step": "infringement"}

class InfringementAgent:
    def run(self, state):
        return {"next_step": "report"}

class ReportGeneratorAgent:
    def run(self, state):
        return {"next_step": "END"}
