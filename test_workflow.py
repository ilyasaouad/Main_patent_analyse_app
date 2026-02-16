
from config.models import PatentAnalysisState
from workflow import create_patent_workflow

state = PatentAnalysisState(description_path="test.pdf")
inputs = state.model_dump()
workflow = create_patent_workflow()
try:
    result = workflow.invoke(inputs)
    print("Workflow successful")
except Exception as e:
    print(f"Workflow failed: {e}")
    import traceback
    traceback.print_exc()
