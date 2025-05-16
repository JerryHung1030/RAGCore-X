from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from src.main import setup_managers, load_jsonl_file
from services.fraud_rag_service import FraudRAGService
from services.compliance_rag_service import ComplianceRAGService, LawComplianceService

app = FastAPI()

# 初始化 managers (可考慮設成 Singleton)
embedding_manager, vector_store_manager, llm_manager, blacklist_manager, regulations_manager = setup_managers()

# 初始化服務
fraud_service = FraudRAGService(
    embedding_manager=embedding_manager,
    vector_store_manager=vector_store_manager,
    llm_manager=llm_manager,
    blacklist_manager=blacklist_manager,
    domain_key="FRAUD",
    selected_llm_name="openai"
)

compliance_service = ComplianceRAGService(
    embedding_manager=embedding_manager,
    vector_store_manager=vector_store_manager,
    llm_manager=llm_manager,
    regulations_manager=regulations_manager,
    domain_key="COMPLIANCE",
    selected_llm_name="openai"
)
law_service = LawComplianceService(compliance_rag=compliance_service)


# --- API schema ---
class FraudCheckRequest(BaseModel):
    text: str

class LawItem(BaseModel):
    text: str
    source: Optional[str] = None
    doc_name: Optional[str] = None
    chapter_no: Optional[str] = None
    chapter_name: Optional[str] = None
    article_no: Optional[str] = None
    effective_date: Optional[str] = None
    clause_id: Optional[str] = None

class ComplianceAuditRequest(BaseModel):
    items: List[LawItem]


# --- API Routes ---

@app.post("/api/fraud/check")
async def check_fraud(request: FraudCheckRequest):
    try:
        result = await fraud_service.generate_answer(request.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compliance/forward_audit")
async def forward_audit(request: ComplianceAuditRequest):
    try:
        result = await law_service.audit([item.dict() for item in request.items])
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compliance/reverse_audit")
async def reverse_audit(request: ComplianceAuditRequest):
    try:
        result = await law_service.audit_reverse([item.dict() for item in request.items])
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
