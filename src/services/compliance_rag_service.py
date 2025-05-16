# src/services/compliance_rag_service.py
import json
from typing import List
from .base_rag_service import BaseRAGService
from managers.regulations_manager import RegulationsManager


class ComplianceRAGService(BaseRAGService):
    """
    合規條文比對 RAG：
    - 先在向量庫中檢索與「user_query(外部法規)」最相近的公司內部段落 (source=internal)
    - 讓 LLM 分析是否符合
    - 回傳: code, label, evidence, confidence, start_idx, end_idx, similarity_score
    """

    def __init__(
        self,
        embedding_manager,
        vector_store_manager,
        llm_manager,
        regulations_manager: RegulationsManager,
        domain_key="COMPLIANCE",
        selected_llm_name=None,
    ):
        super().__init__(
            embedding_manager,
            vector_store_manager,
            llm_manager,
            domain_key,
            selected_llm_name
        )
        self.regulations_manager = regulations_manager
        self.prompt_header = (
            "你是公司法規合規比對助手。\n"
            "請直接輸出JSON array(陣列)，不要有其他解釋文字。\n"
            "每個物件包含: code, label, evidence, confidence, start_idx, end_idx。\n"
            "無對應則輸出 [].\n"
        )

    def build_prompt(self, user_query: str, context_docs: List[dict]) -> str:
        # context_docs = [{"chunk_id":"CSTI-01","text":"本會...", "score":0.88}, ...]
        lines = []
        for i, doc in enumerate(context_docs, start=1):
            lines.append(f"[Internal Doc#{i}] chunk_id={doc['chunk_id']} sim={doc['score']:.3f}\n{doc['text']}\n")
        docs_str = "\n".join(lines)

        example_json = """
        [
        {
            "chunk_id": "CSTI-1-SEC-001-3",
            "code": "CSMA-3",
            "label": "CSTI-1-SEC-001-3: 人員定義",
            "evidence": "本會於業務範...",
            "confidence": 0.85,
            "start_idx": 10,
            "end_idx": 25
        }
        ]
        """

        return (
            f"{self.prompt_header}\n"
            f"外部法規條文:\n{user_query}\n\n"
            f"--- 檢索到的內部文件 ---\n{docs_str}\n\n"
            "請進行比對，並在回傳的JSON中，一定要包含 chunk_id 以對應相似度.需注意evidence欄位必須完全依照原文不可隨意修改或省略.\n"
            f"範例:{example_json}\n"
        )
        
    def post_process(
        self,
        user_query: str,
        raw_json: list[dict],
        hits: List[dict]
    ) -> list[dict]:
        ev_key = self.cfg["evidence_key"]
        s_key, e_key = self.cfg["start_idx_key"], self.cfg["end_idx_key"]

        for rec in raw_json:
            chunk_id_from_llm = rec.get("chunk_id", "")
            match_doc = next((h for h in hits if h["chunk_id"] == chunk_id_from_llm), None)

            if match_doc:
                rec["similarity_score"] = match_doc["score"]
                doc_text = match_doc["text"]
            else:
                rec["similarity_score"] = 0.0
                doc_text = ""

            evidence_txt = rec.get(ev_key, "")
            if evidence_txt:
                start_idx = doc_text.find(evidence_txt)
                end_idx = start_idx + len(evidence_txt) if start_idx != -1 else -1
            else:
                start_idx = -1
                end_idx = -1

            rec[s_key], rec[e_key] = start_idx, end_idx

        return raw_json



class LawComplianceService:
    """
    orchestration:
      1) for each external clause → RAG(搜索internal) → LLM → parse
      2) 回傳對應表
      3) (可選) for each internal -> RAG(搜索external)
    """
    def __init__(self, compliance_rag: ComplianceRAGService):
        self.rag = compliance_rag

    async def audit(self, external_clauses: List[dict]) -> List[dict]:
        results = []
        for clause in external_clauses:
            law_text = clause["text"]
            # 針對每個外部法條, 只搜尋 source=internal
            rag_json = await self.rag.generate_answer(
                user_query=law_text,
                filters={"source": "internal"}
            )
            try:
                evidences = json.loads(rag_json)
            except:
                evidences = []
            results.append({
                "law_clause_id": clause.get("clause_id", ""),
                "law_article_no": clause.get("article_no", ""),
                "law_text": law_text,
                "evidences": evidences
            })
        return results

    async def audit_reverse(self, internal_clauses: List[dict]) -> List[dict]:
        results = []
        for doc in internal_clauses:
            doc_text = doc["text"]
            # 只搜尋外部
            rag_json = await self.rag.generate_answer(
                user_query=doc_text,
                filters={"source": "external"}
            )
            try:
                evidences = json.loads(rag_json)
            except:
                evidences = []
            results.append({
                "internal_clause_id": doc.get("clause_id", ""),
                "internal_article_no": doc.get("article_no", ""),
                "internal_text": doc_text,
                "evidences": evidences
            })
        return results
