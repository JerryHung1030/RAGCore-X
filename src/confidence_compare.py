# src/main.py
import asyncio
import os
import sys
from dotenv import load_dotenv
import logging
import json
import pandas as pd
# 確保可以 import 同層或上層資料夾的模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 以下為原先 managers / adapters / services import
from managers.embedding_manager import EmbeddingManager
from managers.vector_store_manager import VectorStoreManager
from managers.llm_manager import LLMManager
from managers.blacklist_manager import BlacklistManager
from managers.regulations_manager import RegulationsManager

from adapters.openai_adapter import OpenAIAdapter
from adapters.local_llama_adapter import LocalLlamaAdapter

from services.fraud_rag_service import FraudRAGService
from services.compliance_rag_service import ComplianceRAGService, LawComplianceService
from evaluation_utils import save_results_to_json, build_matrix, visualize_matrix

# 載入 .env
load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_managers():
    """根據 .env 配置，初始化所有核心 manager。"""

    # 1) 從 .env 讀取環境變數
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection = os.getenv("QDRANT_COLLECTION")
    embedding_model_name = os.getenv("EMBED_MODEL")

    # 2) 建立 EmbeddingManager
    embedding_manager = EmbeddingManager(
        openai_api_key=openai_api_key,
        embedding_model_name=embedding_model_name
    )

    # 3) 建立 VectorStoreManager (Qdrant)
    vector_store_manager = VectorStoreManager(
        embedding_manager=embedding_manager,
        qdrant_url=qdrant_url,
        collection_name=qdrant_collection
    )

    # 4) 建立 LLMManager，並註冊多個 LLM Adapter
    llm_manager = LLMManager()

    openai_adapter = OpenAIAdapter(
        openai_api_key=openai_api_key,
        temperature=0.0,
        max_tokens=1024
    )

    local_llama_adapter = LocalLlamaAdapter(
        model_path="models/llama.bin",
        temperature=0.0,
        max_tokens=2048
    )

    llm_manager.register_adapter("openai", openai_adapter)
    llm_manager.register_adapter("llama", local_llama_adapter)
    llm_manager.set_default_adapter("openai")

    # 4) 其他
    blacklist_manager = BlacklistManager(blacklist_db=["badurl.com", "lineid123"])
    regulations_manager = RegulationsManager(regulations_db={"some_law": "Lorem ipsum..."})

    return embedding_manager, vector_store_manager, llm_manager, blacklist_manager, regulations_manager


def load_jsonl_file(path: str):
    """小幫手：讀取 JSON lines 檔並回傳 list[dict]."""
    if not path or not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return []
    all_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_data.append(json.loads(line))
    return all_data


async def main():
    # 初始化 Managers
    embedding_manager, vector_store_manager, llm_manager, blacklist_manager, regulations_manager = setup_managers()

    # 讀取詐騙 patterns 並加入向量庫
    scam_file = os.getenv("SCAM_PATTERNS_FILE")
    scam_patterns = load_jsonl_file(scam_file)
    vector_store_manager.add_documents(
        domain="FRAUD",
        docs=scam_patterns,
        metadata={"type": "scam_pattern"}
    )

    # 建立 FraudRAGService
    fraud_service = FraudRAGService(
        embedding_manager=embedding_manager,
        vector_store_manager=vector_store_manager,
        llm_manager=llm_manager,
        blacklist_manager=blacklist_manager,
        domain_key="FRAUD",
        selected_llm_name="openai"
    )

    # 讀取 input_data.xlsx 的「使用者輸入」欄位
    df = pd.read_excel("input_data_c70_k10.xlsx")
    print("Excel 欄位名稱如下：", df.columns.tolist())
    input_col = next((col for col in df.columns if "使用者輸入" in col), None)
    if not input_col:
        logger.error("找不到包含『使用者輸入』字樣的欄位")
        return

    results = []

    for idx, user_input in enumerate(df[input_col].dropna()):
        print(f"處理第 {idx+1} 筆輸入...")
        raw_output = await fraud_service.generate_answer(user_input)

        # 提取需要欄位
        simplified = [
            {
                "id": item.get("code"),
                "label": item.get("label"),
                "evidence": item.get("evidence"),
                "confidence": item.get("confidence")
            }
            for item in raw_output
            if item.get("confidence", 0) >= 0.7
        ]

        results.append(simplified)

    # 印出整理後的 JSON
    print(json.dumps(results, ensure_ascii=False, indent=2))
    # 將結果轉成文字（例如 JSON 字串）
    result_strings = [json.dumps(r, ensure_ascii=False) for r in results]

    # 若欄位不存在，新增空白欄
    result_col = "GPT4o的詐騙模式偵測結果"
    if result_col not in df.columns:
        df[result_col] = ""

    # 將結果寫入 DataFrame（與原始資料對齊）
    non_empty_indices = df[input_col].dropna().index
    for i, idx in enumerate(non_empty_indices):
        df.at[idx, result_col] = result_strings[i]

    # 儲存回 Excel（可改名以避免覆蓋）
    df.to_excel("input_data_c80_k10.xlsx", index=False)
    print("已將結果寫入 output_data.xlsx")

if __name__ == "__main__":
    asyncio.run(main())
