import pandas as pd
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

# ============================================================
# 1. LOAD PDF → CHUNK → EMBED → STORE (VECTOR STORE)
# ============================================================

pdf_path = "bank_risk_definitions.pdf"  # <-- your bank-specific definitions

loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Chunk the PDF
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embed chunks into FAISS vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ============================================================
# 2. MODEL SETUP
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 3. PROMPTS
# ============================================================

classification_prompt = ChatPromptTemplate.from_template("""
You are a senior risk officer.

Below are the bank's authoritative risk definitions:
----------------
{context}
----------------

Risk event description:
{issue_description}

TASKS:
1. Classify the event into one or more *bank-defined risk types*.
2. Provide a 1–2 line summary.
3. Output ONLY valid JSON with these fields:
   - "risk_type": list of strings
   - "issue_summary": string
""")

# ============================================================
# 4. BUILD RAG CHAIN USING RunnableSequence
# ============================================================

rag_chain = (
    RunnableParallel(
        context=retriever, 
        issue_description=RunnablePassthrough()
    )
    | classification_prompt
    | llm
)

# ============================================================
# 5. PROCESS ENTIRE DATAFRAME USING ASYNC abatch()
# ============================================================

async def process_dataframe(df):
    inputs = df["issue_description"].tolist()

    # convert into proper runnable inputs
    runnable_inputs = [{"issue_description": text} for text in inputs]

    # run async batch
    try:
        print("Running async batch through LLM...")
        outputs = await rag_chain.abatch(runnable_inputs)
    except Exception as e:
        print("Batch failed, falling back to sequential async:", e)
        outputs = [await rag_chain.ainvoke(i) for i in runnable_inputs]

    summaries, risks = [], []

    for i, response in enumerate(outputs):
        raw = response.content.strip()

        try:
            result = eval(raw)       # safe because model outputs JSON-like
            summary = result.get("issue_summary", "")
            risk_types = result.get("risk_type", [])
        except:
            summary = df["issue_description"].iloc[i][:120] + "..."
            risk_types = ["Other"]

        summaries.append(summary)
        risks.append(", ".join(risk_types))

    df["issue_summary"] = summaries
    df["risk_type"] = risks

    df.to_csv("risk_events_enriched.csv", index=False)
    print("✅ Output saved to: risk_events_enriched.csv")


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("banking_dummy_issues.csv")   # <-- your file
    asyncio.run(process_dataframe(df))
