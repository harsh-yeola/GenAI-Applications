import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

# ============================================================
# 1. READ USER-UPLOADED FILES
# ============================================================

csv_path = "/mnt/data/banking_system_issues.csv"
risk_def_path = "/mnt/data/Non Financial Risk.txt"

df = pd.read_csv(csv_path)

with open(risk_def_path, "r", encoding="utf-8") as f:
    risk_definitions_text = f.read()

# ============================================================
# 2. CHUNK RISK DEFINITIONS AND CREATE VECTOR STORE
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

risk_chunks = text_splitter.split_text(risk_definitions_text)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.from_texts(risk_chunks, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ============================================================
# 3. BUILD RAG LLM CHAIN
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a senior banking risk officer.
Use the retrieved bank-specific risk definitions below to classify the event.

=== Risk Definitions ===
{context}

=== Event Description ===
{issue_description}

TASKS:
1. Produce a 5–8 word "gist" capturing the core issue.
2. Provide a short "rationale" (1 sentence) explaining *why* the chosen risk applies.
3. Select ONLY the single most appropriate "risk_type" using the risk definitions.

Return ONLY JSON in this format:
{{
  "gist": "...",
  "rationale": "...",
  "risk_type": "..."
}}
""")

rag_chain = (
    RunnableParallel(
        context=retriever,
        issue_description=RunnablePassthrough()
    )
    | prompt
    | llm
)

# ============================================================
# 4. PROCESS THE ENTIRE DATASET
# ============================================================

gists = []
rationales = []
risk_types = []

for desc in df["issue_description"].tolist():
    out = rag_chain.invoke({"issue_description": desc})
    try:
        parsed = eval(out.content)
        gists.append(parsed.get("gist", ""))
        rationales.append(parsed.get("rationale", ""))
        risk_types.append(parsed.get("risk_type", ""))
    except:
        # fallback
        gists.append("Unable to extract gist")
        rationales.append("Unable to infer rationale")
        risk_types.append("Unknown")

df["gist"] = gists
df["rationale"] = rationales
df["risk_type"] = risk_types

# ============================================================
# 5. SAVE FINAL DATAFRAME
# ============================================================

output_path = "/mnt/data/risk_classified_output.csv"
df.to_csv(output_path, index=False)

output_path
