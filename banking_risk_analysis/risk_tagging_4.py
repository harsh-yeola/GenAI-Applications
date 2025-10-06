import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# -------------------------------
# 1) Load CSV into DataFrame
# -------------------------------
file_path = "banking_dummy_issues.csv"  # <-- replace with your file path
df = pd.read_csv(file_path)

# -------------------------------
# 2) Setup LLM (requires OPENAI_API_KEY)
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt template for issue_summary and multi-label risk_type
prompt_template = ChatPromptTemplate.from_template("""
You are an expert in banking risk management.

Issue description: {issue_description}

1. Provide a concise 1-2 line summary of the issue.
2. Assign one or more risk types from the following categories (multi-label allowed):
   - Operational Risk
   - Technology Risk
   - Compliance Risk
   - Financial Risk
   - Reputational Risk
   - Other

Return your answer in JSON format with keys: issue_summary (string), risk_type (list of strings).
""")

# -------------------------------
# 3) Build chain using RunnableSequence
# -------------------------------
chain = prompt_template | llm

# -------------------------------
# 4) Rule-based multi-label fallback classifier
# -------------------------------
def rule_based_risk_classification(issue_description: str) -> list:
    desc = issue_description.lower()
    labels = []

    if any(word in desc for word in ["server", "system", "app", "software", "technology", "portal", "atm", "network"]):
        labels.append("Technology Risk")
    if any(word in desc for word in ["delay", "manual", "clerical", "error", "batch", "process", "job failure"]):
        labels.append("Operational Risk")
    if any(word in desc for word in ["compliance", "kyc", "regulation", "audit"]):
        labels.append("Compliance Risk")
    if any(word in desc for word in ["interest", "charges", "payment", "debit", "credit", "funds", "amount"]):
        labels.append("Financial Risk")
    if any(word in desc for word in ["complaint", "apology", "customer dissatisfaction", "trust", "reputation"]):
        labels.append("Reputational Risk")

    if not labels:
        labels.append("Other")

    return labels

# -------------------------------
# 5) Process DataFrame
# -------------------------------
summaries = []
risks = []

for desc in df["issue_description"]:
    try:
        # Call the chain
        response = chain.invoke({"issue_description": desc})
        raw_output = response.content.strip()

        # Try parsing JSON-like response
        result = eval(raw_output)
        issue_summary = result.get("issue_summary", "").strip()
        risk_type = result.get("risk_type", [])

        if not issue_summary:
            issue_summary = desc[:100] + "..."
        if not risk_type:
            risk_type = rule_based_risk_classification(desc)

    except Exception:
        # If LLM fails, use fallback
        issue_summary = desc[:100] + "..."
        risk_type = rule_based_risk_classification(desc)

    summaries.append(issue_summary)
    risks.append(", ".join(risk_type))  # store as comma-separated string

df["issue_summary"] = summaries
df["risk_type"] = risks

# -------------------------------
# 6) Save updated DataFrame
# -------------------------------
df.to_csv("banking_issues_with_multi_risks.csv", index=False)

print("✅ Processing complete! File saved as 'banking_issues_with_multi_risks.csv'")
