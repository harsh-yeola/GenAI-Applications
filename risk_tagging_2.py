import pandas as pd
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# -------------------------------
# 1) Load CSV into DataFrame
# -------------------------------
file_path = "banking_dummy_issues.csv"  # <-- replace with your file path
df = pd.read_csv(file_path)

# -------------------------------
# 2) Setup LLM (requires OPENAI_API_KEY)
# -------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt template for issue_summary and risk_type
prompt_template = ChatPromptTemplate.from_template("""
You are an expert in banking risk management.

Issue description: {issue_description}

1. Provide a concise 1-2 line summary of the issue.
2. Assign the most appropriate risk type from the following categories:
   - Operational Risk
   - Technology Risk
   - Compliance Risk
   - Financial Risk
   - Reputational Risk
   - Other

Return your answer in JSON format with keys: issue_summary, risk_type.
""")

chain = LLMChain(llm=llm, prompt=prompt_template)

# -------------------------------
# 3) Rule-based fallback classifier
# -------------------------------
def rule_based_risk_classification(issue_description: str) -> str:
    desc = issue_description.lower()
    if any(word in desc for word in ["server", "system", "app", "software", "technology", "portal", "atm", "network"]):
        return "Technology Risk"
    elif any(word in desc for word in ["delay", "manual", "clerical", "error", "batch", "process", "job failure"]):
        return "Operational Risk"
    elif any(word in desc for word in ["compliance", "kyc", "regulation", "audit"]):
        return "Compliance Risk"
    elif any(word in desc for word in ["interest", "charges", "payment", "debit", "credit", "funds", "amount"]):
        return "Financial Risk"
    elif any(word in desc for word in ["complaint", "apology", "customer dissatisfaction", "trust", "reputation"]):
        return "Reputational Risk"
    else:
        return "Other"

# -------------------------------
# 4) Process DataFrame
# -------------------------------
summaries = []
risks = []

for desc in df["issue_description"]:
    try:
        # Try LLM first
        response = chain.run(issue_description=desc)
        result = eval(response)  # assuming model returns valid JSON-like string
        issue_summary = result.get("issue_summary", "").strip()
        risk_type = result.get("risk_type", "").strip()

        # Fallback if LLM gives empty response
        if not issue_summary or not risk_type:
            issue_summary = desc[:100] + "..."  # truncate long description
            risk_type = rule_based_risk_classification(desc)

    except Exception as e:
        # If LLM fails, use rule-based fallback
        issue_summary = desc[:100] + "..."  # first 100 chars as summary
        risk_type = rule_based_risk_classification(desc)

    summaries.append(issue_summary)
    risks.append(risk_type)

df["issue_summary"] = summaries
df["risk_type"] = risks

# -------------------------------
# 5) Save updated DataFrame
# -------------------------------
df.to_csv("banking_issues_with_risks.csv", index=False)

print("✅ Processing complete! File saved as 'banking_issues_with_risks.csv'")
