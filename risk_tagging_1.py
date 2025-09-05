import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# 1) Load CSV into DataFrame
file_path = "banking_dummy_issues.csv"  # <-- replace with your file path
df = pd.read_csv(file_path)

# 2) Setup LLM (requires OPENAI_API_KEY environment variable)
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

# 3) Define LangChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# 4) Apply chain to DataFrame
summaries = []
risks = []

for desc in df["issue_description"]:
    response = chain.run(issue_description=desc)
    try:
        # Parse JSON-like response
        result = eval(response)
        summaries.append(result.get("issue_summary", ""))
        risks.append(result.get("risk_type", ""))
    except:
        summaries.append("Parsing error")
        risks.append("Unknown")

df["issue_summary"] = summaries
df["risk_type"] = risks

# 5) Save updated DataFrame
df.to_csv("banking_issues_with_risks.csv", index=False)

print("✅ Processing complete! File saved as 'banking_issues_with_risks.csv'")
