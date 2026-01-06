from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import json
import math


def clean_nans(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj


llm = ChatOpenAI(model="openai/gpt-oss-20b:free", temperature=0.3)


def excel_report_chain(df: pd.DataFrame):
    """
    More advanced LangChain report generator.
    Automatically analyzes dataset, summaries, anomalies, correlations,
    and produces insights suitable for Excel/BI reporting.

    Args:
        df: pandas DataFrame input.

    Returns:
        Function run(query) -> detailed Excel friendly report.
    """


    # ----------------- 📊 Detailed Report Data -----------------
    report_data = {
        "filename": "filename_placeholder",  # Filename is not passed to this function
        "file_extension": "ext_placeholder",  # Extension is not passed to this function
        "column_names": df.columns.tolist(),
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "total_elements": df.size,
        "column_data_types": df.dtypes.apply(str).to_dict(),
        "null_counts_per_column": df.isnull().sum().to_dict(),
        "descriptive_statistics": df.describe().to_dict(
            orient="list"
        ),  # include='all' to get stats for non-numeric too
        "unique_values_count_per_column": df.nunique().to_dict(),
        "head": df.head().to_dict(orient="list"),
        "tail": df.tail().to_dict(orient="list"),
        "sample": df.sample(min(5, len(df))).to_dict(orient="list"),
        "duplicate_rows_count": df.duplicated().sum(),
        "correlation_matrix_sample": df.select_dtypes(include=["number"])
        .corr()
        .head()
        .to_dict(orient="index"),
        "top_5_value_counts_per_column": {
            col: df[col].value_counts().head(5).to_dict()
            for col in df.columns
            if df[col].nunique() < 20 and df[col].dtype == "object"
        },
        "missing_values_per_column": df.isnull().sum().to_dict(),
    }

    report_data_to_string = str(report_data)

    # ------------------------ 📄 Final Report Prompt -----------------------------

    template = """
    You are a Senior Data Analyst and BI Report Intelligence System.
    Your job is to analyze the dataset summaries and generate a domain-aware report.
    
    ====================================================
    📁 DATA ANALYSIS SUMMARY
    {df_preview}
    
    ====================================================
    USER REQUEST / EXPECTATION:
    {query}
    ====================================================
    
    ### 🔥 INTELLIGENT BEHAVIOR REQUIRED
    
    1. Detect dataset theme automatically  
       - If columns hint Revenue/Orders → treat as SALES report  
       - If Leads/Campaign/Click data → treat as MARKETING  
       - If Salary/Employee/Attendance → treat as HRMS  
       - If Profit/Expense/Ledger → treat as FINANCE  
       - If Stock/Inventory/Logistics/Warehouse → treat as SUPPLY CHAIN  
       - If none match → generate a neutral analytical report  
    
    2. Never assume — infer insight from data patterns
    
    ====================================================
    
    ### 📌 FINAL OUTPUT FORMAT
    
    ## 1) Executive Summary (Auto-Domain Aware)
    Present 8–12 sharp insights customized to domain.
    
    ## 2) Column-Wise Deep Exploration (Structured Output)
    Return a table with Observed Pattern and Key Insight.
    
    ## 3) Trend + Correlation Story
    - Detect correlation clusters  
    - Identify seasonal cycles  
    - Detect performance inflection points  
    
    ## 4) Data Quality & Risk Evaluation
    Show data integrity score and suggestions.
    
    ## 5) KPI + Chart Blueprint
    Dynamically propose metrics and visuals.
    
    ## 6) Strategic Recommendation System
    Provide improvement decisions with measurable outcome.
    
    ====================================================
    Generate the final business report now.
    """

    prompt = PromptTemplate(
        input_variables=["df_preview", "query"],
        template=template,
    )

    final_chain = prompt | llm | StrOutputParser()

    def run(query: str):
        # Generate final report using the pre-calculated report data
        result = final_chain.invoke(
            {
                "df_preview": report_data_to_string,
                "query": query,
            }
        )
        return {"report": result}

    return run


# How to Use
# report_chain = excel_report_chain(df)
# response = report_chain("Generate a sales trend analysis + KPI suggestions")
# print(response["report"])
