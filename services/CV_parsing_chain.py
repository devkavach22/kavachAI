import json
import os
import time

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from schemas.CV_Schema import CVData_formate

# -----------------------------------------------------
# 1️⃣ Define Pydantic Models
# -----------------------------------------------------

# -----------------------------------------------------
# 2️⃣ Initialize OpenAI or vLLM API
# -----------------------------------------------------
# os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
# os.environ["OPENAI_API_KEY"] = "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"
# api_key = os.getenv("OPENAI_API_KEY")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key = os.getenv("OPENAI_API_KEY"),
# )

# print("✅ Using OpenRouter API Base:", OPENAI_API_BASE)
# print("✅ Using OpenRouter API Key:", OPENAI_API_KEY)
# print("✅ Using OpenRouter API Key:", os.getenv("DEBUG"))


llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL","openai/gpt-oss-20b:free"), #openai/gpt-oss-20b:free
    temperature=0,
    max_tokens=4096,
)


# -----------------------------------------------------
# 3️⃣ Create Output Parser
# -----------------------------------------------------
parser = PydanticOutputParser(pydantic_object=CVData_formate)


# -----------------------------------------------------
# 4️⃣ Create Prompt Template
# -----------------------------------------------------
schema_json_safe = json.dumps(CVData_formate.model_json_schema(), indent=2)
schema_json_safe = schema_json_safe.replace("{", "{{").replace("}", "}}")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""
You are an intelligent AI CV parser. 
Your task is to extract and structure **maximum possible information** from a candidate’s CV/resume text into a clean JSON format that strictly follows this schema:

{schema_json_safe}

### 📘 Parsing & Extraction Guidelines:
1. **Extract All Possible Information**  
   - Parse every relevant section: personal info, experience, education, training, projects, and skills.  
   - Even if some details are missing or implicit, infer them when possible (e.g., derive location or company if mentioned contextually).

2. **Fill Every Field Intelligently**  
   - If a field is missing, use an empty string (`""`) or an empty list (`[]`) — but never omit it.  
   - Always include all schema fields.  
   - For nested models (like skills or projects), ensure all subfields are present, even if blank.

3. **Be Strict About JSON Validity**  
   - The output must be a **valid JSON** that can be parsed without errors.  
   - Do not include markdown, code fences, comments, or explanations.

4. **Data Normalization Rules**
   - Use concise, clean text for all values (no bullet characters, tabs, or special symbols).
   - Standardize dates as `"MM/YYYY"` or `"YYYY"`.
   - Combine scattered experiences into structured lists.
   - Use reasonable defaults if not clearly stated:
     - Missing end_date → `"present"`
     - Missing location → `""`
     - Missing technologies or skills → empty list `[]`

5. **Accuracy & Completeness**
   - Preserve original spelling for names and companies.
   - Include as much as possible under each field.
   - Prefer more detailed and structured extraction over brevity.

6. **Character-Level Completeness**
   - Use **all available text** from the CV to fill in the schema fields.
   - **Do not skip, omit, summarize, or ignore any characters, numbers, punctuation, or details** that could belong to a field.
   - Every identifiable piece of information from the CV must be represented somewhere in the JSON output, even if partial or uncertain.

Return only the structured JSON that matches the schema exactly.
        """
    ),
    ("user", "{cv_text}")
])



# -----------------------------------------------------
# 5️⃣ Combine Chain
# -----------------------------------------------------
chain = prompt | llm | parser


# -----------------------------------------------------
# 6️⃣ Function to Run the Chain
# -----------------------------------------------------
def get_cv_data_from_openrouter_model(cv_text: str) -> CVData_formate: 
    result = chain.invoke({"cv_text": cv_text})

    # print(" Result -> ",result)
    
    #convert restult into dict/json
    result_dict = result.model_dump()
    
    # print("converted to dict -1",result_dict)
    # print("�� Converted to dict/json:", type(result_dict))

    return result_dict


# -----------------------------------------------------
# 7️⃣ Example Usage
# -----------------------------------------------------
if __name__ == "__main__":
    sample_cv_text = """
    Ravina Akashkumar Soni

    """

    structured_data = get_cv_data_from_openrouter_model(sample_cv_text)
    # print(structured_data)
    
    converted_dict_1 = structured_data.model_dump()
    
    # print(" testing -->> ",CVData_formate(**converted_dict_1))
    
    # print("converted to dict -1",type(converted_dict_1))
    # print("--------------------------------")
    # print("converted data ->",converted_dict_1) 
    
