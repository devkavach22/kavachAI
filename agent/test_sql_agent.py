"""
Test script for SQL Agent with MySQL database.
"""

from SQLExcel_report_agent import run_sql_query, SQLAgent

# Database configuration
DB_USER = "root"
DB_PASSWORD = "Kavach1234"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "excel_db"

# Construct database URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Table name
TABLE_NAME = "excel_data_temp"

print("=" * 70)
print("SQL Agent Test with MySQL Database")
print("=" * 70)
print(f"\nDatabase: {DB_NAME}")
print(f"Table: {TABLE_NAME}")
print(f"Connection: mysql+pymysql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
print("=" * 70)

# Test queries
test_queries = [
    "How many records are in the table?",
    "What are the column names in the table?",
    "Show me the first 5 rows of data",
    "What is the schema of the table?",
]

print("\n\n### Running Test Queries ###\n")

for i, query in enumerate(test_queries, 1):
    print(f"\n{'=' * 70}")
    print(f"Query {i}: {query}")
    print("=" * 70)

    try:
        result = run_sql_query(
            database_url=DATABASE_URL,
            query=query,
            table_name=TABLE_NAME,
            llm_model="gpt-3.5-turbo",
            verbose=True,
        )

        print(f"\n✓ Final Answer: {result}")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")

    print("\n" + "=" * 70)

print("\n\nTest completed!")
