from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from services.data_extraction import FileDataExtractor
from services.excel_report_chain import excel_report_chain
from pydantic import BaseModel
from db import excelDB
import pandas as pd
import os
import json
import xlrd
import io
import uuid

router = APIRouter()

# add pydantic model for excel report funtion
class ExcelReportRequest(BaseModel):
    file: UploadFile
    query: str

df = None
@router.post("/excel_report")
async def excel_report(file: UploadFile = File(...), query: str = Query(...)):
    # Validate that the uploaded file has the expected extension
    _, file_ext = os.path.splitext(file.filename.lower())
    if file_ext.lower() not in [".xlsx",".xls", ".csv", ".xlsm", ".xltx", ".xltm",]:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file extension does not match the required Excel extension.",
        )

    # Choose appropriate engine based on file extension
    if file_ext.lower() == ".xls":
        engine = "xlrd"
    else:
        engine = "openpyxl"

    df = pd.read_excel(io.BytesIO(file.file.read()), engine=engine)

    # generate a unique table name
    # table_name = "excel_data_" + str(uuid.uuid4())

    df.to_sql(
        "excel_data_temp", con=excelDB.sqlEngine, if_exists="replace", index=False
    )

    # response = query_dataframe_agent(df, query)

    # report_chain = excel_report_chain(df)
    # response = report_chain("Generate a report based on the data provided in the dataframe ")

    return {
        "message": "Excel report generated successfully",
        # "excel report": response["report"],
        # "report_data": report_data,
        # "csv_data": csv_data,
        # "query": response,
    }




# # Convert every column list value into list of strings
# df_dict_result = {col: df[col].tolist() for col in df.columns}