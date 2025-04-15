import requests
import os
import csv
from datetime import datetime, timedelta
import pytz
import pandas as pd
import time

CSV_FILENAME = "./data/fraud_case_165dashboard.csv"
URL = "https://165dashboard.tw/CIB_DWS_API/api/CaseSummary/GetCaseSummaryList"
NUMPERPAGE = 100

HEADERS = {
    "Content-Type": "application/json; charset=utf-8", 
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

FIELD_NAMES = ['Id', 'CaseDate', 'CityName', 'CityId', 'Summary', 'CaseTitle']


def get_last_case_date():
    """Check for csv file and return the date of the latest record (last row)"""
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME)
        if not df.empty:
            last_row_date = df.iloc[-1]['CaseDate']
            return datetime.strptime(last_row_date, "%Y-%m-%dT%H:%M:%SZ")
    return False
    

def get_yesterday():
    """Return the date of yesterday"""
    today = datetime.now(pytz.timezone("Asia/Taipei")) - timedelta(days=1)
    return today.astimezone(pytz.utc).replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=None)


def fetch_case_data(case_date=None, page=1):
    """Fetch data for a given date"""
    payload = {
        "UsingPaging": True,
        "NumberOfPerPage": NUMPERPAGE,
        "PageIndex": page,
        "SortOrderInfos": [{"SortField": "CaseDate", "SortOrder": "ASC"}],
        "SearchTermInfos": [],
        "Keyword": None,
        "CityId": None,
        "CaseDate": case_date.strftime("%Y-%m-%dT%H:%M:%SZ") if case_date else None
    }
    
    response = requests.post(URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return None
    
    
def fetch_fraud_cases(date=None):
    # Case 1: No existing csv file or case data
    if not date:
        data = fetch_case_data(date)    
        if data:
            body = data.get("body", {})
            total_pages = body.get("TotalPages", 0)
            print("Total pages: ", total_pages)
            
            with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8-sig') as file:
                csv_writer = csv.DictWriter(file, fieldnames=FIELD_NAMES)
                csv_writer.writeheader()
            
                for page_id in range(1, total_pages + 1):
                    time.sleep(2)
                    print("Crawling page ", page_id)
                    data = fetch_case_data(date, page_id)
                    body = data.get("body", {})
                    detail = body.get("Detail", [])
                    
                    for case in detail:
                        csv_writer.writerow(case)
    # Case 2: csv file exists, append new data                
    else:       
        end_date = get_yesterday()
        start_date = date + timedelta(days=1)
        print("Start date (latest data): ", start_date)
        print("End date (previous day): ", end_date)
        
        current_date = start_date
        if current_date <= end_date:
            with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8-sig') as file:
                csv_writer = csv.DictWriter(file, fieldnames=FIELD_NAMES)
            
                while current_date <= end_date:
                    print("------")
                    print("Crawling date ", current_date)
                    data = fetch_case_data(current_date)    
                    if data:
                        body = data.get("body", {})
                        if body.get("Detail", []) == []:
                            current_date += timedelta(days=1)
                            continue
                        total_pages = body.get("TotalPages", 0)
                        print("Total pages: ", total_pages)
                        
                        for page_id in range(1, total_pages + 1):
                            time.sleep(2)
                            print("Crawling page ", page_id)
                            data = fetch_case_data(current_date, page_id)
                            body = data.get("body", {})
                            detail = body.get("Detail", [])
                            
                            for case in detail:
                                csv_writer.writerow(case)
                                    
                    current_date += timedelta(days=1)
    
    
if __name__ == "__main__":
    last_date = get_last_case_date()
    print("Last Fetched Date:", last_date)
    fetch_fraud_cases(last_date)
