import csv
import datetime
import pytz
import requests
import time


CSV_FILENAME = "./data/fraud_case_165dashboard_20250409.csv"        # change to test date
URL = "https://165dashboard.tw/CIB_DWS_API/api/CaseSummary/GetCaseSummaryList"
NUMPERPAGE = 100

HEADERS = {
    "Content-Type": "application/json; charset=utf-8", 
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

FIELD_NAMES = ['Id', 'CaseDate', 'CityName', 'CityId', 'Summary', 'CaseTitle']


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
    current_date = date
    data = fetch_case_data(current_date)    
    if data:
        with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8-sig') as file:
            csv_writer = csv.DictWriter(file, fieldnames=FIELD_NAMES)
            body = data.get("body", {})
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
                    
    
if __name__ == "__main__":
    date = datetime(2025, 4, 9, tzinfo=pytz.utc)        # change to test date
    fetch_fraud_cases(date)
