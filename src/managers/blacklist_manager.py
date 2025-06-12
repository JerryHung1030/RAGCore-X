# src/managers/blacklist_manager.py
import re
import whois
import numpy as np
import pandas as pd

# from typing import List, Tuple, Optional, Dict
from urllib.parse import urlparse
from datetime import datetime, timezone
from dateutil import parser
import onnxruntime as rt

LINE_RE_LIST = [
    {'re': "(lineid)([^@])?([@]?[a-z0-9-_+]{4,})", 'idx': 2},
    {'re': "(line).+?(id|帳號)([^@a-z0-9])?([@]?[a-z0-9-_+*]{4,})", 'idx': 3}
]
URL_RE_LIST = [
    {'re': r"(https?:\/\/|www\.)([^\s\"\-\'\.\/<>]+)\.([a-zA-Z0-9\.\/#=]+)"}
]


class BlacklistManager:
    LINE_RE_LIST = [
        {'re': "(lineid)([^@])?([@]?[a-z0-9-_+]{4,})", 'idx': 2},
        {'re': "(line).+(id|帳號)([^@a-z0-9])?([@]?[a-z0-9-_+*]{4,})", 'idx': 3}
    ]
    URL_RE_LIST = [
        {'re': r"(https?:\/\/|www\.)([^\s\"\-\'\.\/<>]+)\.([a-zA-Z0-9\.\/#=]+)"}
    ]
    
    def __init__(self, model_path: str = None, lineid_blacklist_path: str = None, domain_blacklist_path: str = None):
        self.model_path = model_path or "src/data/model/GradientBoostingClassifier_model.onnx"
        self.lineid_blacklist_path = lineid_blacklist_path or "src/data/database/lineid_database.xlsx"
        self.domain_blacklist_path = domain_blacklist_path or "src/data/database/url_bad_database.xlsx"
        self.lineid_blacklist = self._load_blacklist(self.lineid_blacklist_path, "id")
        self.domain_blacklist = self._load_blacklist(self.domain_blacklist_path, "url")
    
    # ---------------------------------------
    #  main function to analyze a paragraph
    # ---------------------------------------
    def analyze(self, paragraph: str) -> dict:
        lineid_lst, lineid_results = self.check_line_ids(paragraph)
        url_lst, url_results = self.check_urls(paragraph)
        
        results = {
            "line_id_list": lineid_lst,
            "url_list": url_lst,
            "line_id_details": lineid_results,
            "url_details": url_results
        }
        return results

    # load blacklist to dataframe from excel file
    def _load_blacklist(self, path: str, column: str) -> list[str]:
        try:
            df_blacklist = pd.read_excel(path)
            return df_blacklist[column].dropna().astype(str).tolist()
        except (FileNotFoundError, KeyError):
            print(f"[Error] Failed to load {column} blacklist from {path}")
            return []
    
    # -----------------------------
    # Line ID Logic
    # -----------------------------
    """
    Find line ids in a given paragraph
    """
    def parse_lineid(self, paragraph: str) -> list[str]:
        found_ids = []
        for id_pattern in LINE_RE_LIST:
            match_lst = re.findall(id_pattern['re'], paragraph, re.IGNORECASE)
            if match_lst:
                for match in match_lst:
                    found_id = match[id_pattern['idx']]
                    if found_id.lower() not in ["", "line", "id", "https", "http", "7-11"]:
                        found_ids.append(found_id)
        return found_ids    

    """
    Check the line ids in the found list: 
        1. compare with lineid blacklist
    """
    def check_line_ids(self, paragraph: str) -> tuple[list[str], list[dict]]:
        lineid_results = []
        lineid_lst = self.parse_lineid(paragraph)
        
        # Check if the line ids are in the blacklist
        if lineid_lst:
            for lineid in lineid_lst:
                if lineid in self.lineid_blacklist:
                    result = 0      # lineid in blacklist
                else:
                    result = 1      # lineid not in blacklist
                    
                # Add result to the list
                lineid_results.append({
                    "id": lineid,
                    "result": result,
                    "source": "dataset"
                })
            
        return lineid_lst, lineid_results
    
    # -----------------------------
    # URL Logic
    # -----------------------------
    """
    Find urls in a given paragraph
    """
    def parse_url(self, paragraph: str) -> list[str]:
        found_urls = []
        for url_pattern in URL_RE_LIST:
            match_lst = re.findall(url_pattern['re'], paragraph, re.IGNORECASE)
            if match_lst:
                for match in match_lst:
                    found_urls.append(f"{match[0]}{match[1]}.{match[2]}")
        return found_urls
  
    """
    Check the urls in the found list: 
        1. compare with url blacklist
        2. use model to predict probability
    """
    def check_urls(self, paragraph: str) -> tuple[list[str], list[dict]]:      
        url_results = []
        url_lst = self.parse_url(paragraph)
        
        # Check if the urls are in the blacklist
        if url_lst:
            for url in url_lst:
                try:
                    # Parse url to get domain
                    parsed_url = urlparse(url)
                    domain = (parsed_url.hostname or url).lower().strip()
                    
                    if not domain or "." not in domain:
                        print(f"[ERROR] Bad domain parsed: {url}")
                        url_results.append({
                            "status": 0,
                            "url": url,
                            "error_code": 0,
                            "problem": "Failed to parse url domain."
                        })
                        continue
                    # 1. Check if the domain is in the blacklist
                    if domain in self.domain_blacklist:
                        url_results.append({
                            "status": 1,
                            "url": url,
                            "result": 0,
                            "source": "dataset",
                            "scam_probability": 1.0,
                            "level": "HIGH"
                        })
                    # 2. Use model to predict probability
                    else:
                        """
                        Return cases:
                        - Failed to get whois data
                        - Empty or invalid WHOIS data
                        - Success, return prediction result
                        """
                        result = self.url_analysis(domain)
                        if result["status"] == 0:
                            url_results.append({
                                "status": 0,
                                "url": url,
                                "error_code": result.get("error_code", 99),
                                "error_msg": result.get("error_msg", "Unknown error")
                            })
                        else:
                            url_results.append({
                                "status": 1,
                                "url": url,
                                "result": result["result"],
                                "source": "model",
                                "scam_probability": result["scam_probability"],
                                "level": result["level"]
                            })
                except Exception as e:
                    print(f"[ERROR] Error parsing url: {url}", e)
                    url_results.append({
                        "status": 0,
                        "url": url,
                        "problem": "Failed to parse url domain."
                    })
        
        return url_lst, url_results
    """
    Analyze url scam probability:
        1. Get whois data
        2. Process whois data
        3. Use model to predict
    """
    def url_analysis(self, domain: str):       # url not from dataset has no found date
        try:
            whois_data = whois.whois(domain)
            
            if not whois_data or isinstance(whois_data, str):
                return {
                    "status": 0,
                    "error_code": 1,
                    "error_msg": "Failed to call whois data."
                }

            # Check if whois data is valid
            creation_date_raw = whois_data.get("creation_date")
            if creation_date_raw is None:
                if creation_date_raw is None:
                    return {
                        "status": 0,
                        "error_code": 2,
                        "error_msg": "Empty or invalid whois data."
                    }
            
            # Get domain tlds
            tlds = self.get_tlds(domain)

            # Get and process all time related data needed
            creation_date = self.extract_datetime(creation_date_raw, "min")
            reference_date = datetime.now(timezone.utc).date()
            expiration_date = self.extract_datetime(whois_data.get("expiration_date"), "max")

            # Calculate time related columns
            register_time = (reference_date - creation_date).days if creation_date else 0
            remaining_time = (expiration_date - reference_date).days if expiration_date else 0
            total_life_time = (expiration_date - creation_date).days if expiration_date else 0
            register_country = whois_data.get("country", "Unknown")
            
            # Determine existence columns
            if whois_data.get("org") or whois_data.get("organization"):
                org_exist = True
            if whois_data.get("dnssec") is not None:
                dnssec = whois_data.get("dnssec")

            input_data = {
                "tlds": np.array([[tlds]]),
                "dnssec": np.array([[dnssec]]),
                "register_country": np.array([[register_country]]),
                "org_exist": np.array([[str(org_exist)]], dtype=object),
                "register_time": np.array([[register_time]], dtype=np.float32),
                "remaining_time": np.array([[remaining_time]], dtype=np.float32),
                "total_life_time": np.array([[total_life_time]], dtype=np.float32),
            }

            sess = rt.InferenceSession(self.model_path)
            outputs = sess.run(None, input_data)
            pred_label = outputs[0][0]
            pred_prob_bad = outputs[1][0][0]
            
            if pred_prob_bad > 0.75:
                scam_level = "HIGH"
            elif pred_prob_bad > 0.5:
                scam_level = "MEDIUM"
            else:
                scam_level = "LOW"
            
            return {
                "status": 1,
                "result": int(pred_label),
                "scam_probability": pred_prob_bad,
                "level": scam_level
            }
        except Exception as e:
            return {
                "status": 0,
                "error_code": 99,
                "error_msg": f"Failed to load model: {str(e)}"
            }
    
    # Get domain tlds
    def get_tlds(self, domain: str):
        return domain.split(".")[-1].lower() if "." in domain else ""
    
    # Define the date to be used from a list of dates
    def extract_datetime(self, dates, pick='min'):
        try:
            if not dates:
                return None

            if isinstance(dates, list):
                parsed_dates = []
                for d in dates:
                    dt = d if isinstance(d, datetime) else parser.parse(d)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    parsed_dates.append(dt)
                dt = min(parsed_dates) if pick == 'min' else max(parsed_dates)
            else:
                dt = dates if isinstance(dates, datetime) else parser.parse(dates)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

            return dt.date()
        except Exception as e:
            return None