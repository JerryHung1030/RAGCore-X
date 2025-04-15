import pandas as pd
import re
import random
import unicodedata

ALL_CASE_FILENAME = "./data/fraud_case_165dashboard.csv"
CASE_TOTAL_NUM = 100
NUM_PER_CASE = [7, 5]

TYPES = [
    "假投資",
    "假交友",
    "假中獎",
    "假求職",
    "假檢警",
    "釣魚簡訊",
    ["騙取金融帳戶", "騙取金融卡片"],
    "猜猜我是誰",
    "網路購物",
    "假買家",
    "假交友",
    "假借銀行貸款",
    "假廣告",
    "色情應召詐財",
    "虛擬遊戲"
]


def clean_case_title(text):
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    text = re.sub(r'\s+', '', text)
    return text


def flatten_types(types_list):
    type_map = {}
    for entry in types_list:
        if isinstance(entry, list):
            for keyword in entry:
                type_map[keyword] = entry[0]
        else:
            type_map[entry] = entry
    return type_map


def get_case_type(title, type_keywords):
    for keyword in type_keywords:
        if keyword in title:
            return keyword
    return "其他"


def select_cases(input_file, output_file):
    all_cases_df = pd.read_csv(input_file)
    all_cases_df["CaseTitle"] = all_cases_df["CaseTitle"].apply(clean_case_title)
    
    type_keywords = flatten_types(TYPES)
    all_cases_df["Type"] = all_cases_df["CaseTitle"].apply(lambda x: get_case_type(x, type_keywords))
    
    result_df = pd.DataFrame()
    
    for group_name, group_df in all_cases_df.groupby("Type"):
        group_size = len(group_df)
        if group_name == "其他":
            sample_n = min(NUM_PER_CASE[1], group_size)
        else:
            sample_n = min(NUM_PER_CASE[0], group_size)
        sampled_df = group_df.sample(n=sample_n, random_state=random.randint(0, 9999))
        result_df = pd.concat([result_df, sampled_df], ignore_index=True)
        print(f"{group_name} - picked {sample_n} cases（from {group_size} cases）")
    
    print(f"\nResult: {len(result_df)} cases in total")
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
  
  
if __name__ == "__main__":
    part_case_filename = f"./data/fraud_case_165dashboard_part{CASE_TOTAL_NUM}.csv"
    select_cases(ALL_CASE_FILENAME, part_case_filename)
