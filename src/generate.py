import pandas as pd
import json

def excel_to_jsonl(excel_file, text_column_name, output_file):
    """
    從 Excel 檔讀取指定欄位并輸出為 JSONL 格式。

    :param excel_file: Excel 檔名 (路徑)
    :param text_column_name: 欲讀取的文字欄位名稱，例如 "使用者輸入"
    :param output_file: 輸出 JSONL 檔名
    """
    # 讀取 Excel 檔
    df = pd.read_excel(excel_file)

    # 建立或覆寫輸出檔
    with open(output_file, 'w', encoding='utf-8') as f:
        # 逐列處理
        for idx, row in df.iterrows():
            # 取得文字內容
            user_text = row[text_column_name]

            # 組成目標格式
            data = {
                "id": str(idx + 1),   # 依序產生 ID，可視需求調整
                "text": str(user_text)
            }

            # 寫入 JSONL，每筆資料一行
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # Excel 檔名 (請注意實際副檔名)
    excel_file_name = "input_data_c70_k10.xlsx"
    
    # 欲讀取的欄位名稱
    column_name = "使用者輸入"
    
    # 輸出 JSONL 檔名
    output_jsonl = "fraud.jsonl"
    
    # 執行轉換
    excel_to_jsonl(excel_file_name, column_name, output_jsonl)
    
    print(f"已將 '{excel_file_name}' 中的 '{column_name}' 欄位轉換為 JSONL 格式，輸出至 '{output_jsonl}'")
