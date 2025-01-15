# Create CSV data.
import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

data = {
    "입력 (X)": [1, 2, 3, 4, 5],
    "정답 (t)": [2, 3, 4, 5, 6]
}

df = pd.DataFrame(data)

document_path = os.path.join(os.path.expanduser('~'), 
                             r"C:\Users\skygr\OneDrive\바탕 화면\롤체 덱 이길 확률 높이는 AI 만들기 project"
                             )
file_path = os.path.join(document_path, "./data-15-input_answer.csv")
df.to_csv(file_path, index = False, encoding='utf-8')
print(f"CSV 파일이 저장되었습니다: {file_path}")
