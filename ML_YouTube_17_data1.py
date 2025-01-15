# Create CSV data.
import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

data = {
    "공부시간 (X)":     [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "Fail/Pass (t)":   [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

document_path = os.path.join(os.path.expanduser('~'), 
                             r"C:\Users\skygr\OneDrive\바탕 화면\롤체 덱 이길 확률 높이는 AI 만들기 project"
                             )

file_path = os.path.join("./data-17-time_failpass.csv")
df.to_csv(file_path, index=False, encoding='utf-8')
print(f"CSV 파일이 저장되었습니다: {file_path}")
