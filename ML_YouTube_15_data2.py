# Create CSV data.
import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

data = {
    "X1": [73, 93, 89, 96, 73, 53, 69, 47, 87, 79, 69, 70, 93, 79, 70, 93, 78, 81, 88, 78, 82, 86, 78, 76, 96],
    "X2": [80, 88, 91, 98, 66, 46, 74, 56, 79, 70, 70, 65, 95, 80, 73, 89, 75, 90, 92, 83, 86, 82, 83, 83, 93],
    "X3": [75, 93, 90, 100, 70, 55, 77, 60, 90, 88, 73, 74, 91, 73, 78, 96, 68, 93, 86, 77, 90, 89, 85, 71, 95],
    "t" : [152, 185, 180, 196, 142, 101, 149, 115, 175, 164, 141, 141, 184, 152, 148, 192, 147, 183, 177, 159, 177, 175, 175, 149, 192]
}

df = pd.DataFrame(data)

document_path = os.path.join(os.path.expanduser('~'),
                             r"C:\Users\skygr\OneDrive\바탕 화면\롤체 덱 이길 확률 높이는 AI 만들기 project"
                             )

file_path = os.path.join(document_path, "./data-15-test_score.csv")
df.to_csv(file_path, index=False, encoding='utf-8')
print(f"CSV 파일이 저장되었습니다: {file_path}")
