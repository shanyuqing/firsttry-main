import os
import pandas as pd
from openai import OpenAI

# Specify the directory
directory = '/root/firsttry/CMIN/CMIN-US/price/processed'

# List all files in the directory
filenames = os.listdir(directory)
company_list = []
for filename in filenames:
    filename = filename[:-4]
    company_list.append(filename)


client = OpenAI(api_key='sk-Hdf6Ybl_685eMpO2-wbhke_SblUC65NaULFM2cfOkHT3BlbkFJVcE21Wnb_pHC7uBq4bbgEq8GiWNcwuoXVrc7BhIwYA')
# Function to get the correlation between two companies
def query_gpt_for_correlation(sentence):
    systemPrompt = f"""Based on fundamental information, estimate the correlation coefficient of the following stocks. 
    You need to answer with a floating-point number between the range [-1,1], where 1 represents perfect positive correlation, -1 represents perfect negative correlation, and 0 represents no correlation.
    For example: 
                Sample Input: [BAC, BABA]
                Sample Output: Value: -0.22, Explanation:Bank of America (BAC) operates in the financial sector, while Alibaba (BABA) is in e-commerce and technology;BAC's performance is heavily influenced by the U.S. economy and financial markets, while BABA is more tied to the Chinese economy and global e-commerce;While both companies may react to global economic factors, their sensitivity to specific economic policies, such as interest rates or trade policies, differs.
                Sample Input: [TSLA, AAPL]
                Sample Output: Value: 0.25, Explanation: Both Tesla (TSLA) and Apple (AAPL) are tech-driven companies, which means they both tend to move similarly when the broader technology sector is affected by investor sentiment or market conditions;Tesla operates in the electric vehicle and renewable energy space, while Apple is focused on consumer electronics;Both stocks are high-growth companies and are often included in tech-focused investment portfolios. Positive or negative sentiment around tech stocks can lead to both moving in the same direction, contributing to a moderate correlation. 
"""
    prompt = sentence
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": systemPrompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].message.content


n_company =len(company_list)

# define a function to extract the correlation value from the response
def get_value(sample):
    import re
    pattern = r'Value: (-?\d+\.\d+)'
    match = re.search(pattern, sample)
    value= float(match.group(1))
    return value

correlation_value_list = []
company_pairs = []
for i in range(n_company):
    for j in range(i+1):
        if i != j:
            company_pairs.append([company_list[i], company_list[j]])

for i in range(len(company_pairs)):
    company = company_pairs[i][0]
    company2 = company_pairs[i][1]
    sentence = f"Input: [{company}, {company2}]"
    response = query_gpt_for_correlation(sentence)
    with open('data/answer_from_gpt.txt', 'a', encoding= "utf-8") as f:
        f.write(response)
        f.write('\n')
    correlation_value = get_value(response)
    correlation_value_list.append(correlation_value)
    print(sentence, correlation_value)


# Create a DataFrame to store the correlation values
correlation_df = pd.DataFrame(correlation_value_list, columns=['Correlation'])
correlation_df['Company'] = company_pairs
correlation_df = correlation_df[['Company', 'Correlation']]
correlation_df.to_csv('data/generate_correlation.py')



# 清理并提取公司对和相关系数
correlation_df['Company'] = correlation_df['Company'].apply(lambda x: eval(x))  # 将字符串转换为列表
correlation_df['Company1'] = correlation_df['Company'].apply(lambda x: x[0])
correlation_df['Company2'] = correlation_df['Company'].apply(lambda x: x[1])

# 创建矩阵格式的透视表
matrix_df = correlation_df.pivot_table(index='Company1', columns='Company2', values='Correlation')

# 通过镜像矩阵来填补缺失值（因为相关性是对称的）
matrix_df = matrix_df.combine_first(matrix_df.T)

# 对角线填入1（公司与自身的相关性为1）
for company in matrix_df.columns:
    matrix_df.loc[company, company] = 1.0
matrix_df.to_csv('/root/firsttry/data/topology_matrix.csv')