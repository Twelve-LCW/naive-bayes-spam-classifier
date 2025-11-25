import pandas as pd
import re
import os
from pathlib import Path

def clean_and_tokenize(text):
    """
    清洗文本并返回分词列表
    """
    if not isinstance(text, str):  # 防止 NaN 或非字符串
        text = ""
    # 转小写
    text = text.lower()
    
    # 移除 URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 移除电子邮件地址（可选）
    text = re.sub(r'\S+@\S+', '', text)
    
    # 移除数字
    text = re.sub(r'\d+', '', text)
    
    # 移除标点符号和特殊字符，只保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 分词
    words = text.split()
    
    # 过滤无效词：长度 < 2 或 > 20
    words = [w for w in words if 2 <= len(w) <= 20]
    
    return words


def process_dataset():
    """
    读取原始 CSV,清洗 message 列，并保存为新 CSV
    """
    # 获取当前脚本所在目录的父目录（即项目根目录）
    project_root = Path(__file__).parent.parent
    
    input_csv = project_root / 'data' / 'messages.csv'
    output_csv = project_root / 'data' / 'cleaned_messages.csv'

    # 检查输入文件是否存在
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # 读取 CSV（无 header，所以列名是第一行）
    df = pd.read_csv(input_csv, header=None)

    # 假设列顺序为：subject, message, label
    # 所以第0列=subject, 第1列=message, 第2列=label
    message_col = df.iloc[:, 1]  # 用于清洗
    label_col = df.iloc[:, 2]    # 标签（0/1）

    # 清洗每条消息
    cleaned_messages = []
    for msg in message_col:
        cleaned_words = clean_and_tokenize(msg)
        cleaned_text = ' '.join(cleaned_words)
        cleaned_messages.append(cleaned_text)

    # 创建新 DataFrame
    cleaned_df = pd.DataFrame({
        'message': cleaned_messages,
        'label': label_col
    })
    cleaned_df=cleaned_df[1:]
    # 确保输出目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 保存为新 CSV（带 header）
    cleaned_df.to_csv(output_csv, index=False, header=True)

    print(f"Data cleaning completed! Saved to:{output_csv}")


if __name__ == "__main__":
    process_dataset()