import pandas as pd
import re
from pathlib import Path

def clean_and_tokenize(text):
    """ 清洗并分词 """
    if not isinstance(text, str):
        return []
    text = text.lower()
    # 移除 URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 移除邮箱
    text = re.sub(r'\S+@\S+', '', text)
    # 移除数字
    text = re.sub(r'\d+', '', text)
    # 只保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    # 过滤长度不在 [2, 20] 的词
    words = [w for w in words if 2 <= len(w) <= 20]
    return words


def process_dataset():
    """ 主函数：清洗并保存为 cleaned_messages.csv """
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / 'messages.csv'
    output_path = project_root / 'data' / 'cleaned_messages.csv'

    if not input_path.exists():
        raise FileNotFoundError(f"原始数据文件未找到: {input_path}")

    # ✅ 正确方式：不指定 names，让 pandas 自动读取第一行为 header
    df = pd.read_csv(
        input_path,
        header=0,  # 明确告诉 pandas 第一行是 header
        skip_blank_lines=True,
        skipinitialspace=True
    )

    # 提取 message 和 label 列
    raw_messages = df['message'].fillna('')
    labels = df['label']

    # 清洗每条消息
    cleaned_messages = []
    for msg in raw_messages:
        words = clean_and_tokenize(msg)
        cleaned_text = ' '.join(words)
        if cleaned_text == '':
            cleaned_text = 'aaa'  # 空内容填充
        cleaned_messages.append(cleaned_text)

    # 构建新 DataFrame
    cleaned_df = pd.DataFrame({
        'message': cleaned_messages,
        'label': labels
    })

    # 保存时只写入列名一次（默认行为）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"Data cleaning completed! Saved to:{output_path}")
    print(f"Total sample size {len(cleaned_df)}")
    print(f"Number of empty messages (now filled with 'aaa'):{sum(1 for m in cleaned_messages if m == 'aaa')}")


if __name__ == "__main__":
    process_dataset()