import pandas as pd
import re
from pathlib import Path

def clean_and_tokenize(text):
    """ Clean and segment words """
    if not isinstance(text, str):
        return []
    text = text.lower()
    # Remove URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove email
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    # Filter words whose length is not in the range [2, 20].
    words = [w for w in words if 2 <= len(w) <= 20]
    return words


def process_dataset():
    """ Main function: Clean and save the messages as cleaned_messages.csv """
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / 'messages.csv'
    output_path = project_root / 'data' / 'cleaned_messages.csv'

    if not input_path.exists():
        raise FileNotFoundError(f"Original data file not found: {input_path}")

    df = pd.read_csv(
        input_path,
        header=0,  #The first line is the header.
        skip_blank_lines=True,
        skipinitialspace=True
    )

    # Extract message and label columns
    raw_messages = df['message'].fillna('')
    labels = df['label']

    # Clean each message
    cleaned_messages = []
    for msg in raw_messages:
        words = clean_and_tokenize(msg)
        cleaned_text = ' '.join(words)
        if cleaned_text == '':
            cleaned_text = 'aaa'  # Fill empty content ('aaa' alreay exists,not only one)
        cleaned_messages.append(cleaned_text)

    # Create a new DataFrame
    cleaned_df = pd.DataFrame({
        'message': cleaned_messages,
        'label': labels
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    print(f"Data cleaning completed! Saved to:{output_path}")
    print(f"Total sample size {len(cleaned_df)}")
    print(f"Number of empty messages (now filled with 'aaa'):{sum(1 for m in cleaned_messages if m == 'aaa')}")


if __name__ == "__main__":
    process_dataset()