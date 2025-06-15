import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
from nltk.corpus import stopwords
import nltk
import numpy as np
import json
from PIL import Image


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Лемматизация + частотность

import json
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3 as pymorphy2

# Загрузка данных NLTK (если ещё не загружены)
nltk.download('punkt')
nltk.download('stopwords')

# Инициализация лемматизатора
morph = pymorphy2.MorphAnalyzer()

def load_chat_history(file_path):
    """Загружает JSON-файл истории чата."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_messages(chat_data):
    """Извлекает текст сообщений из JSON."""
    messages = []
    if 'messages' in chat_data:
        for message in chat_data['messages']:
            if isinstance(message.get('text'), str):
                messages.append(message['text'])
            elif isinstance(message.get('text'), list):
                text_parts = []
                for part in message['text']:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                messages.append(' '.join(text_parts))
    return messages

def preprocess_text(text):
    """Очищает текст, токенизирует и лемматизирует слова."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # - пунктуацию
    words = word_tokenize(text, language='russian')
    
    # Удаляем стоп-слова и короткие слова
    stop_words = set(stopwords.words('russian'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Лемматизация
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return lemmas

def count_word_frequency(messages):
    """Подсчитывает частотность лемм."""
    all_lemmas = []
    for msg in messages:
        all_lemmas.extend(preprocess_text(msg))
    return Counter(all_lemmas).most_common()

if __name__ == "__main__":
    file_path = 'result-0.json'  # Путь к JSON-файлу result
    chat_data = load_chat_history(file_path)
    messages = extract_messages(chat_data)
    lemma_freq = count_word_frequency(messages)

    # Вывод топ-20 самых частых лемм
    print("Топ-20 самых частых слов (в начальной форме):")
    for lemma, count in lemma_freq[:20]:
        print(f"{lemma}: {count}")

    # Сохранение в CSV (опционально)
    import csv
    with open('lemma_frequency.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Лемма', 'Частота'])
        writer.writerows(lemma_freq)


# Сборка WORD CLOUD

# Загрузка стоп-слов
nltk.download('stopwords')
stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english'))).union(open('stopw.txt').readlines())
custom_stopwords = {'этот', 'это', 'вот', 'ну', 'да', 'нет', 'ещё', 'уже', 'который', 'весь', 'чтоть','это','то','весь'}
stop_words.update(custom_stopwords)
# print(stop_words)

# Функция очистки текста
def clean_text(text):
    text = str(text).lower()
    # text = re.sub(r'[^а-яёa-z\s]', '', text)
    words = [word for word in text.split() 
             if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Загрузка данных
file_path = 'lemma_frequency.csv'

df = pd.read_csv(file_path)

# Определение текстовой колонки
text_column = 'lemma' if 'lemma' in df.columns else df.columns[0]

# Очистка текста
df['cleaned_text'] = df[text_column]\
    .apply(clean_text)

# Создание частотного словаря
word_freq = {}
for text in df['cleaned_text']:
    for word in text.split():
        word_freq[word] = word_freq.get(word, 0) + 1

decoder = json.decoder.JSONDecoder()
params = decoder.decode(open("param.json").read())

# Генерация облака слов
wordcloud = WordCloud(
    **params,
    stopwords = stop_words
).generate_from_frequencies(word_freq)

# Сохранение и отображение
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
output_path = input('Output path:')
plt.savefig(output_path, dpi=int(input('resolution (dpi):')), bbox_inches='tight')
plt.show()