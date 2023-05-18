import streamlit as st
import numpy as np
import pandas as pd
import csv
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import re
import gdown
import os

# モデルファイルをダウンロード
file_id = "1a2VIjUpt6sWIkntMZsmU3ewUXUl0-XgH"
url = f"https://drive.google.com/uc?id={file_id}"
output = "trained_model"
gdown.download(url, output, quiet=False)
    
# 施設・サービスに関する検索準備
# FAQリストを読み込む
faq_file = 'QA.csv'
with open(faq_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    faqs = list(reader)

# ALBERTモデルとTokenizerを事前に読み込む
tokenizer = AutoTokenizer.from_pretrained("ken11/albert-base-japanese-v1")
model = AutoModelForMaskedLM.from_pretrained("ken11/albert-base-japanese-v1")

# FAQリストの質問と入力された質問の類似度を計算する関数
def get_similarity(question1, question2):
    encoded_question1 = tokenizer.encode_plus(question1, add_special_tokens=True, return_tensors='pt')
    encoded_question2 = tokenizer.encode_plus(question2, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs_question1 = model(**encoded_question1)[0][:, 0, :]
        outputs_question2 = model(**encoded_question2)[0][:, 0, :]
    cosine_similarity = torch.nn.functional.cosine_similarity(outputs_question1, outputs_question2, dim=1)
    return cosine_similarity.item()

# 応答メッセージを作成する関数
def reply_message(user_input):
    max_similarity = 0
    best_answer = ''
    for faq in faqs:
        similarity = get_similarity(user_input, faq[0])
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = faq[1]
    return best_answer 

def reply_message_2(user_input_2):
    threshold = 0.5  # 適切な値を設定する
    # カテゴリのリストを定義
    categories = ['東京限定', '商品名・メーカー名検索', '仏事', 'おもたせ', '日持ち']
    # 商品検索に関する前処理
    # まずは質問文から羽田OR空港の文字があるか確認する

    # 商品リストをcsv形式で読み込む
    df = pd.read_csv('商品リスト.csv')

    # 入力文に空港もしくは羽田が入っていれば「羽田限定」列にフィルター
    if '羽田' in user_input_2 or '空港' in user_input_2:
        df1 = df[df['羽田限定'] == 1]
    else:
        df1 = df

    # 数字が入っていてかつ100以上なら「金額」列にフィルター、100未満であれば「入数」列にフィルター
    numbers = re.findall(r'\d+', user_input_2)  # 入力文字列から数値を抽出
    if numbers:
        df2 = df1
        for number in numbers:
            number = int(number)  # 数値を取得
            if number >= 100:
                df2 = df2[df2['金額'] <= number]
            else:
                df2 = df2[df2['入数'] >= number]
    else:
        df2 = df1

    # モデルファイルをダウンロード
    model_file_id = "1a2VIjUpt6sWIkntMZsmU3ewUXUl0-XgH"
    model_url = f"https://drive.google.com/uc?id={model_file_id}"
    model_output = "trained_model"
    gdown.download(model_url, model_output, quiet=False)
    
    # ALBERTモデルとTokenizerを事前に読み込む
    tokenizer2 = AutoTokenizer.from_pretrained(model_output)
    model2 = AutoModelForSequenceClassification.from_pretrained(model_output)

    # 質問をトークン化してエンコーディング
    encoded_input_2 = tokenizer2(user_input_2, padding=True, truncation=True, return_tensors="pt")

    # カテゴリの予測
    with torch.no_grad():
        model_output = model2(**encoded_input_2)
        logits = model_output.logits
        probabilities = torch.sigmoid(logits)
        predicted_categories = [categories[i] for i, prob in enumerate(probabilities[0]) if prob > threshold]

    # 絞り込み条件に応じて、データフレームをフィルターする
    if predicted_categories:
        filtered_df = df2[df2[predicted_categories].any(axis=1)]
    else:
        filtered_df = df2

    # 商品リストの特定の列を表示する
    selected_columns = ['メーカー名', '商品名', '金額', '入数']  # 表示したい列のリスト
    filtered_df_selected = filtered_df[selected_columns]

    return filtered_df_selected

# 施設・サービスについての検索画面
def page_service():
    st.title('施設・サービスについての確認')
    
    user_input = st.text_input('質問を入力してください','')

    reply_text = reply_message(user_input)
    if st.button(label='確認'):
        st.write(reply_text)

# 商品検索についての検索画面
def page_item():
    st.title('商品検索')

    user_input_2= st.text_input('検索条件を入力してください','')
    
    reply_text_2 = reply_message_2(user_input_2)
    if st.button(label='検索'):
        st.write(reply_text_2)selected

# サイドバーの設定
st.sidebar.header('接客支援アプリ')
purpose = st.sidebar.selectbox('検索項目を選択してください', ('施設・サービスについての確認', '商品検索'))

if purpose == '施設・サービスについての確認':
    page_service()

if purpose == '商品検索':
    page_item()

else:
    pass
