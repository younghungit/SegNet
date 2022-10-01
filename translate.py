from googletrans import Translator
from papago import Translator as Translator_papago
import requests
import getopt

def translate(trans, text, source, target):
    if text == '': return
    if source == 'en':
        text = check_text(text.replace('\n', ' ').lower())
    elif source == 'ko':
        text = text.replace('\n', ' ')
    # print(text)
    try:
        translated_text = translate_papago(text, source=source, target=target)
        print('finish translating with papago')
    except:
        translated_text = translate_google(trans, text, dest=target)
        print('finish translating with google')

    return translated_text


def translate_papago(text, source='en', target='ko', honorific='true'):
    request_url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {"X-Naver-Client-Id": "mQAuRlxsDfzXSmouEyGc", "X-Naver-Client-Secret": "07j70H_j_U"}
    params = {"honorific": honorific, "source": source, "target": target, "text": text}
    response = requests.post(request_url, headers=headers, data=params)
    result = response.json()
    return result['message']['result']['translatedText']


def translate_google(trans, text, dest='ko'):
    result = trans.translate(text, dest=dest)
    return result.text


def check_text(text):
    for char in text:
        if char not in 'abcdefghijklmnopqrstuvwxyz1234567890,.?!()[]{}\'- ':
            text = text.replace(char, '')
    text = text.replace('1', 'I')
    text = text.replace('yoo', 'you')
    ### make more replacement
    return text


if __name__ == "__main__":
    text = 'How are you?'
    source = 'en'
    target = 'ko'

    trans = Translator()
    translated_text = translate(trans, text, source, target)
    print(text, ': ', translated_text)