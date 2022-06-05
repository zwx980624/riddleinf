import urllib
import urllib.parse
import urllib.request
import requests


def kmcha_same(word):
    page=requests.get('https://kmcha.com/similar/' + urllib.parse.quote(word))
    page = page.text
    # print(page)
    similar = page.split("strong>%s的近义词"%(word))[-1]

    similar = similar.split("<p>")[1].split("</p>")[0]
    # print(similar)
    ls = similar.split("<span>")
    ret = []
    for i, dt in enumerate(ls):
        if i == 0:
            continue
        ret.append(dt.split("</span>")[0])
    return ret

def kmcha_similar(word):
    page=requests.get('https://kmcha.com/similar/' + urllib.parse.quote(word))
    page = page.text
    # similar = page.split("strong>%s的近义词"%(word))[-1]
    similar = page.split("strong>%s的相似词"%(word))[-1]

    similar = similar.split("<p>")[1].split("</p>")[0]
    # print(similar)
    ls = similar.split("<span>")
    ret = []
    for i, dt in enumerate(ls):
        if i == 0:
            continue
        ret.append(dt.split("</span>")[0])
    return ret

if __name__ == "__main__":
    kmcha_same("卧龙")
    kmcha_similar("卧龙")