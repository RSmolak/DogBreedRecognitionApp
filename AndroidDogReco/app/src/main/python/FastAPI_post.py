import requests

def main(filePath):

    file = open(filePath, 'rb')
    img = requests.post("https://dog-breed-recognition-edkh2yjooq-lz.a.run.app/img-to-string",
                         files={"file": file})
    return img.text
