import requests

file = open("322868_1100-800x825.jpg", 'rb')
img = requests.post("https://dog-breed-recognition-edkh2yjooq-lz.a.run.app/img-to-string",
                files={"file": file})
print(img.text)