#이미지 이름 바꾸는 프로그램

import glob
import os

#디렉토리안의 확장자를 가지고 있는 파일들을 찾아준다.
file_list = glob.glob("*.jpg") 

#어떤 번호부터 숫자를 메길지 선택한다.
prefix = 101
count = 0

for name in file_list:
    
    #카운트할 숫자를 1씩 증가시킨다. 파이썬의 for문 사용의 특성상 이렇게 사용해야 한다.
    count = count + 1

    #mp3파일의 가장앞의 숫자(가지고 있는 파일을 기준으로)2자리를 슬라이싱 해준다.
    name_change =  "images-" + str((prefix - 1) + count) + ".jpg"
  
    #파일이름을 변환시킨다.
    os.rename(name, name_change)
