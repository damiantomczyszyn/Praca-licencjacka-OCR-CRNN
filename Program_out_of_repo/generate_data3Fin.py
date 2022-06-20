import numpy as np
import string
from PIL import Image,ImageFont,ImageDraw
import argparse
import random
import os
import imgaug.augmenters as iaa
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('--n_samples',type=int,default=50)
parser.add_argument('--word_type',default='all')
parser.add_argument('--rotation_degree',type=int, default=0)
args=parser.parse_args()

global gray_back
kernel=np.ones((2,2),np.uint8)
kernel2=np.ones((1,1),np.uint8)
punclist='.?,()":'

#Character sets to choose from.
smallletters=string.ascii_lowercase
capitalletters=string.ascii_uppercase
digits=string.digits
alll=smallletters+capitalletters+digits+punclist




#Base backgound.
backfilelist=os.listdir('./background/')
backgroud_list=[]

for bn in backfilelist:
    fileloc='./background/'+bn
    backgroud_list.append(Image.fromarray(cv2.imread(fileloc,0)))


#Different fonts to be used.
fonts_list=os.listdir('./fonts/')
fonts_list=['./fonts/'+f for f in fonts_list]

#Lengths of the words.
word_lengths=[]
for l in range(1,21):
    word_lengths.append(l)

#Font size.
font_size=[]
for l in range(10,30):
    font_size.append(l)


file_counter=0

def random_brightness(img):
    img=np.array(img)
    brightness=iaa.Multiply((0.6,1.1))
    img=brightness.augment_image(img)
    return img

def dilation(img):
    img=np.array(img)
    img=cv2.dilate(img,kernel2,iterations=1)
    return img

def erosion(img):
    img=np.array(img)
    img=cv2.erode(img,kernel,iterations=1)
    return img

def blur(img):
    img=np.array(img)
    img=cv2.blur(img,ksize=(3,3))

def fuse_gray(img):
    img=np.array(img)
    ht,wt=img.shape[0],img.shape[1]
    gray_back=cv2.imread('gray_back.jpg',0)
    gray_back=cv2.resize(gray_back,(wt,ht))

    blended=cv2.addWeighted(src1=img,alpha=0.8,src2=gray_back,beta=0.4,gamma=10)
    return blended



def random_transformation(img):
    if np.random.rand()<0.5:
        img=fuse_gray(img)
    elif np.random.rand()<0.5:
        img=random_brightness(img)
    elif np.random.rand()<0.5:
        img=dilation(img)
    elif np.random.rand()<0.5:
        img=erosion(img)

    else:
        img=np.array(img)
    return Image.fromarray(img)
from random import seed
from random import randint  
seed(1) 
def rand_pad():
	
    value = randint(3, 6)
    return value

import shutil
from pathlib import Path

dirpath = Path('images')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

Path("images").mkdir(parents=True, exist_ok=True)

file=open('annotation.txt','w')

file_counter=0
max_label_len = 0

for _ in range(args.n_samples):

    back_c=random.choice(backgroud_list).copy()
    start_cap=random.choice(capitalletters)
    filename=''.join([random.choice(smallletters) for c in range(random.choice([5,6,7,8,9,10,11]))])
    font=ImageFont.truetype(random.choice(fonts_list),size=random.choice(font_size))
    if args.word_type=='lowercase':
        word=''.join([random.choice(smallletters) for b in range(random.choice(word_lengths))])
    elif args.word_type=='uppercase':
        word=''.join([random.choice(capitalletters) for b in range(random.choice(word_lengths))])
    elif args.word_type=='firstcapital':
        word=''.join([random.choice(smallletters) for b in range(random.choice(word_lengths)-1)])
        word=start_cap+word
    elif args.word_type=='digits':
        word=''.join([random.choice(digits) for b in range(random.choice(word_lengths))])
    elif args.word_type=='all':								#moja zmiana
         word=''.join([random.choice(alll) for b in range(random.choice(word_lengths))])
    elif args.word_type=='punctuation':
        word=''.join([random.choice(smallletters) for b in range(random.choice(word_lengths))])
        word=word+str(random.choice(punclist))
    else:
        raise Exception("Invalid word choice.")
    w,h=font.getsize(word)[0],font.getsize(word)[1]
    
            # compute maximum length of the text
    if len(word) > max_label_len:
        max_label_len = len(word)
    
    if args.rotation_degree!=0:
		width, height = font.getsize(word)
		image2 = Image.new('RGBA', (width, height), (0, 0, 0, 0))
		draw2 = ImageDraw.Draw(image2)
		draw2.text((0, 0), text=word, font=font, fill=(0, 0, 0))

		angle = random.randrange(-1*rotation_degree,rotation_degree)
		image2 = image2.rotate(angle, resample=Image.BICUBIC, expand=True)
		sx, sy = image2.size

	   
		back_c=back_c.resize(( 10 + sx, 10 + sy))
		back_c.paste(image2, (5, 5, 5 + sx, 5 + sy), image2)
		back_c=random_transformation(back_c)
    else:
		back_c=back_c.resize((w+5,h+5))# aby dodać losowo to zamiast +5 daj +rand_pad()
		draw=ImageDraw.Draw(back_c)#narysuj maskę    
		draw.text((0,0),text=word,font=font,fill='rgb(0,0,0)')# narysuj text na masce
		back_c=random_transformation(back_c)# zniekształcanie obrazu
    
    
    
    back_c.save(f'images/{file_counter}_{filename}.jpg')
    file.writelines(str(file_counter)+'_'+filename+'.jpg'+'~'+word+'\n')
    file_counter+=1



#file=open('max_label_len.txt','w')
#file.writelines(str(max_label_len))



    
