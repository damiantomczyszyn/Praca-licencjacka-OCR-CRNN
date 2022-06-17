import numpy as np
import string
from PIL import Image,ImageFont,ImageDraw
import argparse
import random
import os
import imgaug.augmenters as iaa
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('--n_samples',type=int)
parser.add_argument('--word_type',default='lowercase')
args=parser.parse_args()

global gray_back
kernel=np.ones((2,2),np.uint8)
kernel2=np.ones((1,1),np.uint8)
punclist='.?:;"'
punclist2="-+/()[]!`,|*&^%$#@'"

#Character sets to choose from.
smallletters=string.ascii_lowercase
capitalletters=string.ascii_uppercase
digits=string.digits
alll=smallletters+capitalletters+digits+punclist+punclist2




#Base backgound.
backfilelist=os.listdir('background/')
backgroud_list=[]

for bn in backfilelist:
    fileloc='background/'+bn
    backgroud_list.append(Image.fromarray(cv2.imread(fileloc,0)))


#Different fonts to be used.
fonts_list=os.listdir('fonts/')
fonts_list=['fonts/'+f for f in fonts_list]

#Lengths of the words.
word_lengths=[]
for l in range(1,21):# długośc znaku od 1 do 21
    word_lengths.append(l)

#Font size.
font_size=[]
for l in range(10,30):# rozmiar od 10 do 30
    font_size.append(l)


file_counter=0


def random_brightness(img):
    img=np.array(img)
    brightness=iaa.Multiply((0.2,1.2))
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


file=open('annotation.txt','a+')

file_counter=0

for _ in range(args.n_samples):

    back_c=random.choice(backgroud_list).copy()# wybór tła
    start_cap=random.choice(capitalletters)#Narazie nie ma znaczenia
    filename=''.join([random.choice(smallletters) for c in range(random.choice([5,6,7,8,9,10,11]))])# nazwa pliku
    font=ImageFont.truetype(random.choice(fonts_list),size=random.choice(font_size))#font
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
    w,h=font.getsize(word)[0],font.getsize(word)[1]# zapisanie rozmiaru z czcionki dla wylosowanego wyrazu wylosowanego fontu i wielkości
    back_c=back_c.resize((w*2,h*2))# nie wiem jakie ma wymiary ostateczne
    #draw=ImageDraw.Draw(back_c)#narysuj maskę    
    #draw.text((0,0),text=word,font=font,fill='rgb(0,0,0)')# narysuj text na masce
    # jeśli chcemy rysować pochylony tekst to należy zmniejszyć maskę do minimum
    #Narysować na niej tekst
    #obrócić całość o kąt
    # i wklić w większą maskę całego już obrazu która jest pozioma
    
    
    
    
    char_image = np.zeros((w*2, h*2, 3), np.uint8)

# convert to pillow image
    pillowImage = Image.fromarray(char_image)

# draw the text
    


    
    angle=10# kąt będzie losowy
    max_dim = max(w+5, h+5)# rozmiar naszego obrazka
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)
    
    draw2 = ImageDraw.Draw(mask)
    draw2.text((max_dim, max_dim),text=word,font=font,fill='rgb(0,0,0)')
    
    mask.show()
       
    bigger_mask = mask.resize((max_dim*8, max_dim*8),resample=Image.BICUBIC)
    rotated_mask = bigger_mask.rotate(angle).resize(mask_size, resample=Image.LANCZOS)
    
    mask_xy = (max_dim - 0, max_dim - 0)# tutaj nie wiem jak przycinać
    b_box = mask_xy + (mask_xy[0] + w, mask_xy[1] + h)
    mask = rotated_mask.crop(b_box)
    

    
    # paste the appropriate color, with the text transparency mask

    color_image = Image.new('RGBA', back_c.size, (0,0,0))
    pillowImage.paste(color_image, mask)
    
    
    pillowImage.show()
    
    
    
    #back_c = back_c.rotate(30)

    back_c=random_transformation(back_c)# zniekształcanie obrazu
    back_c.save(f'images/{file_counter}_{filename}.jpg')
    file.writelines(str(file_counter)+'_'+filename+'.jpg'+'~'+word+'\n')
    file_counter+=1







    
