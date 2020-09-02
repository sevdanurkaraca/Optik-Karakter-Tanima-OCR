import cv2
import numpy as np
import pytesseract
import imutils

#ilgili resmi programa dahil ediyoruz
img = cv2.imread("D:\\ocrProje\\plaka_tanima\\9.1 licence_plate.jpg.jpg")

#COLOR_BGR2GRAY çevirmek istediğimiz format //gri tona çevirme
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gri resmi filtrelemek için: (resmin köşelerini yumuşatma oranları)
#çap,sigmacolor,sigmaspace çap arttıkça bozunum artıyor
filtered = cv2.bilateralFilter(gray,5,250,250)

#yumuşatılmış resim üzerinde köşeleri algılamak için    [edged = köşeleri algılandırılmış, köşelendirilmiş (min,max)]
edged = cv2.Canny(filtered,30,200)

#koordinatları bulmak için konturlama işlemi=sınırların koordinatlarını bulma
contours = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#imutils kütüphanesini kullanarak uygun konturları çekme işlemi(uygun kontur değerlerini yakalama)
cnts = imutils.grab_contours(contours)

#yakalanan konturları sıralama işlemi: yakalanan koordinatlara bakarak bir dikdörtgen şekli yakalamaya çalışıyoruz = plaka dikdörtgen şeklinde olduğu için
#alana göre sıralamayı sağlar = key = cv2.contourArea
#reverse = true ile koordinatları alanlarına göre ters çevirerek sırala 0 dan 10 a kadar
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:10]

#kapalı şekil bulup bulamadığını anlamak için screen değişkeni oluşturup none değeri veriyoruz ilk olarak daha sonra bu değer değişecek
#koordinatları tutuyor kısaca
screen = None

#koordinatlardan bir dikdörtgen elde etmek için for döngüsü oluşturuyoruz
for c in cnts:
    #koturlara yaklaşmak için: konturları daha düzgün hale getirmak için:yaklaşım
    epsilon = 0.018*cv2.arcLength(c,True) #konturların yay uzunluğunu bulmak için
    #konturlara yaklaşım için kullanılan fonksiyon
    approx = cv2.approxPolyDP(c,epsilon,True)#konturların daha da sınırlara yaklaşılmış hali
    if len(approx) ==4:
        screen = approx
        break


#gray in içinde tuttuğu boyutları kadar siyah bir ekran oluşturuyoruz
mask = np.zeros(gray.shape,np.uint8)

#plaka bölgesini beyaza çevirme. Araba oluşturduğumuz maskın arkasında kalıyor
new_img = cv2.drawContours(mask,[screen],0,(255,255,255),-1)

#plaka alanına yazıyı yapıştırma işlemi: mask alanına img yi yapıştırma
new_img = cv2.bitwise_and(img,img,mask = mask)

#en son aşamada oluşan resmi kırpmak için: (x,y) koordinatları beyaz olan bölgeleri tutacaktır. Dizi halinde belirliyor
(x,y) = np.where(mask == 255)

#bilgisayarlarda sol üst köşe (0,0) olarak kabul edilir ve aşağıya doğru koordinatlar artar
#dikdörtgende üstleri bulmak için:
(topx,topy) = (np.min(x),np.min(y))

#dikdörtgende alt kısımları bulmak için:
(bottomx,bottomy) = (np.max(x),np.max(y))

#kırpma işlemini yapıyoruz köşeler belirlendikten sonra(dikdörtgenin yerini belirledikten sonra)
cropped = gray[topx:bottomx+1,topy:bottomy+1]

#kırpılmış resimden metini okuma:SON
text = pytesseract.image_to_string(cropped,lang ="eng")
print("algilanmis metin(PLAKA):",text)




cv2.imshow("mask",mask)
cv2.imshow("mask",new_img)   
cv2.imshow("kirpilmis",cropped)   


cv2.imshow("1.original",img)
cv2.imshow("2.gray",gray)
cv2.imshow("3.filtered",filtered)
cv2.imshow("4.edged",edged)


cv2.waitKey(0)
cv2.destroyAllWindows()

