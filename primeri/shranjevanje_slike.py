import spremeni_velikost_slike as svs
import cv2 as cv
def shranjevanje_slike(slika, ime_slike):
    '''Shrani sliko.'''
    cv.imwrite(ime_slike, slika)

if __name__ == '__main__':
    slika = cv.imread('../.utils/lenna.png')
    # Preverimo, če je slika pravilno naložena
    if slika is not None:
        slika_faktor = svs.spremeni_velikost_slike_faktor(slika, 0.5)

        slika_diskretno = svs.spremeni_velikost_slike_diskretno(slika, 100, 100)

        # Shranimo obe sliki
        shranjevanje_slike(slika_faktor, 'slika_faktor_0.5.jpg')
        shranjevanje_slike(slika_diskretno, 'slika_diskretno_100x100.jpg')
