import cv2 as cv

def spremeni_velikost_slike_faktor(slika, f):
    '''Spremeni velikost slike za faktor f.'''
    return cv.resize(slika, (0, 0), fx=f, fy=f)

def spremeni_velikost_slike_diskretno(slika, sirina, visina):
    '''Spremeni velikost slike na določeno širino in višino.'''
    return cv.resize(slika, (sirina, visina))

if __name__ == '__main__':
    slika = cv.imread('../.utils/lenna.png')
    # Preverimo, če je slika pravilno naložena
    if slika is not None:
        slika_faktor_1 = spremeni_velikost_slike_faktor(slika, 0.5)
        slika_faktor_2 = spremeni_velikost_slike_faktor(slika, 1.5)
        # Prikažemo obe sliki spremenjeni za faktor
        cv.imshow('Slika faktor 0.5', slika_faktor_1)
        cv.imshow('Slika faktor 1.5', slika_faktor_2)        
    
        cv.waitKey(0)

        slika_diskretno_1 = spremeni_velikost_slike_diskretno(slika, 100, 100)
        slika_diskretno_2 = spremeni_velikost_slike_diskretno(slika, 30, 100)
        #Prikažemo obe sliki spremenjeni na določeno velikost
        cv.imshow('Slika diskretno 100x100', slika_diskretno_1)
        cv.imshow('Slika diskretno 30x100', slika_diskretno_2)
        cv.waitKey(0)
        # Zapremo vsa okna
        cv.destroyAllWindows()