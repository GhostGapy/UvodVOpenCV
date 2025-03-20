import cv2 as cv

def izrezi_del_slike(slika, x, y, sirina, visina):
    '''Izreži del slike.'''
    return slika[y:y+visina, x:x+sirina]

if __name__ == '__main__':
    slika = cv.imread('../.utils/lenna.png')
    # Preverimo, če je slika pravilno naložena
    if slika is not None:
        slika_izrez = izrezi_del_slike(slika, 100, 100, 200, 200)
        # Prikažemo izrezano sliko
        cv.imshow('Izrezana slika', slika_izrez)
        cv.waitKey(0)
        # Zapremo vsa okna
        cv.destroyAllWindows()