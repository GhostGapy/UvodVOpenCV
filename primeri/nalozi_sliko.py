import cv2 as cv

if __name__ == '__main__':
    # Naložimo sliko
    slika = cv.imread('../.utils/lenna.png')
    # Preverimo, če je slika pravilno naložena
    if slika is None:
        print('Slika ni bila naložena.')
    else:
        print('Slika je bila naložena.')
        # Prikažemo sliko
        cv.imshow('Slika', slika)
        # Počakamo na pritisk tipke
        cv.waitKey(0)
        # Zapremo vsa okna
        cv.destroyAllWindows()