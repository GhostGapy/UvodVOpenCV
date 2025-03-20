import cv2 as cv

def dostop_do_piksla(slika, x, y):
    '''Dostop do piksla.'''
    return slika[y, x]

if __name__ == '__main__':
    slika = cv.imread('../.utils/lenna.png')
    
    if slika is not None:
        piksel = dostop_do_piksla(slika, 100, 100)
        print('Piksel na koordinatah (100, 100) je:', piksel)
    
        cv.imshow('Slika', slika)
        print('Oblika slike:', slika.shape)
        cv.waitKey(0)
        # Zapremo vsa okna
        cv.destroyAllWindows()