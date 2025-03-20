import cv2 as cv

if __name__ == '__main__':
    # Naložimo kamero
    kamera = cv.VideoCapture(0)
    # Preverimo, če je kamera pravilno naložena
    if not kamera.isOpened():
        print('Kamera ni bila odprta.')
    else:
        while True:
            # Preberemo sliko iz kamere
            ret, slika = kamera.read()
            cv.imshow('Kamera', slika)
            # Če pritisnemo tipko 'q', zapremo okno
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        # Zapremo okno
        kamera.release()
        cv.destroyAllWindows()