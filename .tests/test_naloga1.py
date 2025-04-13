import sys
import os
sys.path.append(os.getcwd())

import naloga1
import cv2 as cv
import numpy as np
import math
import functools, operator

def test_zmanjsaj_sliko():
    slika = cv.imread('.utils/slika.png') 
    slika_zmanjsana = naloga1.zmanjsaj_sliko(slika, 100, 200)
    assert slika_zmanjsana.shape[0] == 200
    assert slika_zmanjsana.shape[1] == 100
    assert slika_zmanjsana.shape[2] == 3
 
def test_prestej_piksle_z_barvo_koze():
    slika = cv.imread('.utils/slika.png')
    barva_koze = (np.array([0, 0, 0]), np.array([255, 255, 255]))
    stevilo_piklov = naloga1.prestej_piksle_z_barvo_koze(slika, barva_koze)
    assert stevilo_piklov == slika.shape[0] * slika.shape[1]


def test_obdelaj_sliko():
    slika = cv.imread('.utils/slika.png')

    def st_kvadratov(sirina_slike, visina_slike, sirina_skatle, visina_skatle):
        return ((sirina_slike - sirina_skatle) // sirina_skatle + 1) * ((visina_slike - visina_skatle) // visina_skatle + 1)

    # Primer 1
    sirina_skatle = 100
    visina_skatle = 100
    barva_koze = (np.array([255, 0, 0]), np.array([255, 0, 0]))
    skatle = naloga1.obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze)
    assert len(skatle) == st_kvadratov(slika.shape[1], slika.shape[0], sirina_skatle, visina_skatle)

    # Primer 2
    sirina_skatle = 50
    visina_skatle = 50
    barva_koze = (np.array([0, 255, 0]), np.array([0, 255, 0]))
    skatle = naloga1.obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze)
    assert len(skatle) == st_kvadratov(slika.shape[1], slika.shape[0], sirina_skatle, visina_skatle)

    # Primer 3
    sirina_skatle = 200
    visina_skatle = 100
    barva_koze = (np.array([0, 0, 255]), np.array([0, 0, 255]))
    skatle = naloga1.obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze)
    assert len(skatle) == st_kvadratov(slika.shape[1], slika.shape[0], sirina_skatle, visina_skatle)
