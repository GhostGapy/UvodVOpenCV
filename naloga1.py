import cv2
import numpy as np
import time


def doloci_barvo_koze(slika, levo_zgoraj, desno_spodaj):

    # izrezi obmocje koze
    x1, y1 = levo_zgoraj
    x2, y2 = desno_spodaj
    obmocje_koze = slika[y1:y2, x1:x2]

    # pretvori obmocje v 2D matriko pikslov
    piksli = obmocje_koze.reshape(-1, 3)

    # izracunamo povp in std
    povprecje = np.mean(piksli, axis=0)
    std_dev = np.std(piksli, axis=0)

    # Doloci meje barve koze
    spodnja_meja = np.array([
        max(0, povprecje[0] - 1.5 * std_dev[0]),
        max(0, povprecje[1] - 1.5 * std_dev[1]),
        max(0, povprecje[2] - 1.5 * std_dev[2])
    ], dtype=np.uint8)

    zgornja_meja = np.array([
        min(255, povprecje[0] + 1.5 * std_dev[0]),
        min(255, povprecje[1] + 1.5 * std_dev[1]),
        min(255, povprecje[2] + 1.5 * std_dev[2])
    ], dtype=np.uint8)

    return spodnja_meja, zgornja_meja


def zmanjsaj_sliko(slika, sirina, visina):
    return cv2.resize(slika, (sirina, visina), interpolation=cv2.INTER_AREA)


def prestej_piksle_z_barvo_koze(slika, barva_koze):
    spodnja_meja, zgornja_meja = barva_koze
    maska = cv2.inRange(slika, spodnja_meja, zgornja_meja)

    #cv2.imshow('Maska kože', maska)

    return cv2.countNonZero(maska)


def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze):
    visina, sirina = slika.shape[:2]
    rezultati = []

    # gremo cez celotno sliko in izrezemo skatle
    for y in range(0, visina - visina_skatle + 1, visina_skatle):
        for x in range(0, sirina - sirina_skatle + 1, sirina_skatle):
            # izrezemo podsliko
            podsklika = slika[y:y + visina_skatle, x:x + sirina_skatle]

            # prestej piksle z barvo koze
            stevilo_pikslov = prestej_piksle_z_barvo_koze(podsklika, barva_koze)

            # dodamo rez v seznam
            rezultati.append((x, y, sirina_skatle, visina_skatle, stevilo_pikslov))

    return rezultati


def najdi_obraze(slika, rezultati, prag=0.8):

    if not rezultati:
        return []

    # sortiramo rezultate po stevilu pikslov od najvecjega do najmanjsega
    rezultati.sort(key=lambda x: x[4], reverse=True)

    # izracunamo max st pikslov
    max_pikslov = rezultati[0][2] * rezultati[0][3]

    # ibremo samo tiste skatle, ki imajo dovolj pikslov
    obrazi = []
    for x, y, sirina, visina, stevilo_pikslov in rezultati:
        if stevilo_pikslov / max_pikslov >= prag:
            obrazi.append((x, y, sirina, visina))

    # zdruzimo prekrivajoce se skatle
    obrazi_nms = []
    while obrazi:
        # vzamemo skatlo z največ piksli
        x1, y1, w1, h1 = obrazi.pop(0)

        # primerjamo z ostalimi skatlami
        i = 0
        while i < len(obrazi):
            x2, y2, w2, h2 = obrazi[i]

            # preverimo prekrivanje
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y

            # ce je prekrivanje dovolj veliko lahko damo skatli ven
            if overlap_area > 0.3 * min(w1 * h1, w2 * h2):
                obrazi.pop(i)
            else:
                i += 1

        obrazi_nms.append((x1, y1, w1, h1))

    return obrazi_nms


def main():
    video = '.utils/ORV.MP4'
    kamera = cv2.VideoCapture(video)

    video_fps = kamera.get(cv2.CAP_PROP_FPS)

    #print(f"fps videa: {video_fps}")

    # ciljna velikost slike
    ciljna_sirina = 320
    ciljna_visina = 240

    # doloceno obmocje za kalibracijo
    #levo_zgoraj = (int(ciljna_sirina * 0.3), int(ciljna_visina * 0.3))
    #desno_spodaj = (int(ciljna_sirina * 0.7), int(ciljna_visina * 0.7))

    levo_zgoraj = (int(ciljna_sirina * 0.35), int(ciljna_visina * 0.25))
    desno_spodaj = (int(ciljna_sirina * 0.65), int(ciljna_visina * 0.75))

    # velikost slike za iskanje obraza
    sirina_skatle = int(ciljna_sirina * 0.15)
    visina_skatle = int(ciljna_visina * 0.15)

    barva_koze = None
    fps = 0
    prev_time = time.time()
    frame_count = 0

    kalibracija_opravljena = False

    print("Postavi obrazec v kvadrat.")
    print("Pritisni 'space' za kalibracijo, 'q' za izhod, 'r' za ponastavi kalibracijo.")

    while True:
        # zajamemo sliko
        ret, okvir = kamera.read()
        if not ret:
            print("Napaka pri branju")
            # probamo spet ce je napaka
            kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, okvir = kamera.read()
            if not ret:
                break

        #okvir = cv2.rotate(okvir, cv2.ROTATE_90_CLOCKWISE)
        okvir = cv2.flip(okvir, 1)

        # pomanjsamo sliko
        okvir = zmanjsaj_sliko(okvir, ciljna_sirina, ciljna_visina)

        # mermimo cas za calc FPS
        trenutni_cas = time.time()
        frame_count += 1

        # calc fps vsakih 10 slicic
        if frame_count >= 10:
            fps = frame_count / (trenutni_cas - prev_time)
            prev_time = trenutni_cas
            frame_count = 0

        # izrisemo obmocje za kalibracijo
        if not kalibracija_opravljena:
            cv2.rectangle(okvir, levo_zgoraj, desno_spodaj, (0, 255, 0), 2)
        else:
            # obdelamo sliko s skatlicami
            rezultati = obdelaj_sliko_s_skatlami(okvir, sirina_skatle, visina_skatle, barva_koze)

            # najdemo skatle z barvo koze
            obrazi = najdi_obraze(okvir, rezultati)

            # izrisemo skatle ki so zaznani kot obraz
            for x, y, sirina, visina in obrazi:
                cv2.rectangle(okvir, (x, y), (x + sirina, y + visina), (0, 0, 255), 2)


        okvir = cv2.putText(okvir, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # prikazemo window
        #cv2.imshow('Detekcija obraza na osnovi barve', okvir)


        # pocakamo na fps videa za pravilno hitrost predvajanja
        tipka = cv2.waitKey(1) & 0xFF


        if tipka == ord('q'):
            break

        elif tipka == ord(' ') and not kalibracija_opravljena:
            barva_koze = doloci_barvo_koze(okvir, levo_zgoraj, desno_spodaj)
            kalibracija_opravljena = True
            print("Kalibracija opravljena!")
            print(f"Spodnja meja: {barva_koze[0]}")
            print(f"Zgornja meja: {barva_koze[1]}")

        elif tipka == ord('r'):
            kalibracija_opravljena = False
            print("Kalibracija ponastavljena. Pritisni 'space' za ponovno kalibracijo.")


    kamera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Program se zaganja...")
    main()