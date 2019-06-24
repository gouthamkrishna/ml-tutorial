import cv2
import os


def change_contrast(path):
    # -----Reading the image-----------------------------------------------------
    img = cv2.imread(path, 1)
    # cv2.imshow("img", img)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(1, 1))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)

    return final


def change_gray(path):
    print(path)
    img = cv2.imread(path, 0)

    return img


def main():
    for filename in os.listdir('sign_classifier/images/speedlimit30/'):
        src_path = os.path.join(
            'sign_classifier/images/speedlimit30/', filename)
        dest_path = os.path.join(
            'sign_classifier/images/speedlimit30gray/', filename)

        # contra_image = change_contrast(src_path)
        gray_image = change_gray(src_path)

        cv2.imwrite(dest_path, gray_image)


if __name__ == "__main__":
    main()
