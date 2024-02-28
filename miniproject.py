import cv2
import numpy as np
import matplotlib.pyplot as plt
x=516
y=1
global mag
dframes=[[0]*750]*1000
def View():
    cv2.imshow('Image',mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
vid=cv2.VideoCapture("1705951007967.mp4")
while(vid.isOpened()):
    ret,frame=vid.read()
    if(x % 516 == 0 and y==1):
        frame=frame[:,:,2]
        # plt.imshow(frame,cmap='gray',vmin=0,vmax=255)
        # plt.show()
        # ft=np.fft.fft2(frame)
        # mag=20*np.log(np.abs(ft))
        # mag=np.asarray(mag,dtype=int)
        # View()
        f=cv2.dft(np.float32(frame),flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shift=np.fft.fftshift(f)
        f_complex=f_shift[:,:,0]+1j*f_shift[:,:,1]
        f_abs=np.abs(f_complex)
        f_bounded=20*np.log(f_abs)
        f_img=255*f_bounded/np.max(f_bounded)
        f_img=f_img.astype(np.uint8)
        dframes=dframes+f_img
        plt.imshow(dframes,cmap='hsv',vmin=0,vmax=255)
        plt.show()
    y=y-1    
    x=x-1
    if(x==0):
        x=516


