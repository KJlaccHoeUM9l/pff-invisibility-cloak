import cv2
import time
import numpy as np


from invisibility_cloak import InvisibilityCloak


def main():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('3rdparty/raw_videos/scene_{}.mp4'.format(time.time()), fourcc, 24.0, (640, 480))

    invisibility_cloak = InvisibilityCloak()
    invisibility_cloak.take_background_image(cap, 5, 30)
    while True:
        _, frame = cap.read()
        frame = np.flip(frame, axis=1)
        current_key = cv2.waitKey(1) & 0xFF
        invisibility_cloak.update_mode(current_key)

        invisibility_cloak.tune_color_threshold()
        image_color_mask = invisibility_cloak.find_color_mask(frame)
        background_mask = invisibility_cloak.get_background_mask(image_color_mask)
        final_image = invisibility_cloak.get_final_image(frame, background_mask, image_color_mask)

        video_writer.write(final_image)
        cv2.imshow('Camouflage Pro G63 Note Max Super Incredible X5M 3.5 JZ Turbo S+', final_image)
        if current_key & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
