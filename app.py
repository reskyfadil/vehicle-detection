import streamlit as st
import cv2
import numpy as np
import moviepy.editor as moviepy

st.title('Vehicle Detection')
st.header('Sistem Pendeteksi Objek Yang Melintas Pada Jalan Raya')



min_width_rect = 80
min_height_rect = 80
count_line_position = 550
algo = cv2.bgsegm.createBackgroundSubtractorMOG()


def center_point(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy


detect = []
offset = 8
counter = 0
st.subheader("Masukkan video")
video_data = st.file_uploader("Upload file", ['mp4','mov', 'avi'])

temp_file_to_save = './temp_file_1.mp4'
temp_file_result  = './temp_file_2.mp4'

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

if video_data:
    # save uploaded video to disc
    write_bytesio_to_file(temp_file_to_save, video_data)

    # read it with cv2.VideoCapture(), 
    # so now we can process it with OpenCV functions
    cap = cv2.VideoCapture(temp_file_to_save)

    # grab some parameters of video to use them for writing a new, processed video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int
    st.write(width, height, frame_fps)
    
    # specify a writer to write a processed video to a disk frame by frame
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height),isColor = True)
   
    while True:
        ret,frame1 = cap.read()
        if not ret: break
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = algo.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (25, count_line_position), (1250, count_line_position), (0, 0, 0), 4)

        for (i, c) in enumerate(counterShape) :
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w>= min_width_rect) and (h>= min_height_rect)
            if not validate_counter :
                continue

            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)

            center = center_point(x, y, w, h)
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 255, 0), -1)


            for (x, y) in detect :
                if y<(count_line_position+offset) and y>(count_line_position-offset) :
                    counter += 1
                    cv2.line(frame1, (25, count_line_position), (1250, count_line_position), (255, 255, 255), 4)
                    print("Jumlah Kendaraan : " + str(counter))
                cv2.putText(frame1,"Jumlah Kendaraan : " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                detect.remove((x,y))
        out_mp4.write(frame1)
    
    ## Close video files
    out_mp4.release()
    cap.release()

    ## Reencodes video to H264 using ffmpeg
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    convertedVideo = "./testh264.mp4"
    #subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "), shell = True)
    clip = moviepy.VideoFileClip(temp_file_result)
    clip.write_videofile(convertedVideo)




    ## Show results
    col1,col2 = st.columns(2)
    col1.header("Original Video")
    col1.video(temp_file_to_save)
    #col2.header("Output from OpenCV (MPEG-4)")
    #col2.video(temp_file_result)
    col2.header("Result")
    col2.video(convertedVideo)