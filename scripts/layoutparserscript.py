import requests
import layoutparser as lp
import cv2

image = cv2.imread("../data/data-html/12khari-2023-01-26T04:03:47.608899+05:45.fullpage.jpg")
image = image[..., ::-1]
    # Convert the image from BGR (cv2 default loading style)
    # to RGB
model = lp.Detectron2LayoutModel('output/config.yaml')
    # Load the deep layout model from the layoutparser API
    # For all the supported model, please check the Model
    # Zoo Page: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html
layout = model.detect(image)
lp.draw_box(image, layout, box_width=3).show()

#layout = lp.load_pdf("../csv-files/homepage-newsworthiness-with-internet-archive/csv-files/1newsnz-2023-03-30T11:15:01.603835+13:00.html.csv")