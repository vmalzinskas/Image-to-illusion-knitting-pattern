import cv2 as cv
import numpy as np
from icecream import ic
import math
from sklearn.cluster import MiniBatchKMeans
import numpy as np

ic.disable()

class Stitch:

    position_top_left: (int, int)
    stitch: np.array
    color: str

    def __init__(self, colour):
        self.color = colour
        self.stitch = np.zeros([10, 10, 3], dtype=np.uint8)
        if colour == "light":
            self.light()
        if colour == "shadow":
            self.shadow()
        if colour == "dark":
            self.dark()
        self.array = self.stitch
        # ic(self.stitch)
        # cv.imshow("Display", self.stitch)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def shadow(self):
        self.stitch.fill(150)
        self.grading = "shadow"

    def dark(self):
        self.stitch.fill(70)
        self.grading = "dark"

    def light(self):
        self.stitch.fill(255)
        self.grading = "light"

# first = Stitch()
class Pattern:

    image: np.ndarray
    desired_width = 10
    number_of_pixels_per_10_stitch = 100
    background: np.ndarray
    black = 0
    grey = 150
    white = 255

    def __init__(self, path):
        self.image = cv.imread(path)

    def show_image(self):
        cv.namedWindow("Display, cv.WINDOW_AUTOSIZE")
        cv.imshow("Display", self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def get_size(self):
        return self.image.shape[:2]

    def grey_scale(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # cv.imshow("Display", self.image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def map_image_to_background(self):
        pattern = self.background
        for i in range(0, pattern.shape[0]):
            for ii in range(0, pattern.shape[1]):
                if i < pattern.shape[0] - 20 and ii < pattern.shape[1]:
                    if i % 10 == 0 and ii % 10 == 0:

                        if np.array_equiv(pattern[i, ii], [70, 70, 70]):
                            if np.median(self.image[i:i+10, ii:ii+10]) < 70:
                                pattern[i+10:i+20, ii:ii+10] = Stitch("light").array

                            if np.median(self.image[i:i+10, ii:ii+10]) > 215:
                                pattern[i:i+10, ii:ii+10] = Stitch("light").array

        number_of_pixels_per_stitch = 10
        for i in range(0, pattern.shape[0]):
            for ii in range(0, pattern.shape[1]):
                if i % number_of_pixels_per_stitch == 0:
                     pattern[i][ii] = [0, 0, 0]
                if ii % number_of_pixels_per_stitch == 0:
                    pattern[i][ii] = [0, 0, 0]
        for i in range(0, pattern.shape[0]):
            for ii in range(0, pattern.shape[1]):
                if i % self.number_of_pixels_per_10_stitch == 0:
                     pattern[i][ii] = [0, 0, 255]
                if ii % self.number_of_pixels_per_10_stitch == 0:
                    pattern[i][ii] = [0, 0, 255]

        cv.imwrite("pattern_dinosaur1_very_high_detail.jpg", pattern)


        # cv.imshow("Display", pattern)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


    def height_to_width(self):
        h, w, _ = self.image.shape
        return h/w

    def make_background(self, width_desired): #width of image in cm or knit knots
        number_of_pixels_per_stitch = int(0.1*self.number_of_pixels_per_10_stitch)
        width_of_output = width_desired * self.number_of_pixels_per_10_stitch
        height_of_output = int(width_of_output * self.height_to_width())
        background = np.zeros([height_of_output+10, int(width_of_output), 3], dtype=np.uint8)
        background.fill(150)
        for i in range(0, height_of_output):
            for ii in range(0, width_of_output):
                if i <= height_of_output and ii <= width_of_output:
                    if i % 10 == 0 and i % 20 != 0 and ii % 10 ==0:
                        background[i:i+10, ii:ii+10] = Stitch("dark").array



        # for i in range(0, background.shape[0]):
        #     for ii in range(0, background.shape[1]):
        #         if i % number_of_pixels_per_stitch != 0 and row_grey:
        #              background[i][ii] = [200, 200, 200]
        #         if ii % number_of_pixels_per_stitch != 0 and row_grey:
        #             background[i][ii] = [200, 200, 200]
        #     if i % 10 == 0:
        #         row_grey = not row_grey
        #         ic(row_grey)
        # for i in range(0, background.shape[0]):
        #     for ii in range(0, background.shape[1]):
        #         if i % number_of_pixels_per_stitch == 0:
        #              background[i][ii] = [0, 0, 0]
        #         if ii % number_of_pixels_per_stitch == 0:
        #             background[i][ii] = [0, 0, 0]
        # for i in range(0, background.shape[0]):
        #     for ii in range(0, background.shape[1]):
        #         if i % self.number_of_pixels_per_10_stitch == 0:
        #              background[i][ii] = [0, 0, 255]
        #         if ii % self.number_of_pixels_per_10_stitch == 0:
        #             background[i][ii] = [0, 0, 255]
        # ic(background)
        self.background = background

        # cv.imshow("Display", background)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def process_image(self, width_desired):
        # width_desired = width_desired*self.number_of_pixels_per_10_stitch
        (height_of_image, width_of_image, _) = self.image.shape
        # ic(height_of_image, width_of_image)
        width = width_desired * self.number_of_pixels_per_10_stitch
        height = int((width_desired * self.number_of_pixels_per_10_stitch) * self.height_to_width())

        color_list = []
        for i in range(0, height_of_image):
            for ii in range(0, width_of_image):
                color_list.append([self.image[i][ii][0],
                                   self.image[i][ii][1],
                                   self.image[i][ii][2]])
        unique_colours = np.unique(color_list, axis=0)
        #
        black = unique_colours[0]
        grey = unique_colours[1]
        white = unique_colours[2]

        ic(black)
        ic(grey)
        ic(white)

        black_mask = cv.inRange(self.image, black, black)
        grey_mask = cv.inRange(self.image, grey, grey)
        white_mask = cv.inRange(self.image, white, white)

        self.image[black_mask > 0] = (0, 0, 0)
        self.image[grey_mask > 0] = (150, 150, 150)
        self.image[white_mask > 0] = (255, 255, 255)
        self.image = cv.resize(self.image, (width, height))

        # cv.imshow("Display", self.image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def preprocess_image(self):
        image = self.image
        (height, width) = image.shape[:2]
        ic(height, width)
        image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=3, n_init=3)
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((height, width, 3))
        quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
        self.image = quant

        # cv.imshow("Display", quant)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

path_to_image = r"dinosaur1.jpg"
width_end_product = 40
pattern = Pattern(path_to_image)
pattern.preprocess_image()
pattern.process_image(width_end_product)
pattern.make_background(width_end_product)
pattern.map_image_to_background()
