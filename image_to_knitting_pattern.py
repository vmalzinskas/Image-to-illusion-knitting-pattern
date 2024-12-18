import cv2 as cv
import numpy as np
from icecream import ic
import math
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# ic.disable
class image_convert:

    image: np.array
    def __init__(self, path):
        self.image = cv.imread(path)

    def get_shade(self, row, col):
        return np.median(self.image[row*10:row*10+10, col*10:col*10+10])

    def resize_image(self, width_desired, pixels_per_red_unit): #in red squre units 100 pixels to red unit
        width = width_desired * pixels_per_red_unit
        height = int(width * self.height_to_width)
        self.image = cv.resize(self.image, (width, height))

    def grade_colours(self):
        image = self.image
        (height, width) = image.shape[:2]
        image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=3, n_init=3)
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((height, width, 3))
        quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
        self.image = quant

    def convert_to_grey(self):

        height_of_image, width_of_image, _ = self.image.shape
        color_list = []
        for i in range(0, height_of_image):
            for ii in range(0, width_of_image):
                color_list.append([self.image[i][ii][0],
                                   self.image[i][ii][1],
                                   self.image[i][ii][2]])
        unique_colours = np.unique(color_list, axis=0)

        black = unique_colours[0]
        grey = unique_colours[1]
        white = unique_colours[2]

        black_mask = cv.inRange(self.image, black, black)
        grey_mask = cv.inRange(self.image, grey, grey)
        white_mask = cv.inRange(self.image, white, white)

        self.image[black_mask > 0] = (0, 0, 0)
        self.image[grey_mask > 0] = (150, 150, 150)
        self.image[white_mask > 0] = (255, 255, 255)

    def show_image(self):
        cv.imshow("Display", self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @property
    def height_to_width(self):
        h, w, _ = self.image.shape
        return h / w


class Stitch:

    position_top_left: (int, int)
    stitch: np.array
    colour: str

    def __init__(self, colour):
        self.colour = colour
        self.stitch = np.zeros([10, 10, 3], dtype=np.uint8)
        if colour == "light":
            self.light()
        if colour == "shadow":
            self.shadow()
        if colour == "dark":
            self.dark()
        self.array = self.stitch
    def black_border(self):
        for i in range(0, len(self.stitch[0])):
            for ii in range(0, len(self.stitch[1])):
                if i % 10 == 0 or ii % 10 == 0:
                    self.stitch[i, ii] = [0, 0, 0]


    def shadow(self):
        self.stitch.fill(150)
        self.black_border()
        self.colour = "shadow"

    def dark(self):
        self.stitch.fill(70)
        self.black_border()
        self.colour = "dark"

    def light(self):
        self.stitch.fill(255)
        self.black_border()
        self.colour = "light"

    @property
    def express(self):
        return self.stitch

    def colour(self):
        return self.colour

class Canvas:

    height: int
    width: int
    co_ords: ()
    stitch_matrix: []
    pattern_array: np.array

    def __init__(self, height, width): # h and w in red square values
        self.height = int(math.ceil(height * 10))
        self.width = width * 10
        self.stitch_matrix = [[Stitch("shadow") for _ in range(self.width)] for i in range(self.height)]

    def __iter__(self):
        self.co_ords = (0, 0)
        return self

    def __next__(self):
        x = (self.co_ords[0], self.co_ords[1])
        row = self.co_ords[0]
        col = self.co_ords[1]
        last_row = (self.height - 1)
        last_col = (self.width - 1)

        if row < last_row:
            if col < last_col:
                self.co_ords = (row, col + 1)
            elif col == last_col:
                self.co_ords = (row + 1, 0)
        elif row == last_row:
            if col <= last_col:
                self.co_ords = (row, col+1)
        if col == self.width:
            raise StopIteration
        else:
            return x, self.stitch_matrix[x[0]][x[1]]

    def add_grid_and_save(self, pixels_per_10_stitch, path):
        image_array = np.empty((self.height * 10, self.width * 10, 3), dtype=np.uint8)
        for stored_stitch in self:
            row, col = stored_stitch[0]
            stitch = stored_stitch[1].express
            image_array[row * 10: row * 10 + 10, col * 10: col * 10 + 10] = stitch

        for i in range(0, image_array.shape[0]):
            for ii in range(0, image_array.shape[1]):
                if i % pixels_per_10_stitch == 0:
                    image_array[i][ii] = [0, 0, 255]
                if ii % pixels_per_10_stitch == 0:
                    image_array[i][ii] = [0, 0, 255]

        cv.imwrite(path, image_array)

    @property
    def express(self):
        image_array = np.empty((self.height*10, self.width*10, 3), dtype=np.uint8)
        for stored_stitch in self:
            row, col = stored_stitch[0]
            stitch = stored_stitch[1].express
            image_array[row*10: row*10 + 10, col*10: col*10 + 10] = stitch

        cv.imshow("Display", image_array)
        cv.waitKey(0)
        cv.destroyAllWindows()

class Pattern:

    canvas: Canvas
    image: image_convert
    number_of_pixels_per_stitch = 10
    number_of_pixels_per_10_stitch = 100

    def __init__(self, path, width_desired): #width in red squares
        self.width_desired = width_desired
        self.image = image_convert(path)
        self.image.resize_image(width_desired, self.number_of_pixels_per_10_stitch)
        self.image.grade_colours()
        self.image.convert_to_grey()
        # self.image.show_image()

    def make_canvas(self): #width of image in cm or knit knots
        self.canvas = Canvas((self.width_desired * self.image.height_to_width), self.width_desired)
        # self.canvas.express

    def prep_canvas(self):
        for stitch in self.canvas:
            row, col = stitch[0]
            stitch = stitch[1]
            if row % 2 == 0:
                stitch.dark()
        # self.canvas.express

    def paint_canvas(self, path):
        # self.canvas.express
        '''
        0 <= dark is < 70
        70 <= shadow is < 215
        215 <= light <= 255
        :return:
        '''
        for stitch in self.canvas:
            row, col = stitch[0]
            stitch = stitch[1]
            if stitch.colour == "dark":
                if self.image.get_shade(row, col) > 215:
                    stitch.light()
            if stitch.colour == "shadow":
                if self.image.get_shade(row-1, col) < 70:
                    stitch.light()
                elif self.image.get_shade(row-1, col) >= 70:
                    stitch.shadow()
        self.canvas.add_grid_and_save(self.number_of_pixels_per_10_stitch, path)





path_to_image = r"images\Hagrid.png"
pattern_one = Pattern(path_to_image, 20)
pattern_one.make_canvas()
pattern_one.prep_canvas()
pattern_one.paint_canvas(r"patterns\Hagrid_pattern.png")
