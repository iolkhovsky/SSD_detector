

class ObjectDescriptor:

    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.class_label = None
        self.class_id = None
        return

    def __str__(self):
        text_description = "Object descriptor: "
        text_description += "BBox("+str(self.x)+","+str(self.y)+","+str(self.w)+","+str(self.h)+")"
        text_description += " ClassLabel: " + str(self.class_label) + " Id: " + str(self.class_id)
        return text_description


class SampleDescriptor:

    def __init__(self):
        self.abs_path = None
        self.img_size = None
        self.objects = list()
        return

    def __str__(self):
        text_description = "Sample Descriptor: " + str(self.img_size) + "\n" + str(self.abs_path) + "\n"
        if type(self.objects) == list:
            for obj in self.objects:
                text_description += str(obj) + "\n"
        return text_description


class DetectionDescriptor:

    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.p = None
        self.id = None
        self.label = None

    def __str__(self):
        return f"Id {self.id}, P: {self.p}, box: [{self.x}, {self.y}, {self.w}, {self.h}]"