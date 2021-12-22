class Dictionary(object):
    def __init__(self, num_bins, num_classes) -> None:
        self.num_bins = num_bins
        self.num_classes = num_classes
        # 0 - num_bin coordinate, num_bin+1 - num_bin+num_class class,
        # num_bin+num_class+1 end, num_bin+num_class+2 noise
        self.num_vocal = num_bins + 1 + num_classes + 2


def build_dictionary(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    return Dictionary(args.num_bins, num_classes)