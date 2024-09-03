def filter_images(dataset,labels,labels_old,overlap):
    index = []
    if 0 in labels:
        labels.remove(0)
    labels_cum = labels_old+labels+[0,255]