import os
from lxml import etree

def get_classes(source_path, files):

    class_set = set([])
    for file in files:
        with open(os.path.join(source_path, file), 'rb') as fb:

            xml = etree.HTML(fb.read())
            labels = xml.xpath('//object')
            for label in labels:
                name = label.xpath('./name/text()')[0]
                class_set.add(name)
    return list(class_set)

if __name__ == '__main__':


    xml_path = r'/data1/lsl/wzd/dataset/biandianzhan/train/annotations/xmls/'
    

    files = os.listdir(xml_path)
    classes = get_classes(xml_path, files)


    for cls in classes:
        print(cls)

    print("ok")