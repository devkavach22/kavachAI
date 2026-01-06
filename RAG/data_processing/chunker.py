from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=400,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)


def chunk_pdf_text(
    text: str, splitter: Optional[RecursiveCharacterTextSplitter] = None
):
    """Split `text` into chunks using the provided `splitter` or the
    module-level default `text_splitter`.

    Returns a list of string chunks.
    """
    if splitter is None:
        splitter = text_splitter

    return splitter.split_text(text)


if __name__ == "__main__":
    sample = """Your long text document KDD ’22, August 14–18, 2022, Washington, DC, USA Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter Staar
Table 2: Prediction performance (mAP@0.5-0.95) of object
detection networks on DocLayNet test set. The MRCNN
(Mask R-CNN) and FRCNN (Faster R-CNN) models with
ResNet-50 or ResNet-101 backbone were trained based on
the network architectures from the detectron2 model zoo
(Mask R-CNN R50, R101-FPN 3x, Faster R-CNN R101-FPN
3x), with default con!gurations. The YOLO implementation
utilized was YOLOv5x6 [13]. All models were initialised using pre-trained weights from the COCO 2017 dataset.
human MRCNN FRCNN YOLO
R50 R101 R101 v5x6
Caption 84-89 68.4 71.5 70.1 77.7
Footnote 83-91 70.9 71.8 73.7 77.2
Formula 83-85 60.1 63.4 63.5 66.2
List-item 87-88 81.2 80.8 81.0 86.2
Page-footer 93-94 61.6 59.3 58.9 61.1
Page-header 85-89 71.9 70.0 72.0 67.9
Picture 69-71 71.7 72.7 72.0 77.1
Section-header 83-84 67.6 69.3 68.4 74.6
Table 77-81 82.2 82.9 82.2 86.3
Text 84-86 84.6 85.8 85.4 88.1
Title 60-72 76.7 80.4 79.9 82.7
All 82-83 72.4 73.5 73.4 76.8
to avoid this at any cost in order to have clear, unbiased baseline
numbers for human document-layout annotation. Third, we introduced the feature of snapping boxes around text segments to
obtain a pixel-accurate annotation and again reduce time and e!ort.
The CCS annotation tool automatically shrinks every user-drawn
box to the minimum bounding-box around the enclosed text-cells
for all purely text-based segments, which excludes only Table and
Picture. For the latter, we instructed annotation sta! to minimise
inclusion of surrounding whitespace while including all graphical
lines. A downside of snapping boxes to enclosed text cells is that
some wrongly parsed PDF pages cannot be annotated correctly and
need to be skipped. Fourth, we established a way to "ag pages as
rejected for cases where no valid annotation according to the label
guidelines could be achieved. Example cases for this would be PDF
pages that render incorrectly or contain layouts that are impossible
to capture with non-overlapping rectangles. Such rejected pages are
not contained in the #nal dataset. With all these measures in place,
experienced annotation sta! managed to annotate a single page in
a typical timeframe of 20s to 60s, depending on its complexity.
5 EXPERIMENTS
The primary goal of DocLayNet is to obtain high-quality ML models
capable of accurate document-layout analysis on a wide variety
of challenging layouts. As discussed in Section 2, object detection
models are currently the easiest to use, due to the standardisation
of ground-truth data in COCO format [16] and the availability of
general frameworks such as detectron2 [17]. Furthermore, baseline
numbers in PubLayNet and DocBank were obtained using standard
object detection models such as Mask R-CNN and Faster R-CNN.
As such, we will relate to these object detection methods in this
Figure 5: Prediction performance (mAP@0.5-0.95) of a Mask
R-CNN network with ResNet50 backbone trained on increasing fractions of the DocLayNet dataset. The learning curve
"attens around the 80% mark, indicating that increasing the
size of the DocLayNet dataset with similar data will not yield
signi!cantly better predictions.
paper and leave the detailed evaluation of more recent methods
mentioned in Section 2 for future work.
In this section, we will present several aspects related to the
performance of object detection models on DocLayNet. Similarly
as in PubLayNet, we will evaluate the quality of their predictions
using mean average precision (mAP) with 10 overlaps that range
from 0.5 to 0.95 in steps of 0.05 (mAP@0.5-0.95). These scores are
computed by leveraging the evaluation code provided by the COCO
API [16].
Baselines for Object Detection
In Table 2, we present baseline experiments (given in mAP) on Mask
R-CNN [12], Faster R-CNN [11], and YOLOv5 [13]. Both training
and evaluation were performed on RGB images with dimensions of
1025⇥1025 pixels. For training, we only used one annotation in case
of redundantly annotated pages. As one can observe, the variation
in mAP between the models is rather low, but overall between 6
and 10% lower than the mAP computed from the pairwise human
annotations on triple-annotated pages. This gives a good indication
that the DocLayNet dataset poses a worthwhile challenge for the
research community to close the gap between human recognition
and ML approaches. It is interesting to see that Mask R-CNN and
Faster R-CNN produce very comparable mAP scores, indicating
that pixel-based image segmentation derived from bounding-boxes
does not help to obtain better predictions. On the other hand, the
more recent Yolov5x model does very well and even out-performs
humans on selected labels such as Text, Table and Picture. This is
not entirely surprising, as Text, Table and Pic"""
    chunks = chunk_pdf_text(sample)
    print(f"Chunking text of length {len(sample)}")
    for text in chunks:
        print(f"Chunk text -->>> {text}")
        print(" ")
        print(" ")
    print(len(chunks), "chunks created")
