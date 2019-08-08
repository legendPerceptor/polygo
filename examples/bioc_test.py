# from bioc import BioCReader
# from bioc import BioCWriter

INPUT_FILE="data/chemdner_corpus/training.bioc.xml"
DTD_FILE="data/chemdner_corpus/BioC.dtd"

# bioc_reader = BioCReader(INPUT_FILE, DTD_FILE)
# bioc_reader.read()

import bioc

with bioc.BioCXMLDocumentReader(INPUT_FILE) as reader:
    collection_info = reader.get_collection_info()
    for document in reader:
        print(document)
