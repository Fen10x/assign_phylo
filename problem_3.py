from Bio import SeqIO
from Bio import Entrez
import numpy as np
from helper_functions import neighbor_joining
from helper_functions import plot_nj_tree
from helper_functions import reroot_tree
from helper_functions import sort_children_by_leaves
import matplotlib.pyplot as plt

#function to calculate hamming distance
def hamming_distance(seq1, seq2):
    distance = 0

    for a, b in zip(seq1, seq2):
        if a != b:
            distance += 1

    return distance

#function to recursively search tree and identify outgroup node
def find_node(node, name):
    if node is None:
        return None

    if node.name == name:
        return node

    result = find_node(node.left, name)

    if result is not None:
        return result

    return find_node(node.right, name)

#function to recursively add leaf count to current node
def count_leaves(node):
    if node.left is None and node.right is None:
        return 1

    return count_leaves(node.left) + count_leaves(node.right)

#scrape sequences, labels, and accessions from fasta file
records = list(SeqIO.parse("data/clustal_MSA_coronavirus.fasta", "fasta"))
labels = [record.id.split("|")[0] for record in records]
accessions = [record.id.split("|")[1] for record in records]
sequences = [str(record.seq) for record in records]

Entrez.email = "z5308203@student.unsw.edu.au"

def get_taxonomy(accession):
    with Entrez.efetch(
        db="nuccore",
        id=accession,
        rettype="gb",
        retmode="text"
    ) as handle:
        record = SeqIO.read(handle, "genbank")

    return record.annotations["taxonomy"]


#groups = {}
#for label, accession in zip(labels, accessions):
    #print("Fetching:", label, accession)
    #taxonomy = get_taxonomy(accession)
    #if "Alphacoronavirus" in taxonomy:
    #    groups[label] = "Alphacoronavirus"
    #elif "Betacoronavirus" in taxonomy:
    #    groups[label] = "Betacoronavirus"
    #elif "Gammacoronavirus" in taxonomy:
    #    groups[label] = "Gammacoronavirus"
    #elif "Deltacoronavirus" in taxonomy:
    #    groups[label] = "Deltacoronavirus"
    #else:
    #    groups[label] = "Outgroup"
    #print("Finished:", label)

groups = {
    "Breda": "Outgroup",

    "MunCoV_HKU13": "Deltacoronavirus",
    "BuCoV_HKU11": "Deltacoronavirus",
    "ThCoV_HKU12": "Deltacoronavirus",

    "FIPV": "Alphacoronavirus",
    "PRCV": "Alphacoronavirus",
    "TGEV": "Alphacoronavirus",
    "Sc-BatCoV_512": "Alphacoronavirus",
    "PEDV": "Alphacoronavirus",
    "Mi-BatCoV_HKU8": "Alphacoronavirus",
    "Mi-BatCoV_1A": "Alphacoronavirus",
    "Rh-BatCoV_HKU2": "Alphacoronavirus",
    "HCoV-NL63": "Alphacoronavirus",
    "HCoV-229E": "Alphacoronavirus",

    "SW1": "Gammacoronavirus",
    "TCoV": "Gammacoronavirus",
    "IBV": "Gammacoronavirus",

    "MHV": "Betacoronavirus",
    "HCoV-HKU1": "Betacoronavirus",
    "ECoV": "Betacoronavirus",
    "BCoV": "Betacoronavirus",
    "HCoV-OC43": "Betacoronavirus",
    "PHEV": "Betacoronavirus",
    "Pi-BatCoV_HKU5": "Betacoronavirus",
    "Ty-BatCoV_HKU4": "Betacoronavirus",
    "Ro-BatCoV_HKU9": "Betacoronavirus",
    "SARSr-CoV": "Betacoronavirus",
    "SARSr-Rh-BatCoV_HKU3": "Betacoronavirus",
    "PCoV": "Betacoronavirus",
    "BatCoV_RaTG13": "Betacoronavirus",
    "Human-SARS-CoV-2": "Betacoronavirus"
}

#initialise distance matrix for neigbour_joining
n = len(sequences)
distances = np.zeros((n, n))

#fill distance matrix with hamming distances
for i in range(n):
    for j in range(i + 1, n):
        distance = hamming_distance(sequences[i], sequences[j])

        distances[i, j] = distance
        distances[j, i] = distance


tree = neighbor_joining(distances, labels)
breda_node = find_node(tree, "Breda")

rerooted_tree = reroot_tree(tree, breda_node)
sort_children_by_leaves(rerooted_tree)

fig, ax = plt.subplots(figsize=(14, 10))
plot_nj_tree(rerooted_tree, ax, groups)

#plt.show()

plt.savefig('problem3.svg')
