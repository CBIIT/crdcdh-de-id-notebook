from sklearn.manifold import TSNE
from matplotlib import pylab
import numpy as np
from sklearn.preprocessing import normalize

def plot_word_vecs_tsne(word_vecs_path, vecs_list, output_plot_path, num_points = 400):
    """
    Plot 2D word vectors TSNE.
    @param word_vectors: word vectors
    @param word_labels: word labels
    @param perplexity: 
    """
    word_vectors, word_labels = None, None
    if word_vecs_path:
        word_vectors, word_labels = normalize_word_vector_file(word_vecs_path, num_points)
    elif vecs_list and len(vecs_list) > 0:
        word_vectors, word_labels = normalize_word_vector(vecs_list, num_points)
    else:
        print("No word vectors found.")
        return
    
    tsne = TSNE(perplexity= (40 if len(word_labels) > 40 else len(word_labels) -1), n_components=2, init="pca", n_iter=10000)
    two_d_embeddings = tsne.fit_transform(word_vectors)
    pylab.figure(figsize=(20, 20))
    for i, label in enumerate(word_labels):
        x, y = two_d_embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(
            label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom"
        )
   
    pylab.savefig(output_plot_path)
    pylab.show()

def normalize_word_vector_file(word_vecs_path, num_points = 400):
    """
    Normalize word vector
    @param word_vector_path: word vector path
    @param num_points: Read the 400 most frequent word vectors by default. The vectors in the file are in descending order of frequency.
    @return: word vector
    """
    first_line = True
    index_to_word = []
    with open(word_vecs_path, "r") as f:
        for line_num, line in enumerate(f):
            if first_line:
                dim = int(line.strip().split()[1])
                word_vecs = np.zeros((num_points, dim), dtype=float)
                first_line = False
                continue
            line = line.strip()
            word = line.split()[0]
            vec = word_vecs[line_num - 1]
            for index, vec_val in enumerate(line.split()[1:]):
                vec[index] = float(vec_val)
            index_to_word.append(word)
            if line_num >= num_points:
                break
    word_vecs = normalize(word_vecs, copy=False, return_norm=False)
    return word_vecs, index_to_word 

def normalize_word_vector(vecs_list, num_points=400):
    """
    Normalize word vector
    @param word_vector_path: word vector dict
    @param num_points: Read the 400 most frequent word vectors by default. The vectors in the file are in descending order of frequency.
    @return: word vector, index_to_word
    """
    index_to_word = [item["word"] for item in vecs_list][: num_points]
    word_vecs = [item["vector"] for item in vecs_list][: num_points]
    word_vecs = normalize(word_vecs, copy=False, return_norm=False)
    return word_vecs, index_to_word