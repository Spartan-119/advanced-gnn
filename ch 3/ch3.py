# creating node representations with DeepWalk

import numpy as np
np.random.seed(0)

CONTEXT_SIZE = 2

text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc eu sem 
scelerisque, dictum eros aliquam, accumsan quam. Pellentesque tempus, lorem ut 
semper fermentum, ante turpis accumsan ex, sit amet ultricies tortor erat quis 
nulla. Nunc consectetur ligula sit amet purus porttitor, vel tempus tortor 
scelerisque. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices 
posuere cubilia curae; Quisque suscipit ligula nec faucibus accumsan. Duis 
vulputate massa sit amet viverra hendrerit. Integer maximus quis sapien id 
convallis. Donec elementum placerat ex laoreet gravida. Praesent quis enim 
facilisis, bibendum est nec, pharetra ex. Etiam pharetra congue justo, eget 
imperdiet diam varius non. Mauris dolor lectus, interdum in laoreet quis, 
faucibus vitae velit. Donec lacinia dui eget maximus cursus. Class aptent taciti
sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Vivamus
tincidunt velit eget nisi ornare convallis. Pellentesque habitant morbi 
tristique senectus et netus et malesuada fames ac turpis egestas. Donec 
tristique ultrices tortor at accumsan.
""".split()


# creating skipgrams
skipgrams = []
for i in range(CONTEXT_SIZE, len(text) - CONTEXT_SIZE):
    array = [text[j] for j in np.arange(i - CONTEXT_SIZE, i + CONTEXT_SIZE + 1) if j != i]
    skipgrams.append((text[i], array))

print(skipgrams[0: 2])

print("\n*----------*----------*----------*\n")

vocab = set(text)
VOCAB_SIZE = len(vocab)
print(f"Length of vocabulary = {VOCAB_SIZE}")

# Now that we have the size of our vocabulary, there is one more parameter we need to define: , the
# dimensionality of the word vectors. Typically, this value is set between 100 and 1,000. In this example,
# we will set it to 10 because of the limited size of our dataset.

# Note:There is no activation function: Word2Vec is a linear classifier that models a linear relationship
# between words.

from gensim.models.word2vec import Word2Vec

# initialise the skip-gram model with a Word2Vec object and a 
# skip-gram (sg) parameter 1
model = Word2Vec([text],
                 sg = 1,
                 vector_size=10,
                 min_count=0,
                 window=2,
                 workers=2,
                 seed=0)

print(f"The Shape of w_embed: {model.wv.vectors.shape}")

# next we train the model for 10 epochs
model.train([text], total_examples=model.corpus_count, epochs=10)

# after the training we can print a word embedding to see what the result of this training looks like
print('Word Embedding:')
print(model.wv[0])

print("\n*----------*----------*----------*\n")

# implementing the networkx graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(0)

# generate a random graph using erdos_renyi_graph with a fixed number of nodes (10)
# and a predefined probability of creating an edge between two nodes (0.3)
G = nx.erdos_renyi_graph(10, 0.3, seed = 1, directed = False)

# plotting this random graph
plt.figure(dpi = 300)
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed = 0),
                 node_size = 600,
                 cmap = 'coolwarm',
                 font_size = 14,
                 font_color = 'white'
                 )

plt.savefig('random_graph.png')

# the method to implement random walks
def random_walk(start, length):
    """
    This function takes two parameters:
    the starting node (start) and the length of the walk (length). At every step, we randomly
    select a neighboring node (using np.random.choice) until the walk is complete
    """
    walk = [str(start)] # starting node

    for i in range(length):
        neighbors = [node for node in G.neighbors(start)]
        next_node = np.random.choice(neighbors, 1)[0]
        walk.append(str(next_node))
        start = next_node
    
    return walk

# starting node = 0 and the length = 10
print(random_walk(0, 10))

print("\n*----------*----------*----------*\n")

################################################

# implementing DeepWalk

# The dataset we will use is Zachary’s Karate Club. It simply represents the relationships within a karate
# club studied by Wayne W. Zachary in the 1970s. It is a kind of social network where every node is a
# member, and members who interact outside the club are connected.

# In this example, the club is divided into two groups: we would like to assign the right group to every
# member (node classification) just by looking at their connections:

# import the dataset
G = nx.karate_club_graph()

# convert string class labels into numerical values, ex: (Mr. Hi = 0, Officer = 1)
labels = []
for node in G.nodes:
    label = G.nodes[node]['club']
    labels.append(1 if label == 'Officer' else 0)


# plot the graph using our new labels
plt.figure(figsize=(12, 12), dpi = 300)
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=0),
                 node_color = labels,
                 node_size = 800,
                 cmap = 'coolwarm',
                 font_size = 14,
                 font_color = 'white'
                 )
plt.savefig('DeepWalk_Graph.png')

# generating dataset -> the random walks
# to be exhaustive, will use 80 random walks of length 10 for every node in the graph.
walks = []
for node in G.nodes:
    for _ in range(80):
        walks.append(random_walk(node, 10))

print(walks[0])

# finally implementing Word2Vec using skipgram model with h-softmax
model = Word2Vec(walks,
                 hs = 1, # hierarchical softmax
                 sg = 1, # skipgram
                 vector_size=100,
                 window=10,
                 workers=2,
                 seed=0
                 )

# then we train the model on the random walks that we generated a while ago.
model.train(walks, total_examples=model.corpus_count, epochs=30, report_delay=1)

# now that our model is trained, let's see its different applicatons.
# the first one allows us to find the most similar nodes to a given one (cosine similarity)
print(f'Nodes that are the most similar to node 0:')
for similarity in model.wv.most_similar(positive=['0']):
    print(f' {similarity}')

# another important applicaton is calculating the similarity scores between 
# two nodes
print(f"Similarity between node 0 and 4: {model.wv.similarity('0', '4')}")

# plot the embeddings using t-SNE
from sklearn.manifold import TSNE

# creating two arrays
# one to store the embeddings
# the other one to store the labels
nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(len(model.wv))])
labels = np.array(labels)

# train the t-SNE model with two dimensions
tsne = TSNE(n_components=2,
            learning_rate='auto',
            init='pca',
            random_state=0).fit_transform(nodes_wv)

# plot the 2D vectors produced by the t-SNE model with the corresponding labels
plt.figure(figsize=(6, 6), dpi = 300)
plt.scatter(tsne[:, 0], tsne[:, 1], s = 100, c = labels, cmap = 'coolwarm')
plt.savefig('2D Vectors from t-SNE with labels.png')