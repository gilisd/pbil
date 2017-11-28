# Using PBIL as alternative for gradient descent

Traditionally neural nets are trained using a learning methodology. The outcome of the network is compared to the expected result and the network is altered (by tuning the weights) in the direction where hopefully a better solution exists.

[Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is the most popular of these methodologies. The network is transformed in the direction of a better solution by taking the derivative of the function represented by the network.

Another approach is the use of evolution instead of learning. In this case no processor time is spent constructing the direction to adapt. The network is put in a game of survival of the fittest where the set of weights act as individuals.

We can not use [genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm) due to the structure of the solution space. We can however use [Population Based Incremental Learning (PBIL)](https://en.wikipedia.org/wiki/Population-based_incremental_learning) in which the genotype as a whole evolves over time. The population serves as an input to decide in which direction to evolve.

## Model

We created a new Node class tuned to the flexibility to map from and to a list of weights.

First we build up an initial population at random. We pick the best as initial genotype.

Based on the genotype, we create a new population as variations on this type. We pick out the best as genotype for the next generation. This loop enables the evolution.

To preserve memory we don't keep the entire generation in memory. A single individual gets created, if it performs better than all the previous individuals it is kept, otherwise it gets discarded immediately.

The codeis an example with the minst dataset. Performing rather slow as compared to traditional algorithms it shows a promising evolution:

Note that performance was not an objective. This code should only be seen as POC. Don't expect a finished product.

## Result

1 layer, dense: 56.3% correctly classified with 90% confidence.
```
Generation    0: deviation: 0.10936402560765525 histogram: [694, 129, 75, 45, 32, 14, 7, 3,  1,   0,   0], by digit [ 0,  21, 0,  0,  2,  1,  0,  1, 0,  0]
Generation  500: deviation: 0.4196661245922093  histogram: [545,  11,  8, 10,  3, 10, 8, 4, 19, 358,  24], by digit [71, 109, 0, 57, 48,  2, 52, 61, 0, 23]
Generation 1000: deviation: 0.47178381697742683 histogram: [504,   6,  8,  8,  2,  6, 7, 2,  4, 380,  73], by digit [68, 107, 0, 63, 55,  2, 63, 66, 0, 48]
Generation 1500: deviation: 0.5076223652719496  histogram: [478,   7,  2,  2,  3,  1, 7, 1,  9, 367, 123], by digit [71, 108, 0, 76, 46,  7, 69, 71, 0, 60]
Generation 2000: deviation: 0.5244562119671657  histogram: [451,   9,  7,  6,  2,  5, 4, 3,  9, 362, 142], by digit [68, 112, 0, 62, 70,  9, 72, 76, 0, 56]
Generation 2500: deviation: 0.532450942031954   histogram: [445,  10,  8,  2,  5,  4, 1, 5,  4, 369, 147], by digit [68, 106, 0, 58, 67, 26, 70, 71, 0, 64]
Generation 3000: deviation: 0.549411075520184   histogram: [432,   7,  3,  1,  4,  7, 5, 5, 10, 375, 151], by digit [72, 109, 0, 75, 64, 15, 73, 77, 0, 68]
Generation 3500: deviation: 0.5647468114273075  histogram: [418,   3,  7,  4,  3,  2, 3, 9,  4, 358, 189], by digit [73, 108, 0, 72, 71, 18, 78, 81, 0, 64]
Generation 4000: deviation: 0.5701826147649018  histogram: [410,   9,  4,  3,  1,  5, 5, 3, 10, 369, 181], by digit [73, 110, 0, 71, 65, 22, 77, 82, 0, 73]
Generation 4500: deviation: 0.574109516848085   histogram: [410,   5,  3,  5,  3,  4, 3, 2,  4, 371, 190], by digit [67, 109, 0, 72, 71, 27, 76, 81, 0, 71]
Generation 4999: deviation: 0.5755191851288274  histogram: [408,   6,  4,  5,  1,  2, 5, 4,  2, 367, 196], by digit [67, 111, 0, 72, 73, 25, 76, 81, 0, 71]
```

2 layer, dense (800 nodes internally)
```
Generation    0: deviation: 0.10039366693607508 histogram: [501, 499,   0,  0,  0,  0,  0,  0,   0,   0, 0], by digit [ 0,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   10: deviation: 0.10422098895676248 histogram: [366, 634,   0,  0,  0,  0,  0,  0,   0,   0, 0], by digit [ 0,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   20: deviation: 0.10725636981902958 histogram: [322, 678,   0,  0,  0,  0,  0,  0,   0,   0, 0], by digit [ 0,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   30: deviation: 0.11271586796091984 histogram: [291, 709,   0,  0,  0,  0,  0,  0,   0,   0, 0], by digit [ 0,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   40: deviation: 0.11805130954033943 histogram: [336, 643,  21,  0,  0,  0,  0,  0,   0,   0, 0], by digit [ 0,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   50: deviation: 0.1289431307078555  histogram: [348, 572,  45, 23,  6,  6,  0,  0,   0,   0, 0], by digit [ 6,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   60: deviation: 0.14559751518856479 histogram: [392, 510,  23, 13, 20, 11, 18,  4,   8,   1, 0], by digit [42,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   70: deviation: 0.16018194330397056 histogram: [354, 472,  90, 19, 17, 12, 17,  7,   6,   6, 0], by digit [48,   0,  0,  0,  0,  0,  0,  0,  0,  0]
Generation   80: deviation: 0.17386690881146222 histogram: [352, 394, 136, 53, 12, 13, 19,  9,   6,   6, 0], by digit [52,   0,  0,  0,  0,  0,  1,  0,  0,  0]
Generation   90: deviation: 0.18865984886604822 histogram: [363, 321, 158, 70, 27, 16, 12, 16,   7,  10, 0], by digit [58,   0,  1,  0,  0,  0,  0,  2,  0,  0]
Generation  100: deviation: 0.20445671993544984 histogram: [342, 300, 142, 83, 57, 33, 19, 10,   7,   7, 0], by digit [43,   0,  2,  1,  0,  0,  5, 25,  0,  0]
Generation  200: deviation: 0.3949521974728284  histogram: [297, 107,  68, 58, 70, 78, 83, 78,  91,  70, 0], by digit [72,  87, 29, 69,  0,  0, 59, 61, 23,  0]
Generation  300: deviation: 0.5350980981176207  histogram: [229,  63,  50, 38, 48, 59, 62, 90, 126, 235, 0], by digit [73, 111, 60, 78, 67,  0, 72, 71, 40,  0]
Generation  400: deviation: 0.6112504536291148  histogram: [196,  63,  36, 31, 40, 39, 41, 61, 115, 378, 0], by digit [75, 115, 66, 85, 85,  0, 73, 77, 53,  5]
Generation  500: deviation: 0.6377950666113864  histogram: [178,  51,  32, 37, 35, 42, 51, 59,  96, 419, 0], by digit [77, 117, 80, 80, 67,  0, 69, 67, 57, 53]

```
