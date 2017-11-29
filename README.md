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
Generation  600: deviation: 0.6508968010596488  histogram: [187,  42,  43, 22, 33, 27, 48, 47,  84, 467, 0], by digit [71, 116, 77, 81, 68,  0, 71, 61, 65, 63]
Generation  700: deviation: 0.6651627717890992  histogram: [197,  29,  25, 31, 32, 30, 41, 42,  77, 496, 0], by digit [74, 117, 75, 82, 80,  0, 68, 69, 58, 63]
Generation  800: deviation: 0.6768881978703372  histogram: [192,  27,  32, 32, 19, 26, 39, 45,  67, 521, 0], by digit [75, 120, 78, 84, 77,  1, 71, 84, 51, 57]
Generation  900: deviation: 0.6907121488701845  histogram: [180,  33,  30, 29, 17, 29, 34, 40,  70, 538, 0], by digit [75, 115, 79, 88, 82, 14, 68, 82, 48, 60]
Generation 1000: deviation: 0.6895177657576232  histogram: [179,  37,  31, 25, 22, 27, 32, 47,  66, 534, 0], by digit [71, 114, 81, 79, 85, 18, 71, 80, 51, 56]
Generation 1100: deviation: 0.6966183266228606  histogram: [188,  31,  26, 18, 24, 28, 28, 36,  62, 559, 0], by digit [71, 119, 77, 77, 91, 22, 73, 82, 48, 53]
Generation 1200: deviation: 0.7022039992289594  histogram: [194,  32,  21, 18, 13, 25, 32, 37,  58, 570, 0], by digit [75, 123, 92, 84, 68, 20, 65, 75, 49, 71]
Generation 1300: deviation: 0.7040830591492436  histogram: [194,  35,  22, 17, 21, 20, 20, 30,  46, 595, 0], by digit [76, 120, 80, 82, 69, 19, 71, 81, 50, 63]
Generation 1400: deviation: 0.7105314711518353  histogram: [187,  29,  29, 21, 15, 17, 24, 28,  49, 601, 0], by digit [78, 116, 84, 81, 77, 15, 70, 81, 54, 63]
Generation 1500: deviation: 0.7226030466921316  histogram: [175,  28,  25, 22, 15, 20, 27, 34,  48, 606, 0], by digit [77, 117, 87, 74, 86, 29, 73, 82, 56, 54]
Generation 1600: deviation: 0.7226022333304046  histogram: [185,  28,  22, 17, 17, 19, 22, 26,  41, 623, 0], by digit [72, 120, 71, 74, 89, 31, 72, 84, 64, 54]
Generation 1700: deviation: 0.725926436841985   histogram: [184,  25,  21, 16, 16, 21, 29, 21,  41, 626, 0], by digit [74, 124, 88, 70, 91, 26, 71, 80, 59, 55]
Generation 1800: deviation: 0.7284989261069577  histogram: [181,  29,  19, 19, 16,  9, 31, 24,  50, 622, 0], by digit [71, 114, 85, 81, 84, 30, 70, 80, 61, 60]
Generation 1900: deviation: 0.7183109848839716  histogram: [196,  24,  19, 15, 18, 25, 24, 18,  30, 631, 0], by digit [72, 115, 83, 76, 84, 27, 70, 75, 61, 65]
Generation 2000: deviation: 0.722535854533229   histogram: [205,  23,  17, 16, 15, 10, 14, 23,  28, 649, 0], by digit [75, 116, 82, 77, 79, 28, 74, 70, 54, 69]
Generation 2100: deviation: 0.7177121244360585  histogram: [205,  18,  18, 21, 17, 18, 19, 16,  29, 639, 0], by digit [72, 117, 80, 81, 84, 28, 71, 74, 45, 69]
Generation 2200: deviation: 0.7230723669339371  histogram: [204,  22,  19, 13, 12, 23, 12, 15,  31, 649, 0], by digit [74, 119, 83, 84, 77, 30, 74, 78, 48, 63]
Generation 2300: deviation: 0.7372771351544685  histogram: [192,  14,  23, 12, 18, 16, 19, 16,  30, 660, 0], by digit [74, 119, 90, 78, 87, 26, 73, 76, 58, 60]
Generation 2400: deviation: 0.733517780660773   histogram: [200,  17,  18, 12, 14, 15, 14, 28,  22, 660, 0], by digit [77, 119, 79, 83, 79, 30, 72, 76, 58, 66]
Generation 2500: deviation: 0.7369012306055993  histogram: [192,  31,  11, 10, 12, 16, 17, 18,  28, 665, 0], by digit [72, 119, 83, 85, 78, 32, 73, 74, 62, 66]
Generation 2600: deviation: 0.7439386493517679  histogram: [184,  19,  16, 20, 16, 12, 20, 17,  26, 669, 1], by digit [73, 118, 87, 86, 86, 34, 75, 75, 54, 57]
Generation 2700: deviation: 0.7484673184712323  histogram: [181,  23,  19, 16, 12,  9, 16, 20,  17, 686, 1], by digit [68, 118, 88, 79, 82, 36, 75, 76, 61, 66]
Generation 2800: deviation: 0.7453023132967196  histogram: [183,  28,  12, 16, 14, 11, 14, 19,  29, 674, 0], by digit [70, 113, 89, 76, 80, 40, 76, 70, 62, 71]
Generation 2900: deviation: 0.74611369940527    histogram: [183,  18,  21, 20, 15, 12, 10, 15,  29, 676, 1], by digit [74, 116, 86, 79, 80, 41, 73, 74, 58, 62]
Generation 3000: deviation: 0.7451900175861413  histogram: [192,  17,  17, 16, 11, 13, 15, 14,  23, 681, 1], by digit [72, 116, 83, 88, 81, 35, 77, 73, 57, 65]
Generation 3100: deviation: 0.7512890078341684  histogram: [197,  14,  15, 13,  5,  9, 12, 13,  34, 688, 0], by digit [70, 118, 86, 84, 80, 38, 74, 76, 65, 65]
Generation 3200: deviation: 0.7547021389068076  histogram: [189,  23,  12, 10,  8, 13, 10, 11,  27, 696, 1], by digit [72, 119, 84, 89, 81, 39, 77, 75, 57, 65]
Generation 3300: deviation: 0.7571545096058353  histogram: [178,  24,  18,  9,  9, 19,  7, 18,  21, 696, 1], by digit [64, 119, 91, 84, 82, 45, 74, 75, 59, 69]
Generation 3400: deviation: 0.7605148966568583  histogram: [186,  20,   9, 12,  6, 15, 10, 17,  22, 702, 1], by digit [71, 119, 92, 82, 81, 40, 73, 75, 63, 71]
Generation 3500: deviation: 0.7587183800986148  histogram: [192,  14,  13, 11,  8, 11, 11,  9,  17, 713, 1], by digit [68, 118, 92, 84, 81, 37, 75, 77, 63, 67]
Generation 3600: deviation: 0.7579718047999797  histogram: [196,  16,  14,  5,  9,  6,  8, 14,  23, 708, 1], by digit [68, 119, 89, 82, 83, 40, 77, 77, 60, 65]
Generation 3700: deviation: 0.7588333607220559  histogram: [192,  16,  14, 10,  7,  8,  9, 12,  28, 702, 2], by digit [71, 117, 90, 82, 82, 40, 76, 77, 62, 64]

```
