package be.digan.dl.pbil;

import org.apache.log4j.Logger;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;

// Reduce memory load by evaluating item by item
public class PbilNeuralNetTrainer {
    private static final Logger LOG = Logger.getLogger(PbilNeuralNetTrainer.class);
    private static final int GENERATION_COUNT = 10000;
    private static final int POPULATION = 50;
    private static final int BATCH_SIZE = 250;
    private static final int TEST_COUNT = 1000;
    private Experiment[] mnist_data;

    private Random random = new Random();
    public int chromosoneSize;
    private Experiment[] mnist_test;
    private NeuralNet net;
    private int[] weightStructure;
    private int[] small_list_result;
    private int[] full_list_result;

    public void trainNetwork(NeuralNet net, Experiment[] mnist_data, Experiment[] mnist_test) {
        chromosoneSize = net.getWeightCount();
        weightStructure = net.getWeightStructure();
        LOG.info("NETWORK STRUCTURE: ");
        LOG.info(" - chromosoneSize: " + chromosoneSize);
        LOG.info(" - layers: " + net.toString());
        LOG.info("TRAINING PARAMETERS: ");
        LOG.info("  - generations: " + GENERATION_COUNT);
        LOG.info("  - population: " + POPULATION);
        LOG.info("  - batch size: " + BATCH_SIZE);
        this.net = net;
        this.mnist_data = mnist_data;
        this.mnist_test = mnist_test;

        Map<Integer, List<Integer>> temp = IntStream.range(0, TEST_COUNT).mapToObj(i -> i).collect(Collectors.groupingBy((Integer i) -> mnist_test[i].getOutput()));
        small_list_result = temp.entrySet().stream().mapToInt(i -> i.getValue().size()).toArray();
        temp = IntStream.range(0, mnist_test.length).mapToObj(i -> i).collect(Collectors.groupingBy((Integer i) -> mnist_test[i].getOutput()));
        full_list_result = temp.entrySet().stream().mapToInt(i -> i.getValue().size()).toArray();

        // select batch
        int[] batch = getBatch();
        // Pick initial genotype by random search
        long[] genotype = getInitialGenotype(batch);

        for (int i = 1; i <= GENERATION_COUNT; i++) {
            genotype = nextGeneration(genotype, i);
        }

        LOG.info("histogram count calculated with confidence : [0,0.1,0.2,...,0.9,1]");

        LOG.info("winner: " + Arrays.toString(genotype));
        logQuality(99999900, genotype, 0, mnist_test.length, full_list_result);
    }

    private long[] nextGeneration(long[] genotype, int generation) {
//        if (Arrays.stream(genotype).map(i -> Math.abs(i)).average().getAsDouble() > NeuralNet.FACTOR * 2 / 784) { //NORMALIZE
//            genotype = Arrays.stream(genotype).map(y -> y / 2).toArray();
//            LOG.warn("DIVISION");
//        }
        int[] batch = getBatch();
        long[] currentBestWeights = genotype;
        double currentBestQuality = calculateQuality(batch, currentBestWeights);
        for (int j = 0; j < POPULATION; j++) {

            long[] newWeights = generatePbil(generation, genotype);
            double newQuality = calculateQuality(batch, newWeights);
            double difference = newQuality - currentBestQuality;
            boolean isBetter = difference > 0;
            if (isBetter) {
                currentBestWeights = newWeights;
                currentBestQuality = newQuality;
            }
        }
        genotype = currentBestWeights;
        logQuality(generation, genotype, currentBestQuality / BATCH_SIZE, TEST_COUNT, small_list_result);
        return genotype;
    }

    private long[] getInitialGenotype(int[] batch) {
        long[] bestWeights = getRandom();
        double best = calculateQuality(batch, bestWeights);
        for (int i = 0; i < POPULATION; i++) {
            long[] newWeights = getRandom();
            double newQuality = calculateQuality(batch, newWeights);
            double difference = newQuality - best;
            boolean isBetter = difference > 0;
            if (isBetter) {
                bestWeights = newWeights;
                best = newQuality;
            }
        }
        logQuality(0, bestWeights, best, TEST_COUNT, small_list_result);
        return bestWeights;
    }

    private double calculateQuality(int[] batch, long[] weights) {
        double quality = Arrays.stream(batch).parallel().mapToDouble(element -> calculateResult(element, weights)).sum();
        return quality;
    }

    private double calculateResult(int element, long[] weights) {
        long[] calculate = net.calculate(weights, this.mnist_data[element].getInput());
        double confidence = (double) calculate[this.mnist_data[element].getOutput()] / NeuralNet.FACTOR;
        int found = IntStream.range(0, calculate.length)
                .reduce((a,b)->calculate[a]<calculate[b]? b: a)
                .getAsInt();
        return confidence + (found==mnist_data[element].getOutput()?0:-1);
    }

    private int[] getBatch() {
        return IntStream.range(0, BATCH_SIZE).map(i -> random.nextInt(mnist_data.length)).toArray();
    }

    private void logQuality(int generation, long[] weights, double estimatedQuality, int testCount, int[] compare) {
        if ((generation < 10) || ((generation < 1000) && (generation % 10 == 0)) || (generation % 100 == 0)) {
            double totalQuality = 0;
            int[] histo = new int[11];
            int[] byDigit = new int[10];
            for (int j = 0; j < testCount; j++) {
                int element = j;
                long[] calculate = net.calculate(weights, mnist_test[element].getInput());
                double result = (double) calculate[mnist_test[element].getOutput()] / NeuralNet.FACTOR;
                double quality = result;

                int found = IntStream.range(0, calculate.length)
                        .reduce((a,b)->calculate[a]<calculate[b]? b: a)
                        .getAsInt();
                histo[(int) (result * 10)] = histo[(int) (result * 10)] + 1;
                byDigit[mnist_test[element].getOutput()] += found == mnist_test[element].getOutput() ? 1 : 0;
                // quality: closer to one is better
                totalQuality += quality;
            }
            double quality = totalQuality / testCount;
            double averageAbs = Arrays.stream(weights).map(i -> Math.abs(i)).average().getAsDouble();
            double average = Arrays.stream(weights).average().getAsDouble();
            long min = Arrays.stream(weights).min().getAsLong();
            long max = Arrays.stream(weights).max().getAsLong();
            LOG.info("Generation " + String.format("%5d", generation) + ": average abs: " + String.format("%10f", averageAbs) + ": average: " + String.format("%10f", average) + ", max: " + max + ", min: " + min);
            LOG.info("Generation " + String.format("%5d", generation) + ": random: " + mnist_test[500].getOutput() + " " + writeArray(net.calculate(weights, mnist_test[500].getInput())));
            LOG.info("Generation " + String.format("%5d", generation) + ": deviation: " + String.format("%10f", quality) + " histogram: " + writeArray(histo) + ", by digit " + writeArray(byDigit, compare));
        } else {
            LOG.debug("Generation " + String.format("%5d", generation) + ": estimated: " + String.format("%10f", estimatedQuality));
        }
    }

    private String writeArray(int[] histo) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%3d%%", (i *100 / sum))).collect(joining(", ")) + "]";
    }

    private String writeArray(int[] histo, int[] compare) {
        return "[" + IntStream.range(0,10).mapToObj(i -> String.format("%3d%%", (histo[i] *100 / compare[i]))).collect(joining(", ")) + "]";
    }

    private String writeArray(long[] histo) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%3d%%", (i *100 / sum))).collect(joining(", ")) + "]";
    }
    private String writeArray(int[] histo, int digits) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%" + digits +"d", i)).collect(joining(", ")) + "]";
    }

    private String writeArray(long[] histo, int digits) {
        long sum = Arrays.stream(histo).sum();
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%" + digits +"d", i)).collect(joining(", ")) + "]";
    }

    private long[] getRandom() {
        return IntStream.range(0, chromosoneSize).mapToLong(k -> (random.nextInt(NeuralNet.FACTOR * 2) - NeuralNet.FACTOR) / 784).toArray();
    }


    private long[] generatePbil(int generation, long[] best) {
        int deviation = (int)((.001 + (double) (GENERATION_COUNT - generation) / GENERATION_COUNT ) * NeuralNet.FACTOR / 784) *5;
     //   int deviation = NeuralNet.FACTOR * 2 / 784 ;
        int geneToChange = random.nextInt(weightStructure[weightStructure.length - 1] + 1);
        long[] result = IntStream.range(0, best.length).mapToLong(i -> best[i] + (weightStructure[i] != geneToChange ? 0 : (random.nextInt(deviation * 2 +1) - deviation))).toArray();
        return result;
    }
   private long[] generatePbilOriginal(int generation, long[] best) {
        int deviation = (int)((.001 + (double) (GENERATION_COUNT - generation) / GENERATION_COUNT ) * NeuralNet.FACTOR / 784) *5;
     //   int deviation = NeuralNet.FACTOR * 2 / 784 ;
        Boolean[] keepGene = IntStream.range(0, weightStructure[weightStructure.length - 1] + 1).mapToObj(i -> (Boolean) (Math.random() > .8)).toArray(Boolean[]::new);
        long[] result = IntStream.range(0, best.length).mapToLong(i -> best[i] + (keepGene[weightStructure[i]] ? 0 : (random.nextInt(deviation * 2 +1) - deviation))).toArray();
//        long[] result =  IntStream.range(0, best.length).mapToLong(i -> best[i] + ((random.nextInt(deviation * 2)-deviation))).toArray();
//        if (Arrays.stream(result).map(i -> Math.abs(i)).average().getAsDouble() > NeuralNet.FACTOR * 4 / 784) { //NORMALIZE
//            result = Arrays.stream(result).map(y -> y/2).toArray();
//        }
        return result;
    }


}
