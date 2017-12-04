package be.digan.dl.pbil;

import org.apache.log4j.Logger;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;

// Reduce memory load by evaluating item by item
public class PbilNeuralNetTrainer {
    private static final Logger LOG = Logger.getLogger(PbilNeuralNetTrainer.class);
    private static final int GENERATION_COUNT = 20000;
    private static final int POPULATION = 50;
    private static final int BATCH_SIZE = 250;
    private Experiment[] mnist_data;

    private Random random = new Random();
    public int chromosoneSize;
    private Experiment[] mnist_test;
    private NeuralNet net;
    private int[] weightStructure;

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

        // select batch
        int[] batch = getBatch();
        // Pick initial genotype by random search
        long[] genotype = getInitialGenotype(batch);

        for (int i = 1; i<=GENERATION_COUNT; i++) {
            genotype = nextGeneration(genotype, i);
        }

        LOG.info(" histogram count calculated with confidence : [0,0.1,0.2,...,0.9,1]" );

    }

    private long[] nextGeneration(long[] genotype, int generation) {
        int[] batch = getBatch();
        long[] currentBestWeights = genotype;
        double currentBestQuality = calculateQuality(batch, currentBestWeights);
        for (int j = 0; j<POPULATION; j++) {

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
        logQuality(generation, genotype, currentBestQuality/BATCH_SIZE);
        return genotype;
    }

    private long[] getInitialGenotype(int[] batch) {
        long[] bestWeights = getRandom();
        double best = calculateQuality(batch, bestWeights);
        for (int i = 0; i<POPULATION; i++) {
            long[] newWeights = getRandom();
            double newQuality = calculateQuality(batch, newWeights);
            double difference = newQuality - best;
            boolean isBetter = difference > 0;
            if (isBetter) {
                bestWeights = newWeights;
                best = newQuality;
            }
        }
        logQuality(0, bestWeights, best);
        return bestWeights;
    }

    private double calculateQuality(int[] batch, long[] weights) {
        return Arrays.stream(batch).parallel().mapToDouble(element -> (double)net.calculate(weights, this.mnist_data[element].getInput())[this.mnist_data[element].getOutput()] / NeuralNet.FACTOR).sum();
    }

    private int[] getBatch() {
        return IntStream.range(0, BATCH_SIZE).map(i-> random.nextInt(mnist_data.length)).toArray();
    }

    private void logQuality(int generation, long[] weights, double estimatedQuality) {
        if ( (generation < 10) || ((generation<100) && (generation%10==0)) || (generation%100 == 0)) {
            int testCount = 1000; //mnist_test.length
            double totalQuality = 0;
            int[] histo = new int[11];
            int[] byDigit = new int[10];
            for (int j = 0; j < testCount; j++) {
                int element = j;
                double result = (double)net.calculate(weights, mnist_test[element].getInput())[mnist_test[element].getOutput()] / NeuralNet.FACTOR;
                double quality = result;
                histo[(int) (result * 10)] = histo[(int) (result * 10)] + 1;
                byDigit[mnist_test[element].getOutput()] += result > .5 ? 1 : 0;
                // quality: closer to one is better
                totalQuality += quality;
            }
            double quality = totalQuality / testCount;
            LOG.info("Generation " + String.format("%5d" ,generation) + ": deviation: " + String.format("%10f" ,quality) + " histogram: " + writeArray(histo, 3) + ", by digit " + writeArray(byDigit, 3));
        } else {
            LOG.debug("Generation " + String.format("%5d" ,generation) + ": estimated: " + String.format("%10f" ,estimatedQuality));
        }
    }

    private String writeArray(int[] histo, int digits) {
        return "[" + Arrays.stream(histo).mapToObj(i -> String.format("%"+digits+"d" ,i)).collect(joining(", ")) + "]";
    }

    private long[] getRandom() {
        return IntStream.range(0, chromosoneSize).mapToLong(k -> (random.nextInt(NeuralNet.FACTOR * 2) - NeuralNet.FACTOR) / 784).toArray();
    }




    private long[] generatePbil(int generation, long[] best) {
//        int deviation = (int)((.001 + (double) (GENERATION_COUNT - generation) / GENERATION_COUNT ) * NeuralNet.FACTOR / 784) *2;
        int deviation = NeuralNet.FACTOR *2 / 784;
        Boolean[] keepGene =  IntStream.range(0, weightStructure[weightStructure.length-1]+1).mapToObj(i -> (Boolean)(Math.random() > .8)).toArray(Boolean[]::new);
        long[] result =  IntStream.range(0, best.length).mapToLong(i -> best[i] + (keepGene[weightStructure[i]]?0:(random.nextInt(deviation * 2)-deviation))).toArray();
//        long[] result =  IntStream.range(0, best.length).mapToLong(i -> best[i] + ((random.nextInt(deviation * 2)-deviation))).toArray();
//        if (Arrays.stream(result).map(i -> Math.abs(i)).average().getAsDouble() > NeuralNet.FACTOR * 16 / 784) { //NORMALIZE
//            result = Arrays.stream(result).map(y -> y/2).toArray();
//        }
        return result;
    }




 }
