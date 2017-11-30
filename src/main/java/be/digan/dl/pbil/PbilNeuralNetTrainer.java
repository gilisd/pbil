package be.digan.dl.pbil;

import org.apache.log4j.Logger;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

// Reduce memory load by evaluating item by item
public class PbilNeuralNetTrainer {
    private static final Logger LOG = Logger.getLogger(PbilNeuralNetTrainer.class);
    private static final int GENERATION_COUNT = 100;
    private static final int POPULATION = 100;
    private static final int BATCH_SIZE = 50;
    private Experiment[] mnist_data;

    private Random random = new Random();
    public int chromosoneSize;
    private Experiment[] mnist_test;
    private NeuralNet net;

    public void trainNetwork(NeuralNet net, Experiment[] mnist_data, Experiment[] mnist_test) {
        this.net = net;
        this.mnist_data = mnist_data;
        chromosoneSize = net.getWeightCount();
        this.mnist_test = mnist_test;

        // select batch
        int[] batch = getBatch();
        // Pick initial genotype by random search
        double[] genotype = getInitialGenotype(batch);

        for (int i = 1; i<=GENERATION_COUNT; i++) {
            genotype = nextGeneration(genotype, i);
        }

        LOG.info(" histogram count calculated with confidence : [0,0.1,0.2,...,0.9,1]" );

    }

    private double[] nextGeneration(double[] genotype, int generation) {
        int[] batch = getBatch();
        double[] betterWeights = genotype;
        double better = calculateQuality(batch, betterWeights);
        for (int j = 0; j<POPULATION; j++) {

            double[] newWeights = generatePbil(generation, genotype);
            double newQuality = calculateQuality(batch, newWeights);
            double difference = newQuality - better;
            boolean isBetter = difference > 0;
            if (isBetter) {
                betterWeights = newWeights;
                better = newQuality;
            }
        }
        genotype = betterWeights;
        logQuality(generation, genotype, better/BATCH_SIZE);
        return genotype;
    }

    private double[] getInitialGenotype(int[] batch) {
        double[] bestWeights = getRandom();
        double best = calculateQuality(batch, bestWeights);
        for (int i = 0; i<POPULATION; i++) {
            double[] newWeights = getRandom();
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

    private double calculateQuality(int[] batch, double[] weights) {
        return Arrays.stream(batch).parallel().mapToDouble(element -> net.calculate(weights, this.mnist_data[element].getInput())[this.mnist_data[element].getOutput()]).sum();
    }

    private int[] getBatch() {
        return IntStream.range(0, BATCH_SIZE).map(i-> random.nextInt(mnist_data.length)).toArray();
    }

    private void logQuality(int generation, double[] weights, double estimatedQuality) {
        if ( (generation < 10) || ((generation<100) && (generation%10==0)) || (generation%100 == 0)) {
            int testCount = 1000; //mnist_test.length
            double totalQuality = 0;
            int[] histo = new int[11];
            int[] byDigit = new int[10];
            for (int j = 0; j < testCount; j++) {
                int element = j;
                double result = net.calculate(weights, mnist_test[element].getInput())[mnist_test[element].getOutput()];
                double quality = result;
                histo[(int) (result * 10)] = histo[(int) (result * 10)] + 1;
                byDigit[mnist_test[element].getOutput()] += result > .5 ? 1 : 0;
                // quality: closer to one is better
                totalQuality += quality;
            }
            double quality = totalQuality / testCount;
            LOG.info("Generation " + generation + ": deviation: " + quality + " histogram: " + Arrays.toString(histo) + ", by digit " + Arrays.toString(byDigit));
        } else {
            LOG.debug("Generation " + generation + ": estimated deviation: " + estimatedQuality);
        }
    }

    private double[] getRandom() {
        return IntStream.range(0, chromosoneSize).mapToDouble(k -> (Math.random() * 2 - 1) / 784).toArray();
    }




    private double[] generatePbil(int generation, double[] best) {
        double deviation = .001 + (double) (GENERATION_COUNT - generation) / GENERATION_COUNT / 2;
        return IntStream.range(0, best.length).mapToDouble(i -> best[i] + (Math.random() * 2 - 1) / 784 * deviation).toArray();
    }




 }
