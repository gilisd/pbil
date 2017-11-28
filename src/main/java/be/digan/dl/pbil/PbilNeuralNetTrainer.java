package be.digan.dl.pbil;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

// Reduce memory load by evaluating item by item
public class PbilNeuralNetTrainer {
    private static final int GENERATION_COUNT = 5000;
    private static final int POPULATION = 100;
    private static final int PLAY_COUNT = 50;
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

        // Pick initial genotype by random search
        double[] bestWeights = getRandom();
        for (int i = 0; i<POPULATION; i++) {
            double[] newWeights = getRandom();
            if (isBetter(newWeights, bestWeights)) {
                bestWeights = newWeights;
            }
        }
        logQuality(0, bestWeights);
        for (int i = 0; i<GENERATION_COUNT; i++) {
            double[] betterWeights = bestWeights;
            for (int j = 0; j<POPULATION; j++) {

                double[] newWeights = generatePbil(i, bestWeights);
                if (isBetter(newWeights, betterWeights)) {
                    betterWeights = newWeights;
                }
            }
            bestWeights = betterWeights;
            logQuality(i, bestWeights);
        }
        System.out.println(" histogram count calculated with confidence : [0,.1,.2,...]" );

    }

    private void logQuality(int generation, double[] weights) {
        if (generation%10 !=0) {
            System.out.println("Generation " + generation +": no log");
            return;
        }
        int testCount = 1000; //mnist_test.length
        double totalQuality = 0;
        int[] histo=new int[11];
        int[] byDigit=new int[10];
        for (int j = 0; j < testCount; j++) {
            int element = j;
            double result = net.calculate(weights, mnist_test[element].getInput())[mnist_test[element].getOutput()];
            double quality = result;
            histo[(int)(result *10)]=histo[(int)(result *10)]+1;
            byDigit[mnist_test[element].getOutput()]+=result>.5?1:0;
            // quality: closer to one is better
            totalQuality += quality;
        }
        double quality = totalQuality/ testCount;
        System.out.println("Generation " + generation +": deviation: " + quality + " histogram: " + Arrays.toString(histo) + ", by digit " + Arrays.toString(byDigit));

    }
    private boolean isBetter(double[] newWeights, double[] oldWeights) {
        double total = 0;
        for (int j = 0; j < PLAY_COUNT; j++) {
            int element = random.nextInt(mnist_data.length);
            double qualityOld = net.calculate(oldWeights, mnist_data[element].getInput())[mnist_data[element].getOutput()];
            double qualityNew = net.calculate(newWeights, mnist_data[element].getInput())[mnist_data[element].getOutput()];
            // quality: closer to one is better
            total += qualityNew - qualityOld;
        }
        return total > 0;
    }

    private double[] getRandom() {
        return IntStream.range(0, chromosoneSize).mapToDouble(k -> (Math.random() * 2 - 1) / 784).toArray();
    }




    private double[] generatePbil(int generation, double[] best) {
        double deviation = .001 + (double) (GENERATION_COUNT - generation) / GENERATION_COUNT / 2;
        return IntStream.range(0, best.length).mapToDouble(i -> best[i] + (Math.random() * 2 - 1) / 784 * deviation).toArray();
    }




 }
