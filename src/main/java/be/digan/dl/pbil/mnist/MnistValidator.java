package be.digan.dl.pbil.mnist;

import be.digan.dl.pbil.neuralnet.Experiment;
import be.digan.dl.pbil.neuralnet.NeuralNet;
import be.digan.dl.pbil.pbil.TrainingParameters;
import be.digan.dl.pbil.pbil.Validator;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class MnistValidator implements Validator {
    private Experiment[] data;
    private Random random = new Random();
    private NeuralNet net;
    int batchSize;
    private int[] batch;

    public MnistValidator(Experiment[] data, NeuralNet net, TrainingParameters parameters) {
        this.net = net;
        this.batchSize = parameters.getBatchSize();
        this.data = data;
    }

    @Override
    public void newBatch() {
        batch = IntStream.range(0, batchSize).map(i -> random.nextInt(data.length)).toArray();
    }

    @Override
    public double calculateQuality(long[] weights) {
        double quality = Arrays.stream(batch).parallel().mapToDouble(element -> calculateResult(element, weights)).sum() / batchSize;
        return quality;
    }

    private double calculateResult(int element, long[] weights) {
        long[] calculate = net.calculate(weights, data[element].getInput());
        double confidence = (double) calculate[this.data[element].getOutput()] / NeuralNet.FACTOR;
        int found = IntStream.range(0, calculate.length)
                .reduce((a,b)->calculate[a]<calculate[b]? b: a)
                .getAsInt();
//        double wrongness = found == data[element].getOutput() ? 0 : -(double) calculate[found] / NeuralNet.FACTOR;
        double rightness = found == data[element].getOutput() ?1:0;
        double result = confidence/10 + rightness;
        return result;
    }

}
