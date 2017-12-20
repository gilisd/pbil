package be.digan.dl.pbil.neuralnet.layer;

import be.digan.dl.pbil.neuralnet.NeuralNet;
import be.digan.dl.pbil.neuralnet.activation.ActivationFunction;

import java.util.stream.IntStream;

public class Node {
    private int weightCount;
    private ActivationFunction activation;
    private int weightIndex;

    public Node(int weightCount, ActivationFunction activation) {
        this.weightCount = weightCount;
        this.activation = activation;
    }

    public int getWeightCount() {
        return weightCount + 1;
    }

    public void setWeightIndex(int weightIndex) {
        this.weightIndex = weightIndex;
    }

    public Node clone() {
        return new Node(weightCount, activation);
    }

    public long calculate(final long[] weights, final long[] input) {
        long value = IntStream.range(0, input.length)
                .mapToLong(i -> input[i] * weights[weightIndex + i] / NeuralNet.FACTOR)
                .sum();// + weights[weightIndex + weightCount];
        return activation.activate(value);
    }

    public ActivationFunction getActivation() {
        return activation;
    }
}
