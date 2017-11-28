package be.digan.dl.pbil;

import java.util.stream.IntStream;

public class Node {
    private int weightCount;
    private int weightIndex;

    public Node(int weightCount) {
        this.weightCount = weightCount;
    }

    private Double relu(Double value) {
        return value<0?0:value;
    }

    public int getWeightCount() {
        return weightCount + 1;
    }

    public void setWeightIndex(int weightIndex) {
        this.weightIndex = weightIndex;
    }

    public Node clone() {
        return new Node(weightCount);
    }

    public double calculate(final double[] weights, final double[] input) {
        Double value = IntStream.range(0, input.length)
                .mapToDouble(i -> input[i] * weights[weightIndex + i])
                .sum() + weights[weightIndex + weightCount];
        return relu(value);
    }
}
