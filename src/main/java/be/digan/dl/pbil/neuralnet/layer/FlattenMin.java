package be.digan.dl.pbil.neuralnet.layer;

import be.digan.dl.pbil.neuralnet.NeuralNet;

import java.util.Arrays;

public class FlattenMin implements Layer {


    @Override
    public long[] calculate(long[] weights, long[] input) {
        long min = Arrays.stream(input).min().getAsLong();
        long max = Arrays.stream(input).max().getAsLong();
        long[] collect;
        if (max == min) {
            collect = Arrays.stream(input).map(i -> (long)(NeuralNet.FACTOR / input.length)).toArray();
        } else {
            double[] exp = Arrays.stream(input).mapToDouble(i -> (((double) i - min) / (max - min) / input.length)).toArray();
            double sum = Arrays.stream(exp).sum();
            collect = Arrays.stream(exp).mapToLong(i -> (long) (i / sum * NeuralNet.FACTOR)).toArray();
        }
        return collect;
    }

    @Override
    public int getWeightCount() {
        return 0;
    }


    @Override
    public int setWeightIndex(int weightIndex) {
        return weightIndex;
    }

    @Override
    public int[] appendWeightStructure(int[] baseStructure) {
        return baseStructure;
    }

    @Override
    public String toString() {
        return "FlattenMin()";
    }

}
