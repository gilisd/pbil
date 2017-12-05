package be.digan.dl.pbil;

import java.util.Arrays;

public class FlattenMin implements Layer {


    @Override
    public long[] calculate(long[] weights, long[] input) {
        long min = Arrays.stream(input).min().getAsLong();
        double[] exp = Arrays.stream(input).mapToDouble(i -> i - min ).toArray();
        double sum = Arrays.stream(exp).sum();

        long[] collect = Arrays.stream(exp).mapToLong(i -> (long)((i / sum) * NeuralNet.FACTOR)).toArray();
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
